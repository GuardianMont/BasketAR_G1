import os
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from utils.logger import ExperimentLogger
from utils.plotting import (
    CLASS_NAMES,
    PHASE_CLASS_NAMES,
    get_confusion_matrix_tensor,
    save_confusion_matrix_plot,
)


class Trainer:
    """Incapsulate training, validation and early stopping logic"""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def fit(
        self,
        train_loader,
        val_loader,
        phase: str,
        epochs: int,
        patience: int,
        tb_logger: ExperimentLogger,
        checkpoint_path: str,
        plot_path: str,
        desc: str = "Training",
    ) -> dict:
        label_key = f"label_{phase}"
        best_f1 = 0.0
        best_loss = float("inf")
        best_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        best_preds, best_labels = [], []

        epoch_pbar = tqdm(range(1, epochs + 1), desc=desc, unit="ep", leave=True)

        for epoch in epoch_pbar:
            # --- TRAINING ---
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch["x"].to(self.device)
                labels = batch[label_key].to(self.device)
                lengths = batch["lengths"]

                self.optimizer.zero_grad()
                logits = self.model(x, lengths=lengths)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # --- VALIDATION ---
            self.model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(self.device)
                    labels = batch[label_key].to(self.device)
                    lengths = batch["lengths"]

                    logits = self.model(x, lengths=lengths)
                    loss = self.criterion(logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            tb_logger.log_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_acc=acc,
                val_f1=macro_f1,
            )

            epoch_pbar.set_postfix({
                "TrLoss": f"{avg_train_loss:.4f}",
                "ValLoss": f"{avg_val_loss:.4f}",
                "ValAcc": f"{acc:.4f}",
                "ValF1": f"{macro_f1:.4f}",
            })

            is_improved = (avg_val_loss < best_loss)
            

            if is_improved:
                best_f1, best_loss, best_acc, best_epoch = macro_f1, avg_val_loss, acc, epoch
                best_preds, best_labels = all_preds, all_labels
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epoch_pbar.write(f"    [EARLY STOPPING - {phase.upper()}] Best Ep: {best_epoch}")
                    break

        if best_preds and best_labels:
            phase_class_names = PHASE_CLASS_NAMES.get(phase, CLASS_NAMES)
            save_confusion_matrix_plot(best_labels, best_preds, plot_path, class_names=phase_class_names)
            cm_tensor = get_confusion_matrix_tensor(best_labels, best_preds, class_names=phase_class_names)
            tb_logger.log_image(f"Confusion_Matrix/{phase.upper()}_Best", cm_tensor, step=best_epoch)

        return {
            "best_epoch": best_epoch,
            "val_accuracy": round(best_acc, 4),
            "val_macro_f1": round(best_f1, 4),
            "val_loss": round(best_loss, 4),
        }