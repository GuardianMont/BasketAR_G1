import os
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    """
    Tensorboard wrapper
    """
    def __init__(self, log_dir: str, experiment_name: str, run_id: int):
        # Generates an experiment folder: experiments/runs/EXP-01_Baseline_run1
        self.run_dir = os.path.join(log_dir, f"{experiment_name}_run{run_id}")
        self.writer = SummaryWriter(log_dir=self.run_dir)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float, val_f1: float):
        """
        Logs metrics at the end of each epoch.
        """
        self.writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
        
        self.writer.add_scalar("Validation/Accuracy", val_acc, epoch)
        self.writer.add_scalar("Validation/Macro_F1", val_f1, epoch)

    def log_image(self, tag: str, image_tensor, step: int):
        """Logs a PyTorch Image Tensor to TensorBoard."""
        self.writer.add_image(tag, image_tensor, global_step=step)

    def close(self):
        self.writer.flush()
        self.writer.close()