import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

from models.pipeline import ARPipeline
from data.dataset import ActionRecognitionDataset
from data.utils import pad_and_collate_video_batch
from train import load_preextracted_features, EXPERIMENTS

def evaluate_best_model_on_test(best_checkpoint_path, best_config, test_pt_path, seeds=[42, 69, 104]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluator running on device: {device}")
    
    test_samples = load_preextracted_features(test_pt_path)
    test_dataset = ActionRecognitionDataset(samples=test_samples, is_feature_dataset=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=pad_and_collate_video_batch
    )
    
    accuracies, macro_f1s = [], []
    
    # Run evaluation across seeds
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        backbone = best_config['components']['backbone']({})
        neck = best_config['components']['neck']({})
        head = best_config['components']['head']({})
        
        model = ARPipeline(backbone=backbone, neck=neck, head=head).to(device)
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                labels = batch["labels"].to(device)
                lengths = batch["lengths"]
                
                logits = model(x, lengths=lengths)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        accuracies.append(acc)
        macro_f1s.append(f1)
        
    print("\n" + "="*60)
    print(" FINAL TEST SET EVALUATION RESULTS ")
    print("="*60)
    print(f"Accuracy:  {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"Macro F1:  {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")
    print("\nDetailed Classification Report (Last Seed Run):")
    print(classification_report(all_labels, all_preds, zero_division=0))

if __name__ == "__main__":
    BEST_CHECKPOINT = "checkpoints/EXP-02_BiGRU_Neck_run1_best.pt"
    BEST_CONFIG = EXPERIMENTS["EXP-02_BiGRU_Neck"]
    TEST_PT_PATH = "features/test_features.pt"
    
    evaluate_best_model_on_test(BEST_CHECKPOINT, BEST_CONFIG, TEST_PT_PATH)