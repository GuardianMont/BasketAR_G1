from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np

class BaseProber(ABC):
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the prober on the features extracted by the backbone"""
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates and returns a dictionary of metrics"""
        pass


import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


class MLPProber(BaseProber):
    
    def __init__(
        self, 
        in_features: int, 
        num_classes: int, 
        num_layers: int = 2,
        hidden_dim: int = 256, 
        dropout: float = 0.3,
        lr: float = 1e-3, 
        weight_decay: float = 1e-4,
        epochs: int = 60,
        batch_size: int = 64
    ):
        if num_layers < 1:
            raise ValueError("'num_layers' should be at least 1")
            
        super().__init__(name=f"MLP_L{num_layers}_H{hidden_dim}")
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        layers = []
        
        if num_layers == 1:
            # degenerates to linear classifier
            layers.append(nn.Linear(in_features, num_classes))
        else:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                
            layers.append(nn.Linear(hidden_dim, num_classes))
            
        self.model = nn.Sequential(*layers)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                optimizer.step()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1).numpy()
            
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro"))
        }


from sklearn.neighbors import KNeighborsClassifier


class KNNProber(BaseProber):
    def __init__(self, k: int = 5, metric: str = 'cosine'):
        super().__init__(name=f"kNN_Probe_k{k}_{metric}")
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.knn.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.knn.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro"))
        }
    

from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor


class ProbingEvaluator:
    
    def __init__(self, probers: List[BaseProber]):
        self.probers = probers

    def run_prober(
            prober: BaseProber, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray
    ) -> Tuple[str, Dict[str, float]]:
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        prober.fit(X_tr_scaled, y_tr)
        metrics = prober.evaluate(X_te_scaled, y_te)
        
        return prober.name, metrics

    def evaluate_backbone_features(
            self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray    
    ) -> Dict[str, Any]:
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.run_prober, prober, X_train, y_train, X_test, y_test)
                for prober in self.probers
            ]
            for future in futures:
                name, metrics = future.result()
                results[name] = metrics
        return results
