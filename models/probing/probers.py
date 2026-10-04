from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor


class BaseProber(ABC):
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the prober on the features extracted by the backbone."""
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Returns target predictions array for the given features."""
        pass


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

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1).numpy()
        return preds


class KNNProber(BaseProber):
    def __init__(self, k: int = 5, metric: str = 'cosine'):
        super().__init__(name=f"kNN_Probe_k{k}_{metric}")
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.knn.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.knn.predict(X_test)


# Note: this strategy doesn't allow batching.
# As long as k-NN or clustering will be used as prober it won't be possible
class ProbingEvaluator:
    """Evaluates a list of probers on the same dataset.
    Attributes:
        probers (List[BaseProber]): List of probers to evaluate.
    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: A dictionary where keys are \
            prober names and values are tuples of (true_labels, predicted_labels).    
    """
    def __init__(self, probers: List[BaseProber]):
        self.probers = probers

    def _run_prober(
        self, prober: BaseProber, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray
    ) -> Tuple[str, Dict[str, Any]]:
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        prober.fit(X_tr_scaled, y_tr)
        preds = prober.predict(X_te_scaled)
        
        return prober.name, y_te, preds

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray    
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._run_prober, prober, X_train, y_train, X_test, y_test)
                for prober in self.probers
            ]
            for future in futures:
                name, y_te, preds = future.result()
                results[name] = (y_te, preds)
        return results