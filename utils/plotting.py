import io
import os
import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor

PHASE_CLASS_NAMES = {
    "p1": ["passaggio", "tiro", "altro"],
    "p2": ["tiroDaDue", "tiroDaTre", "tiroLibero"],
    "p3": ["sbagliato", "segnato"],
}

CLASS_NAMES = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
    "idle",
    "non-gioco",
]


def generate_confusion_matrix_figure(y_true, y_pred, class_names=CLASS_NAMES):
    """Generates a normalized confusion matrix plot using Seaborn."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # row normalization
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig_size = (6, 5) if len(class_names) <= 3 else (10, 8)
    fig, ax = plt.subplots(figsize=fig_size)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    return fig


def save_confusion_matrix_plot(y_true, y_pred, save_path, class_names=CLASS_NAMES):
    """Saves the confusion matrix figure to a .png file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = generate_confusion_matrix_figure(y_true, y_pred, class_names=class_names)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def get_confusion_matrix_tensor(y_true, y_pred, class_names=CLASS_NAMES):
    """Converts matplotlib confusion matrix figure to a PyTorch Tensor for

    TensorBoard.
    """
    fig = generate_confusion_matrix_figure(y_true, y_pred, class_names=class_names)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image_tensor = ToTensor()(image)
    plt.close(fig)
    return image_tensor