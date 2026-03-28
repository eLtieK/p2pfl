import torch
from torchmetrics import Accuracy, Metric
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed
import lightning as L


def fix_mnist_shape(x: torch.Tensor) -> torch.Tensor:
    # [B, 28, 28] → [B, 1, 28, 28]
    if x.dim() == 3:
        x = x.unsqueeze(1)

    # [1, B, 28, 28] → [B, 1, 28, 28]
    elif x.dim() == 4 and x.shape[1] != 1:
        x = x.permute(1, 0, 2, 3)

    return x


class SimpleCNN(L.LightningModule):
    def __init__(
        self,
        out_channels: int = 10,
        lr_rate: float = 0.001,
        dropout: float = 0.1,
        metric: type[Metric] = Accuracy,
    ):
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")

        self.lr_rate = lr_rate

        if out_channels == 1:
            self.metric = metric(task="binary")
        else:
            self.metric = metric(task="multiclass", num_classes=out_channels)

        # 🔹 Feature extractor
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 28x28 → 28x28
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                             # → 14x14

            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),  # → 14x14
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                             # → 7x7
        )

        # 🔹 Classifier
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 7 * 7, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fix_mnist_shape(x)
        x = self.conv(x)
        x = self.fc(x)
        return x  # ❌ không softmax

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_idx):
        x = batch["image"].float()
        y = batch["label"]

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["image"].float()
        y = batch["label"]

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", acc, prog_bar=True)

        return loss
    
# Export P2PFL model
def model_build_fn(*args, **kwargs) -> LightningModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return LightningModel(SimpleCNN(*args, **kwargs), compression=compression)