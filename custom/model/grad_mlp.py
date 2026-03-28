from custom.model.grad_lightning_model import LightningModelWithGrad
from p2pfl.examples.mnist.model.mlp_pytorch import MLP

class MLPWithGrad(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.saved_weights = None  # w_old
        self.computed_grads = None  # grad

    def on_fit_start(self):
        """
        Lưu weight trước khi bắt đầu training (toàn bộ fit)
        """
        self.saved_weights = [
            p.detach().cpu().clone()
            for p in self.parameters()
        ]

    def on_fit_end(self):
        """
        Sau khi train xong → compute gradient từ weight diff
        """
        if self.saved_weights is None:
            self.computed_grads = None
            return

        lr = self.lr_rate  # lấy learning rate từ model

        new_weights = [
            p.detach().cpu()
            for p in self.parameters()
        ]

        grads = []
        for w_old, w_new in zip(self.saved_weights, new_weights):
            grad = (w_old - w_new) / lr
            grads.append(grad)

        self.computed_grads = grads
        
# Export P2PFL model
def model_build_fn(*args, **kwargs) -> LightningModelWithGrad:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return LightningModelWithGrad(MLPWithGrad(*args, **kwargs), compression=compression)