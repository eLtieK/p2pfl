from typing import Any
import numpy as np

import lightning as L

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel


class LightningModelWithGrad(LightningModel):
    """
    Extension của LightningModel để support lấy gradient.
    """

    def __init__(
        self,
        model: L.LightningModule,
        params: list[np.ndarray] | bytes | None = None,
        num_samples: int | None = None,
        contributors: list[str] | None = None,
        additional_info: dict[str, Any] | None = None,
        compression: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(model, params, num_samples, contributors, additional_info, compression)

    def get_gradients(self) -> list[np.ndarray]:
        """
        Lấy gradient đã được lưu từ LightningModule (computed_grads).

        Returns:
            List[np.ndarray]: gradient của từng layer
        """
        grads = getattr(self.model, "computed_grads", None)

        if grads is None:
            raise ValueError("Gradients not available.")

        return [
            g.cpu().numpy() if g is not None else None
            for g in grads
        ]
        
    def get_last_parameters(self) -> list[np.ndarray]:
        weights = getattr(self.model, "saved_weights", None)

        if weights is None:
            raise ValueError("Pre-train weights not available.")

        return [
            w.cpu().numpy() if hasattr(w, "cpu") else w
            for w in weights
        ]