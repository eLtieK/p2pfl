import numpy as np
from p2pfl.utils.node_component import NodeComponent


class DualModeNoiseSelector(NodeComponent):
    def __init__(self, nu: float = 1.0, xi: float = 1e-8):
        self.nu = nu
        self.xi = xi

    def compute_indicator(
        self,
        S_all: list[float] | np.ndarray,
        C_all: list[float] | np.ndarray,
    ) -> float:
        S_all = np.asarray(S_all, dtype=float)
        C_all = np.asarray(C_all, dtype=float)

        S_mean = np.mean(S_all)
        S_std = np.std(S_all)

        C_mean = np.mean(C_all)
        C_std = np.std(C_all)

        indicator = (S_std / (S_mean + self.xi)) * np.exp(
            -(C_mean / (C_std + self.xi))
        )

        return float(indicator)

    def select_mode(
        self,
        indicator: float,
        indicator_history: list[float] | np.ndarray,
    ) -> tuple[str, float]:
        indicator = float(indicator)
        indicator_history = np.asarray(indicator_history, dtype=float)

        if len(indicator_history) == 0:
            drift_threshold = indicator
        else:
            drift_threshold = float(
                np.mean(indicator_history) +
                self.nu * np.std(indicator_history)
            )

        mode = "gaussian" if indicator >= drift_threshold else "laplace"
        return mode, drift_threshold