import numpy as np
from p2pfl.utils.node_component import NodeComponent


class DualDimensionalEvaluator(NodeComponent):

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, k=5, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.eps = eps

    # =============================
    # PUBLIC API
    # =============================

    def evaluate(
        self,
        current_grad,
        grad_history,
        current_loss,
        prev_local_loss,
        prev_global_loss,
        global_grad,
    ):
        """
        Compute S_i^t and C_i^t
        """

        if len(grad_history) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        M, DI, S = self.__compute_sensitivity(current_grad, grad_history)

        LIR, GSI, C = self.__compute_contribution(
            current_grad,
            current_loss,
            prev_local_loss,
            prev_global_loss,
            global_grad,
            [self._gradient_variance(g) for g in grad_history]
        )

        return M, DI, S, LIR, GSI, C

    # =============================
    # SENSITIVITY
    # =============================

    def __compute_sensitivity(self, grad, grad_history):

        g_norm = self._gradient_norm(grad)

        history_norms = [
            self._gradient_norm(g) for g in grad_history[-self.k:]
        ]

        mu = self._moving_avg(history_norms)
        sigma = self._moving_std(history_norms)

        M = self._magnitude_instability(g_norm, mu, sigma)

        DI = self._direction_instability(grad, grad_history[-self.k:])

        return M, DI, self.alpha * M + self.beta * DI

    # Algorithm line 6
    def _gradient_norm(self, grad):

        flat = self._flatten_grad(grad)
        return np.linalg.norm(flat)

    # Algorithm line 7
    def _moving_avg(self, values):

        return np.mean(values)

    # Algorithm line 8
    def _moving_std(self, values):

        return np.std(values)

    # Algorithm line 9
    def _magnitude_instability(self, g_norm, mu, sigma):

        return np.tanh(abs(g_norm - mu) / (sigma + self.eps))

    # Algorithm line 10
    def _direction_instability(self, grad, grad_history):

        g1 = self._flatten_grad(grad)

        cos_vals = []

        for gh in grad_history:

            g2 = self._flatten_grad(gh)

            cos = np.dot(g1, g2) / (
                np.linalg.norm(g1) * np.linalg.norm(g2)
            )

            cos_vals.append(abs(cos))

        return 1 - np.mean(cos_vals)

    # =============================
    # CONTRIBUTION
    # =============================

    def __compute_contribution(
        self,
        grad,
        current_loss,
        prev_local_loss,
        prev_global_loss,
        global_grad,
        variance_history
    ):

        LIR = self._loss_improvement_ratio(
            prev_local_loss,
            current_loss,
            prev_global_loss
        )

        sigma_t = self._gradient_variance(grad)

        mu_sigma = self._moving_avg(variance_history[-self.k:])
        sigma_sigma = self._moving_std(variance_history[-self.k:])

        GSI = self._gradient_stability_indicator(
            grad,
            global_grad,
            sigma_t,
            mu_sigma,
            sigma_sigma
        )

        return LIR, GSI, np.maximum(0, LIR) * GSI

    # Algorithm line 14
    def _loss_improvement_ratio(self, prev_local, current, prev_global):

        return (prev_local - current) / (
            max(prev_global, prev_local) + self.eps
        )

    # Algorithm line 15
    def _gradient_variance(self, grad):

        flat = self._flatten_grad(grad)
        return np.var(flat)

    # Algorithm line 19
    def _gradient_stability_indicator(
        self,
        grad,
        global_grad,
        sigma_t,
        mu_sigma,
        sigma_sigma
    ):

        g1 = self._flatten_grad(grad)
        g2 = self._flatten_grad(global_grad)

        cos_sim = np.dot(g1, g2) / (
            np.linalg.norm(g1) * np.linalg.norm(g2) + self.eps
        )

        stability = np.exp(
            -self.gamma * abs(sigma_t - mu_sigma) /
            (sigma_sigma + self.eps)
        )

        return cos_sim * stability

    # =============================
    # UTIL
    # =============================

    def _flatten_grad(self, grad):

        return np.concatenate([g.flatten() for g in grad])