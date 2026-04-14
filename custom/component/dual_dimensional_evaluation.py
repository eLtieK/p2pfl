import numpy as np
from p2pfl.utils.node_component import NodeComponent
from p2pfl.management.logger import logger


class DualDimensionalEvaluator(NodeComponent):

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, k=5, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.eps = eps

    def evaluate(
        self,
        current_grad,
        grad_history,
        current_loss,
        prev_local_loss,
        prev_global_loss,
        global_grad,
    ):

        if len(grad_history) == 0:
            logger.info(self.addr, "⚠️ No grad history → return 0")
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

        logger.info(self.addr,
            f"📊 Dual Eval | "
            f"M={M:.4f}, DI={DI:.4f} → S={S:.4f} | "
            f"LIR={LIR:.4f}, GSI={GSI:.4f} → C={C:.4f}"
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

        logger.info(self.addr,
            f"📐 Norm Stats | g_norm={g_norm:.6f} | history_norms[:3]={history_norms[:3]}"
        )

        M = self._magnitude_instability(g_norm, mu, sigma)
        DI = self._direction_instability(grad, grad_history[-self.k:])

        logger.info(self.addr,
            f"📐 Sensitivity Detail | "
            f"|g_norm - μ|={abs(g_norm - mu):.6f}, σ={sigma:.6f}"
        )

        logger.info(self.addr,
            f"📐 Sensitivity | "
            f"M={M:.4f}, DI={DI:.4f}, S={self.alpha*M + self.beta*DI:.4f}"
        )

        return M, DI, self.alpha * M + self.beta * DI

    def _gradient_norm(self, grad):
        flat = self._flatten_grad(grad)
        norm = np.linalg.norm(flat)
        # logger.info(self.addr, f"🔢 Gradient Norm | size={flat.shape}, norm={norm:.6f}")
        return norm

    def _moving_avg(self, values):
        avg = np.mean(values)
        # logger.info(self.addr, f"📊 Moving Avg | values[:3]={values[:3]}, mean={avg:.6f}")
        return avg

    def _moving_std(self, values):
        std = np.std(values)
        # logger.info(self.addr, f"📊 Moving Std | std={std:.6f}")
        return std

    def _magnitude_instability(self, g_norm, mu, sigma):
        raw = abs(g_norm - mu) / (sigma + self.eps)
        val = np.tanh(raw)
        logger.info(self.addr,
            f"📈 Magnitude Instability | raw={raw:.6f} → tanh={val:.6f}"
        )
        return val

    def _direction_instability(self, grad, grad_history):

        g1 = self._flatten_grad(grad)
        cos_vals = []

        for i, gh in enumerate(grad_history):
            g2 = self._flatten_grad(gh)

            cos = np.dot(g1, g2) / (
                np.linalg.norm(g1) * np.linalg.norm(g2)
            )

            logger.info(self.addr,
                f"🧭 Cos[{i}] | dot={np.dot(g1,g2):.6f}, "
                f"||g1||={np.linalg.norm(g1):.6f}, ||g2||={np.linalg.norm(g2):.6f}, cos={cos:.6f}"
            )

            cos_vals.append(abs(cos))

        di = 1 - np.mean(cos_vals)

        logger.info(self.addr,
            f"🧭 Direction | mean|cos|={np.mean(cos_vals):.6f} → DI={di:.6f}"
        )

        return di

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

        logger.info(self.addr,
            f"📉 Variance Stats | σ_t={sigma_t:.6f}, μσ={mu_sigma:.6f}, σσ={sigma_sigma:.6f}"
        )

        GSI = self._gradient_stability_indicator(
            grad,
            global_grad,
            sigma_t,
            mu_sigma,
            sigma_sigma
        )

        C = np.maximum(0, LIR) * GSI

        logger.info(self.addr,
            f"📉 Contribution | LIR={LIR:.6f}, GSI={GSI:.6f}, C={C:.6f}"
        )

        return LIR, GSI, C

    def _loss_improvement_ratio(self, prev_local, current, prev_global):
        numerator = prev_local - current
        denominator = max(prev_global, prev_local) + self.eps
        val = numerator / denominator

        logger.info(self.addr,
            f"📉 LIR Detail | numerator={numerator:.6f}, denominator={denominator:.6f}, LIR={val:.6f}"
        )

        return val

    def _gradient_variance(self, grad):
        flat = self._flatten_grad(grad)
        var = np.var(flat)
        # logger.info(self.addr,
        #     f"📊 Gradient Variance | size={flat.shape}, var={var:.6f}"
        # )
        return var

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

        dot = np.dot(g1, g2)
        norm1 = np.linalg.norm(g1)
        norm2 = np.linalg.norm(g2)

        cos_sim = dot / (norm1 * norm2)

        raw = abs(sigma_t - mu_sigma) / (sigma_sigma + self.eps)
        stability = np.exp(-self.gamma * raw)

        logger.info(self.addr,
            f"🧩 GSI Detail | dot={dot:.6f}, ||g1||={norm1:.6f}, ||g2||={norm2:.6f}, cos={cos_sim:.6f}"
        )

        logger.info(self.addr,
            f"🧩 Stability Detail | raw={raw:.6f}, exp={stability:.6f}"
        )

        gsi = cos_sim * stability

        logger.info(self.addr,
            f"🧩 GSI | cos_sim={cos_sim:.6f}, stability={stability:.6f} → GSI={gsi:.6f}"
        )

        return gsi

    def _flatten_grad(self, grad):
        flat = np.concatenate([g.flatten() for g in grad])
        return flat