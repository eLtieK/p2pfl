try:
    import opendp.prelude as dp
except ImportError as err:
    raise ImportError("Please install with `pip install p2pfl[dp]`") from err

import numpy as np

from p2pfl.learning.compression.base_compression_strategy import TensorCompressor
from p2pfl.management.logger import logger

dp.enable_features("contrib")

class DualSwitchDPCompressor(TensorCompressor):
    
    def _get_noise_scale(self, noise_type, clip_norm, epsilon, delta):
        if noise_type == "laplace":
            scale = clip_norm / epsilon
        elif noise_type == "gaussian":
            scale = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        else:
            raise ValueError("noise_type must be 'gaussian' or 'laplace'")
        return scale
    
    def _get_noise_scale_dual(
        self,
        noise_type,
        clip_norm,
        epsilon,
        alpha,
        S_i,
        C_i,
        S_all,
        C_all,
        stability_constant=1e-8,
    ):
        max_S = max(S_all) if len(S_all) > 0 else 0
        max_C = max(C_all) if len(C_all) > 0 else 0
        
        delta1_f = clip_norm
        delta2_f = clip_norm

        # =========================
        # Step 1: compute b_i (Laplace)
        # =========================
        modulation_C = 1 + (C_i / (max_C + stability_constant))
        b_i = (delta1_f / (epsilon + stability_constant)) * modulation_C

        # =========================
        # Step 2: compute rho_i from b_i
        # =========================
        rho_i = (alpha * (delta1_f ** 2)) / (2 * (b_i ** 2) + stability_constant)


        # =========================
        # Gaussian (σ)
        # =========================
        if noise_type == "gaussian":
            
            numerator = 2 * alpha * (delta2_f ** 2)
            denominator = 2 * alpha * rho_i - 1

            base_sigma2 = numerator / denominator

            modulation = 1 - (S_i / (max_S + stability_constant))

            sigma2 = base_sigma2 * modulation
            sigma = np.sqrt(max(sigma2, 0.0))

            return sigma

        # =========================
        # Laplace (b)
        # =========================
        elif noise_type == "laplace":
            return b_i

        else:
            raise ValueError("noise_type must be 'gaussian' or 'laplace'")
        
    def _compute_rho_from_laplace(
        self,
        b_i,
        clip_norm,
        alpha,
        stability_constant=1e-8,
    ):
        delta1_f = clip_norm

        rho = (alpha * (delta1_f ** 2)) / (2 * (b_i ** 2) + stability_constant)

        return rho

    def apply_strategy(
        self,
        params: list[np.ndarray],
        additional_info: dict | None = None,
        clip_norm: float = 1.0,
        delta: float = 1e-5,
        alpha: float = 1,
        scale_mode: str = "standard", 
        stability_constant: float = 1e-6,
    ) -> tuple[list[np.ndarray], dict]:
        if not params:
            raise ValueError("DifferentialPrivacyCompressor: list 'params' must not be empty")

        additional_info = additional_info or {}
        dual_info = additional_info.get("dual_evaluation", {})

        epsilon = dual_info.get("epsilon")
        selected_noise_type = dual_info.get("noise_type", {})
        
        self_score = dual_info.get("self_score", {})
        all_scores = dual_info.get("all_scores", {})

        S_i = float(self_score.get("S", 0.0))
        C_i = float(self_score.get("C", 0.0))

        S_all = list(all_scores.get("S", {}).values())
        C_all = list(all_scores.get("C", {}).values())

        if epsilon is None:
            logger.info("Check", "[DP] ❌ Skipped - epsilon not available")
            return params, {
                "dp_applied": False,
                "reason": "epsilon_not_available"
            }

        epsilon = float(epsilon)
        logger.info("Check", f"[DP] Applying | eps={epsilon}, noise={selected_noise_type}")

        # Step 1: Flatten all params and compute global L2 norm
        flat_update = np.concatenate([p.flatten() for p in params])
        total_norm = np.linalg.norm(flat_update, ord=2)  # L2

        # Step 2: Clip if needed
        if total_norm > clip_norm:
            clip_factor = clip_norm / (total_norm + stability_constant)
            clipped_flat_update = flat_update * clip_factor
        else:
            clip_factor = 1.0
            clipped_flat_update = flat_update.copy()

        # Step 3: Add DP noise
        # mech, scale = self._get_noise_mechanism(
        #     noise_type=selected_noise_type,
        #     clip_norm=clip_norm,
        #     epsilon=epsilon,
        #     delta=delta,
        #     vec_len=clipped_flat_update.size,
        # )
        # noisy_flat_update = mech(clipped_flat_update.tolist())
        if scale_mode == "standard":
            scale = self._get_noise_scale(
                selected_noise_type,
                clip_norm,
                epsilon,
                delta
            )
        elif scale_mode == "dual":
            scale = self._get_noise_scale_dual(
                noise_type=selected_noise_type,
                clip_norm=clip_norm,
                epsilon=epsilon,
                alpha=alpha,
                S_i=S_i,
                C_i=C_i,
                S_all=S_all,
                C_all=C_all,
                stability_constant=stability_constant,
            )
        else:
            raise ValueError("scale_mode must be 'standard' or 'dual'")
        
        if selected_noise_type == "gaussian":
            noise = np.random.normal(0, scale, size=flat_update.shape)
        else:  # laplace
            noise = np.random.laplace(0, scale, size=flat_update.shape)
        noisy_flat_update = clipped_flat_update + noise

        # Step 4: Unflatten
        dp_params = []
        current_pos = 0
        for p in params:
            size = p.size
            shape = p.shape
            dtype = p.dtype

            chunk = noisy_flat_update[current_pos: current_pos + size]
            dp_params.append(np.array(chunk, dtype=dtype).reshape(shape))
            current_pos += size

        # Step 5: Metadata
        dp_info = {
            "dp_applied": True,
            "clip_norm": float(clip_norm),
            "epsilon": epsilon,
            "delta": float(delta) if selected_noise_type == "gaussian" else None,
            "noise_type": selected_noise_type,
            "noise_scale": float(scale),
            "original_norm": float(total_norm),
            "clip_factor": float(clip_factor),
            "was_clipped": bool(total_norm > clip_norm),
        }

        return dp_params, dp_info

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        # DP is irreversible by design
        return params