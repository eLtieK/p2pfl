import numpy as np

from p2pfl.management.logger import logger


class SimpleCompressionManager:

    @staticmethod
    def apply(
        params: list[np.ndarray],
        additional_info: dict,
        techniques: dict[str, dict]
    ) -> list[np.ndarray]:
        """
        Apply techniques như filter pipeline.
        Không encode, không reverse.
        """

        from p2pfl.learning.compression import COMPRESSION_STRATEGIES_REGISTRY

        registry = COMPRESSION_STRATEGIES_REGISTRY

        for name, fn_params in techniques.items():
            if name not in registry:
                raise ValueError(f"Unknown technique: {name}")

            instance = registry[name]()

            # skip nếu flag tắt
            if ("apply_compression" in additional_info) and not additional_info["apply_compression"]:
                logger.info("[Compression]", f" ⚠️ Skipping compression for {name} due to additional_info flag")
                continue
                
            params, compression_settings = instance.apply_strategy(
                params,
                additional_info=additional_info,
                **fn_params
            )
                
            logger.info("[Compression]", f" ✅ Done: {name} | settings={compression_settings}")
            
        return params