"""Federated Averaging (FedAvg) Aggregator."""

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


class FedAvgWithGrad(Aggregator):

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Create a Zero Model using numpy
        first_grads = models[0].encode_gradients()
        accum = [np.zeros_like(g) for g in first_grads]

        # Add weighted models
        for m in models:
            grads = m.encode_gradients()
            weight = m.get_num_samples()

            for i, g in enumerate(grads):
                if g is None:
                    continue
                accum[i] += g * weight

        # Normalize Accum
        accum = [g / total_samples for g in accum]
        
        # 2. LẤY WEIGHT HIỆN TẠI
        base_model = models[0] 
        last_weights = base_model.get_last_parameters()

        lr = base_model.model.lr_rate

        # 3. UPDATE = SGD
        new_weights = []
        for w, g in zip(last_weights, accum):
            if g is None:
                new_weights.append(w)
            else:
                new_weights.append(w - lr * g)

        # Get contributors
        contributors: list[str] = []
        for m in models:
            contributors = contributors + m.get_contributors()

        # Return an aggregated p2pfl model
        return models[0].build_copy(params=new_weights, num_samples=total_samples, contributors=contributors)
