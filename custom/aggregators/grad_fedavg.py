"""Federated Averaging (FedAvg) Aggregator."""

import numpy as np

from custom.component.gradient_inversion_attack import GradientInversionAttack
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


class FedAvgWithGrad(Aggregator):

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, attacker: GradientInversionAttack =None, state: NodeState = None, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)
        self.attacker = attacker
        self.state = state
        self._is_final_round: bool = False
        
    def set_attacker(self, attacker):
        self.attacker = attacker
        
    def set_state(self, state):
        self.state = state

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
        
        if self.attacker is not None:
            round_idx = self.state.round if self.state else -1
            total_rounds = self.state.total_rounds if self.state else -1
            
            if total_rounds > 0 and round_idx == total_rounds - 1:
                self._is_final_round = True
            
        for idx, m in enumerate(models):
            grads = m.encode_gradients()
            weight = m.get_num_samples()
            
            if self.attacker is not None and self._is_final_round:
                try:
                    logger.info(self.addr, "🚨 Attacking gradients from client")

                    self.attacker.reconstruct(
                        model=m.model,          # model local
                        gradients=grads,        # gradient thật
                        gt_shape=(1, 28, 28),
                        num_classes=10,
                        client_id=self.addr + "_" + str(idx)
                    )

                except Exception as e:
                    logger.error(self.addr, f"[ATTACK ERROR] {e}")

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
