#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Train stage."""

import time
from typing import Any

import numpy as np

from p2pfl.communication.commands.message.metrics_command import MetricsCommand
from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.stage import EarlyStopException, Stage, check_early_stop
from p2pfl.stages.stage_factory import StageFactory

from custom.component.dual_dimensional_evaluation import DualDimensionalEvaluator
from custom.component.privacy_budget_allocator import PrivacyBudgetAllocator
from custom.command.node_score_command import NodeScoreCommand
from custom.component.dual_mode_noise_selector import DualModeNoiseSelector


class TrainStage(Stage):
    """Train stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "TrainStage"

    @staticmethod
    def execute(
        state: NodeState | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        learner: Learner | None = None,
        aggregator: Aggregator | None = None,
        evaluator: DualDimensionalEvaluator | None = None,
        allocator: PrivacyBudgetAllocator | None = None,
        noise_selector: DualModeNoiseSelector | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if state is None or communication_protocol is None or aggregator is None or learner is None:
            raise Exception("Invalid parameters on TrainStage.")
        
        use_dual = True if evaluator and allocator and noise_selector else False

        try:
            if use_dual: 
                TrainStage.__reset_scores(state)
                
            check_early_stop(state)

            # Set Models To Aggregate
            aggregator.set_nodes_to_aggregate(state.train_set)

            check_early_stop(state)
        
            # Evaluate and send metrics
            TrainStage.__save_global_loss(state, learner, communication_protocol)

            check_early_stop(state)

            # Save global weights before training
            last_global_weights = learner.get_model().get_parameters()
            
            # Train
            logger.info(state.addr, "🏋️‍♀️ Training...")
            learner.fit()
            logger.info(state.addr, "🎓 Training done.")
            
            # Local gradient and loss
            new_local_weights = learner.get_model().get_parameters()
            TrainStage.__save_local_gradient(state, last_global_weights, new_local_weights, learner)
            TrainStage.__save_local_loss(state, learner)

            check_early_stop(state)
            
            # Dual evaluation
            if use_dual: 
                M, DI, S, LIR, GSI, C = TrainStage.__dual_evaluation(state, evaluator)
                
                with state.score_lock:
                    state.score_S[state.addr] = S
                    state.score_C[state.addr] = C
                    state.score_GSI[state.addr] = GSI
                
                communication_protocol.broadcast(
                    communication_protocol.build_msg(
                        NodeScoreCommand.get_name(),
                        [str(S), str(C), str(GSI)],
                        round=state.round
                    )
                )
                
                # Wait score from net work
                TrainStage.__wait_scores_from_network(state)
                
                # Privacy budget allocator
                epsilon = allocator.allocate(
                    self_S=S,
                    self_C=C,
                    self_GSI=GSI,
                    self_LIR=LIR,
                    all_S=list(state.score_S.values()),
                    all_C=list(state.score_C.values()),
                    all_GSI=list(state.score_GSI.values())
                )
                
                logger.info(state.addr, f"🔬 Epsilon Evaluated: {epsilon}")
                
                # Select noise type
                indicator = noise_selector.compute_indicator(
                    S_all=list(state.score_S.values()),
                    C_all=list(state.score_C.values())
                )
                
                noise_type, phi_drift = noise_selector.select_mode(
                    indicator=indicator,
                    indicator_history=state.behavior_indicator_history
                )
                
                state.behavior_indicator_history.append(indicator)

                logger.info(
                    state.addr,
                    f"🔀 Noise selected: {noise_type} | indicator={indicator:.6f} | threshold={phi_drift:.6f}"
                )
            
            # Add info for compression
            model = learner.get_model()
            
            if use_dual: 
                TrainStage.__add_info_for_compression(
                    state=state,
                    S=S,
                    C=C,
                    epsilon=epsilon,
                    noise_type=noise_type,
                    model=model
                )

            # Aggregate Model
            models_added = aggregator.add_model(learner.get_model())

            # send model added msg ---->> redundant (a node always owns its model)
            # TODO: print("Broadcast redundante")
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    ModelsAggregatedCommand.get_name(),
                    models_added,
                    round=state.round,
                )
            )
            TrainStage.__gossip_model_aggregation(state, communication_protocol, aggregator)

            check_early_stop(state)

            # Set aggregated model
            agg_model = aggregator.wait_and_get_aggregation()
            learner.set_model(agg_model)
            
            # Global gradient
            new_global_weights = learner.get_model().get_parameters()
            TrainStage.__save_global_gradient(state, last_global_weights, new_global_weights, learner)

            # Share that aggregation is done
            communication_protocol.broadcast(communication_protocol.build_msg(ModelsReadyCommand.get_name(), [], round=state.round))

            # Next stage
            return StageFactory.get_stage("GossipModelStage")
        except EarlyStopException:
            return None

    @staticmethod
    def __dual_evaluation(state: NodeState, evaluator: DualDimensionalEvaluator):
        current_grad = state.local_gradient_history[-1]
        grad_history = state.local_gradient_history[:-1]
        current_loss = state.local_loss_history[-1]
        prev_local_loss = state.local_loss_history[-2] if len(state.local_loss_history) > 1 else current_loss
        prev_global_loss = state.global_loss_history[-1] if len(state.global_loss_history) > 0 else current_loss
        global_grad = state.global_gradient_history[-1] if len(state.global_gradient_history) > 0 else current_grad
            
        M, DI, S, LIR, GSI, C = evaluator.evaluate(
                current_grad,
                grad_history,
                current_loss,
                prev_local_loss,
                prev_global_loss,
                global_grad,
            )
        
        return M,DI,S,LIR,GSI,C

    @staticmethod
    def __add_info_for_compression(state: NodeState, S, C, epsilon, noise_type, model: P2PFLModel):
        round_info = {
            "node": state.addr,
            "round": state.round,
            "epsilon": epsilon,
            "noise_type": noise_type,
            "self_score": {
                "S": S,
                "C": C,
            },
            "all_scores": {
                "S": dict(state.score_S),
                "C": dict(state.score_C),
            },
        }

        model.add_info("dual_evaluation", round_info)

    @staticmethod
    def __save_global_gradient(
                        state: NodeState,
                        last_global_weights: list[np.ndarray],
                        new_global_weights: list[np.ndarray],
                        learner: Learner):
            
        new_global_grad = TrainStage.__compute_gradient(
            last_global_weights,
            new_global_weights,
            learner.get_model().model.lr_rate
        )

        state.global_gradient_history.append(new_global_grad)# test for mlp_pytorch
        # logger.info(state.addr, f"🔬 New global gradient Evaluated: {new_global_grad}")
        
    @staticmethod
    def __save_local_gradient(
                        state: NodeState,
                        last_global_weights: list[np.ndarray],
                        new_local_weights: list[np.ndarray],
                        learner: Learner):
            
        new_local_grad = TrainStage.__compute_gradient(
            last_global_weights,
            new_local_weights,
            learner.get_model().model.lr_rate
        )

        state.local_gradient_history.append(new_local_grad)# test for mlp_pytorch
        # logger.info(state.addr, f"🔬 New local gradient Evaluated: {new_local_grad}")
            
    @staticmethod
    def __save_local_loss(state: NodeState, learner: Learner):
        metrics = TrainStage.__evaluate_local(state, learner)
            
            # Get local loss
        local_loss = metrics.get("test_loss")

        if local_loss is not None:
            state.local_loss_history.append(local_loss)
            # logger.info(state.addr, f"🔬 New local loss Evaluated: {local_loss}")
            
    @staticmethod
    def __save_global_loss(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol):
        metrics = TrainStage.__evaluate(state, learner, communication_protocol)
            
        # Get local loss
        global_loss = metrics.get("test_loss")

        if global_loss is not None:
            state.global_loss_history.append(global_loss)
            # logger.info(state.addr, f"🔬 New global loss Evaluated: {global_loss}")

    @staticmethod
    def __evaluate(state: NodeState, learner: Learner, communication_protocol: CommunicationProtocol):
        logger.info(state.addr, "🔬 Evaluating...")
        results = learner.evaluate()
        logger.info(state.addr, f"📈 Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(state.addr, "📢 Broadcasting metrics.")
            flattened_metrics = [str(item) for pair in results.items() for item in pair]
            communication_protocol.broadcast(
                communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=state.round,
                )
            )
            
        return results
    
    @staticmethod
    def __evaluate_local(state: NodeState, learner: Learner):
        logger.info(state.addr, "🔬 Evaluating Local...")
        results = learner.evaluate()
        logger.info(state.addr, f"📈 Local Evaluated. Results: {results}")
        # Send metrics
        return results

    @staticmethod
    def __gossip_model_aggregation(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        aggregator: Aggregator,
    ) -> None:
        """
        Gossip model aggregation.

        CAREFULL:
            - Full connected trainset to increase aggregation speed. On real scenarios, this won't
            be possible, private networks and firewalls.
            - Needed because the trainset can split the networks (and neighbors that are not in the
            trainset won't receive the aggregation).
        """

        # Anonymous functions
        def early_stopping_fn():
            return state.round is None

        def get_candidates_fn() -> list[str]:
            candidates = set(state.train_set) - {state.addr}
            return [n for n in candidates if len(TrainStage.__get_remaining_nodes(n, state)) != 0]

        def status_fn() -> Any:
            return [
                (
                    n,
                    TrainStage.__get_aggregated_models(n, state),
                )  # reemplazar por Aggregator - borrarlo de node
                for n in communication_protocol.get_neighbors(only_direct=False)
                if (n in state.train_set)
            ]

        def model_fn(node: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            try:
                model = aggregator.get_model(TrainStage.__get_aggregated_models(node, state))
            except NoModelsToAggregateError:
                logger.debug(state.addr, f"❔ No models to aggregate for {node}.")
                return (
                    None,
                    PartialModelCommand.get_name(),
                    state.round,
                    [],
                )
            model_msg = communication_protocol.build_weights(
                PartialModelCommand.get_name(),
                state.round,
                model.encode_parameters(),
                model.get_contributors(),
                model.get_num_samples(),
            )
            return (
                model_msg,
                PartialModelCommand.get_name(),
                state.round,
                model.get_contributors(),
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True,
        )

    @staticmethod
    def __get_aggregated_models(node: str, state: NodeState) -> list[str]:
        try:
            return state.models_aggregated[node]
        except KeyError:
            return []

    @staticmethod
    def __get_remaining_nodes(node: str, state: NodeState) -> set[str]:
        return set(state.train_set) - set(TrainStage.__get_aggregated_models(node, state))
    
    @staticmethod
    def __compute_gradient(old_weights: list[np.ndarray],
                     new_weights: list[np.ndarray],
                     lr: float | None = None) -> list[np.ndarray]:
        """
        Compute gradient/update between two model weight lists.

        Args:
            old_weights: weights before update
            new_weights: weights after update
            lr: learning rate (optional)

        Returns:
            list of gradients or updates
        """

        gradients = []

        for w_old, w_new in zip(old_weights, new_weights):
            if lr is None:
                # return model update
                grad = w_new - w_old
            else:
                # return gradient estimate
                grad = (w_old - w_new) / lr

            gradients.append(grad)

        return gradients
    
    @staticmethod
    def __wait_scores_from_network(state: NodeState):

        while (
            len(state.score_S) < len(state.train_set)
            or len(state.score_C) < len(state.train_set)
        ):
            time.sleep(1)
            
    @staticmethod
    def __reset_scores(state: NodeState):
        """Reset all node scores at the beginning of each round."""
        with state.score_lock:
            state.score_S.clear()
            state.score_C.clear()
            state.score_GSI.clear()
        
        logger.info(state.addr, "🔄 Scores reset for new round")
