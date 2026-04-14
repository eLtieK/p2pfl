import random
import time
import json

import torch

from custom.aggregators.grad_fedavg import FedAvgWithGrad
from custom.component.gradient_inversion_attack import GradientInversionAttack
from custom.utils import build_output_dir, my_settings
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
# from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn
# from custom.model.simple_cnn_pytourch import model_build_fn
from custom.model.grad_mlp import model_build_fn
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_convergence, wait_to_finish


# ========================
# CONFIG
# ========================
NODES = 12
ROUNDS = 4
EPOCHS = 1
BATCH_SIZE = 32

BASE_CONFIG = {
    "dataset": "mnist",
    "nodes": NODES,
    "rounds": ROUNDS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}

OUTPUT_DIR = build_output_dir(
    BASE_CONFIG,
    prefix="normal"
)

ATTACK_DIR = build_output_dir(
    BASE_CONFIG,
    prefix="normal",
    root_dir="results_attacks"
)


def main():
    my_settings()
    start_time = time.time()

    # ========================
    # DATA
    # ========================
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    data.set_batch_size(BATCH_SIZE)

    partitions = data.generate_partitions(NODES, RandomIIDPartitionStrategy)

    # ========================
    # CREATE NODES
    # ========================

    nodes: list[Node] = []
    for i in range(NODES):
        node = Node(
            model_build_fn(),
            partitions[i],
            aggregator=FedAvgWithGrad(),
            protocol=MemoryCommunicationProtocol(),
            addr=f"node-{i}",
        )
        
        node.start()
        nodes.append(node)

    # ========================
    # CONNECT TOPOLOGY
    # ========================
    adjacency_matrix = TopologyFactory.generate_matrix(
        TopologyType.FULL, len(nodes)
    )
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    wait_convergence(nodes, NODES - 1, only_direct=False, wait=60)

    # ========================
    # TRAIN
    # ========================
    nodes[0].set_start_learning(rounds=ROUNDS, epochs=EPOCHS, trainset_size=NODES)

    wait_to_finish(nodes, timeout=3600)

    # ========================
    # SAVE RESULTS
    # ========================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    torch_model = nodes[0].get_model().get_model()
    torch.save(torch_model.state_dict(), OUTPUT_DIR / "final_model.pt")

    # Save global metrics
    global_logs = logger.get_global_logs()
    if global_logs:
        flattened = []
        for exp, nodes_logs in global_logs.items():
            for node_name, metrics in nodes_logs.items():
                for metric_name, values in metrics.items():
                    for round_num, value in values:
                        flattened.append(
                            {
                                "experiment": exp,
                                "node": node_name,
                                "metric": metric_name,
                                "round": round_num,
                                "value": value,
                            }
                        )

        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(flattened, f, indent=4)

    # Save execution time
    total_time = time.time() - start_time
    
    with open(OUTPUT_DIR / "time.txt", "w") as f:
        f.write(f"{total_time:.4f} seconds")

    print("✅ Training finished")
    print(f"Results saved in {OUTPUT_DIR}")

    # Stop nodes
    for node in nodes:
        node.stop()

if __name__ == "__main__":
    main()