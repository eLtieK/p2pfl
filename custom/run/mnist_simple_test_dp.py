from collections import defaultdict
import time
import json
from pathlib import Path

import torch

from custom.utils import build_output_dir, dp_settings
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
# from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn
from custom.model.grad_mlp import model_build_fn
from custom.aggregators.grad_fedavg import FedAvgWithGrad
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_convergence, wait_to_finish


# ========================
# CONFIG
# ========================
NODES = 12
ROUNDS = 50
EPOCHS = 1
BATCH_SIZE = 32

BASE_CONFIG = {
    "dataset": "mnist",
    "nodes": NODES,
    "rounds": ROUNDS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}

DP_CONFIG = {
    "dp": {
        "clip_norm": 10000,
        "epsilon": 20.0,
        "delta": 1e-5,
        "noise_type": "gaussian",
    }
}

USE_CFL = False
prefix = "cfl_dp" if USE_CFL else "dp"

OUTPUT_DIR = build_output_dir(
    BASE_CONFIG,
    extra_cfgs=[DP_CONFIG],
    prefix=prefix
)

def main():
    dp_settings()
    start_time = time.time()

    # ========================
    # DATA
    # ========================
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    data.set_batch_size(BATCH_SIZE)

    partitions = data.generate_partitions(NODES, RandomIIDPartitionStrategy)

    # ========================
    # CREATE NODES (WITH DP)
    # ========================
    nodes: list[Node] = []
        
    for i in range(NODES):
        
        is_server = (i == 0) if USE_CFL else False
        
        node = Node(
            model_build_fn(compression=DP_CONFIG),
            partitions[i],
            aggregator=FedAvgWithGrad(),
            protocol=MemoryCommunicationProtocol(),
            addr=f"node-{i}",
            
            is_cfl=USE_CFL,
            is_server=is_server,
        )
        node.start()
        nodes.append(node)

    # ========================
    # CONNECT TOPOLOGY
    # ========================
    if USE_CFL:
        adjacency_matrix = TopologyFactory.generate_matrix(
            TopologyType.STAR, len(nodes)
        )
    else:
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


    global_logs = logger.get_global_logs()
    if global_logs:
        flattened = []
        for exp, nodes_logs in global_logs.items():
            for node_name, metrics in nodes_logs.items():
                for metric_name, values in metrics.items():
                    for round_num, value in values:
                        flattened.append(
                            {
                                "node": node_name,
                                "metric": metric_name,
                                "round": round_num,
                                "value": value,
                            }
                        )

        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(flattened, f, indent=4)
    
    comm_logs = logger.get_messages(direction="all")
    
    comm_stats = defaultdict(lambda: {
        "bytes_sent": 0,
        "msgs_sent": 0,
        "bytes_received": 0,
        "msgs_received": 0,
    })

    for log in comm_logs:
        if log["direction"] == "sent":
            node = log["source"]

            comm_stats[node]["bytes_sent"] += log["package_size"]
            comm_stats[node]["msgs_sent"] += 1

        elif log["direction"] == "received":
            node = log["destination"]

            comm_stats[node]["bytes_received"] += log["package_size"]
            comm_stats[node]["msgs_received"] += 1

    # convert list
    comm_stats_list = [
        {"node": k, **v}
        for k, v in comm_stats.items()
    ]

    with open(OUTPUT_DIR / "communication.json", "w") as f:
        json.dump(comm_stats_list, f, indent=4)

    res_logs = []
    
    for node in nodes:
        res = node.monitor.get_stats()

        res_logs.append({
            "node": node.addr,

            "cpu_avg": res.get("cpu_avg", 0),
            "cpu_max": res.get("cpu_max", 0),

            "ram_avg": res.get("ram_avg", 0),
            "ram_max": res.get("ram_max", 0),

            "gpu_avg": res.get("gpu_avg", 0),
            "gpu_max": res.get("gpu_max", 0),

            "gpu_mem_avg": res.get("gpu_mem_avg", 0),
            "gpu_mem_max": res.get("gpu_mem_max", 0),

            "samples": res.get("samples", 0),
        })
        
    with open(OUTPUT_DIR / "resource.json", "w") as f:
        json.dump(res_logs, f, indent=4)


    total_time = time.time() - start_time
    with open(OUTPUT_DIR / "time.txt", "w") as f:
        f.write(f"{total_time:.4f} seconds")

    print("✅ Training with DP finished")
    print(f"Results saved in {OUTPUT_DIR}")

    for node in nodes:
        node.stop()

if __name__ == "__main__":
    main()