from p2pfl.management.logger import logger
from p2pfl.communication.commands.command import Command
from p2pfl.node_state import NodeState

class NodeScoreCommand(Command):

    def __init__(self, state: NodeState):
        self.state = state

    @staticmethod
    def get_name():
        return "NodeScore"

    def execute(self, sender: str, round: int, S, C, GSI):

        S = float(S)
        C = float(C)
        GSI = float(GSI)

        self.state.score_lock.acquire()

        self.state.score_S[sender] = S
        self.state.score_C[sender] = C
        self.state.score_GSI[sender] = GSI

        self.state.score_lock.release()
        
        logger.info(self.state.addr, f"Received score S={S:.4f} C={C:.4f} GSI={GSI:.4f} from {sender}")