import numpy as np
from p2pfl.utils.node_component import NodeComponent


class PrivacyBudgetAllocator(NodeComponent):

    def __init__(
        self,
        epsilon_base=20,
        epsilon_min=5,
        lambda_protection=4.0,
        stability_constant=1e-8
    ):

        self.epsilon_base = epsilon_base
        self.epsilon_min = epsilon_min
        self.lambda_protection = lambda_protection
        self.xi = stability_constant

    def allocate(
        self,
        self_S: float,
        self_C: float,
        self_GSI: float,
        self_LIR: float,
        all_S: list[float],
        all_C: list[float],
        all_GSI: list[float]
    ):
        # convert to numpy
        all_S = np.array(all_S)
        all_C = np.array(all_C)
        all_GSI = np.array(all_GSI)

        UF = self.__compute_contribution_weight(self_C, all_C)
        PF = self.__compute_protection_factor(self_S, all_S)
        AS = self.__compute_anomaly_suppression(self_S, self_GSI, self_LIR, all_S, all_GSI)

        # Final epsilon
        epsilon = (
            self.epsilon_base
            * UF
            * PF
            * AS
        )

        return epsilon

    def __compute_anomaly_suppression(self, self_S, self_GSI, self_LIR, all_S, all_GSI):
        mu_S = np.mean(all_S)
        sigma_S = np.std(all_S)

        mu_GSI = np.mean(all_GSI)
        sigma_GSI = np.std(all_GSI)

        anomaly = (
            (self_S > mu_S + 3 * sigma_S)
            and
            (self_GSI < mu_GSI - 2 * sigma_GSI)
            and
            (self_LIR < 0)
        )
        
        return self.epsilon_min / self.epsilon_base if anomaly else 1.0

    def __compute_protection_factor(self, self_S, all_S):
        mean_S = np.mean(all_S)

        return np.exp(
                    -self.lambda_protection
                    * self_S
                    / (mean_S + self.xi)
                )

    def __compute_contribution_weight(self, self_C, all_C):
        sum_exp_C = np.sum(np.exp(all_C)) + self.xi
        return np.exp(self_C) / sum_exp_C