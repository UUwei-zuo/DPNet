import torch


class SystemModel:
    """System matrices and functions for KalmanNet."""

    def __init__(self, F, Q, H, R, T=None, T_test=None):
        """Initialize model dimensions and priors."""
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.m = F.shape[0]
        self.n = H.shape[0]
        self.T = T if T is not None else 100
        self.T_test = T_test if T_test is not None else 100
        self.f = self.state_transition_function
        self.h = self.observation_function
        self.prior_Q = Q.clone()
        self.prior_Sigma = torch.eye(self.m) * 0.5
        self.prior_S = H @ self.prior_Sigma @ H.T + R

    def state_transition_function(self, x):
        """Apply state transition matrix."""
        return torch.bmm(self.F.expand(x.shape[0], -1, -1), x)

    def observation_function(self, x):
        """Apply observation matrix."""
        return torch.bmm(self.H.expand(x.shape[0], -1, -1), x)

    @staticmethod
    def create_CA_matrices(m, dt, q2):
        """Create constant-acceleration dynamics matrices."""
        F = torch.eye(m)
        F[0, 1] = dt
        F[0, 2] = 0.5 * dt ** 2
        F[1, 2] = dt
        F[3, 4] = dt
        F[3, 5] = 0.5 * dt ** 2
        F[4, 5] = dt

        Q = torch.zeros(m, m)
        ca_block = q2 * torch.tensor(
            [
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt],
            ]
        )
        Q[0:3, 0:3] = ca_block
        Q[3:6, 3:6] = ca_block
        return F, Q

    @staticmethod
    def generate_perturbed_covariance(base_dim, base_value, gain=0.0, gain_factor=0.1):
        """Create perturbed positive-definite covariance."""
        if gain != 0:
            perturbation = gain_factor * base_value * torch.eye(base_dim)
            A = base_value * torch.eye(base_dim) + perturbation
            return torch.transpose(A, 0, 1) * A
        return base_value * torch.eye(base_dim)
