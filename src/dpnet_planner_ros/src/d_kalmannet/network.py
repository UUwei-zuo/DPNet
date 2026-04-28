import torch
import torch.nn as nn
import torch.nn.functional as func


class KalmanNetNN(torch.nn.Module):
    """Neural Kalman gain estimator."""

    def __init__(self):
        """Initialize module."""
        super().__init__()

    def NNBuild(self, SysModel, args):
        """Build network from system model and config."""
        if args.use_cuda:
            if hasattr(args, "gpu_id") and args.gpu_id is not None:
                self.device = torch.device(f"cuda:{args.gpu_id}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        """Initialize Kalman gain submodules."""
        self.seq_len_input = 1
        self.batch_size = args.n_batch

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU()).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
        ).to(self.device)

        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU()).to(self.device)

        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU()).to(self.device)

        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU()).to(self.device)

        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU()).to(self.device)

        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU()).to(self.device)

    def InitSystemDynamics(self, f, h, m, n):
        """Set transition and observation operators."""
        self.f = f
        self.m = m
        self.h = h
        self.n = n

    def InitSequence(self, M1_0, T):
        """Initialize sequence state."""
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    def step_prior(self):
        """Compute prior state and observation."""
        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = self.h(self.m1x_prior)

    def step_KGain_est(self, y):
        """Estimate Kalman gain from normalized features."""
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def _knet_step_core(self, y, return_prior=False):
        """Run one predict-update step."""
        self.step_prior()
        prior_state = self.m1x_prior.clone() if return_prior else None
        self.step_KGain_est(y)
        dy = y - self.m1y
        inov = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + inov
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y

        if return_prior:
            return self.m1x_posterior, prior_state
        return self.m1x_posterior

    def KNet_step(self, y):
        """Run one KalmanNet step."""
        return self._knet_step_core(y, return_prior=False)

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        """Compute Kalman gain network pass."""
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        in_FC5 = fw_update_diff
        out_FC5 = self.FC5(in_FC5)

        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        in_FC6 = fw_evol_diff
        out_FC6 = self.FC6(in_FC6)

        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)
        self.h_Sigma = out_FC4
        return out_FC2

    def forward(self, y):
        """Forward one observation step."""
        y = y.to(self.device)
        return self.KNet_step(y)

    def init_hidden_KNet(self):
        """Initialize GRU hidden states."""
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

    def KNet_step_with_prior(self, y):
        """Run one step and return posterior with prior."""
        return self._knet_step_core(y, return_prior=True)
