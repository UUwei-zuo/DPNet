import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .dynamics import SystemModel
from .network import KalmanNetNN

class KalmanPredictor:
    """Predict obstacle trajectories with KalmanNet."""

    def __init__(self, model_path="weights/model.pt", horizon=None, dt=None, device=None, gpu_id=1):
        """Initialize predictor state and model."""
        if horizon is None or dt is None:
            raise ValueError("Both 'horizon' and 'dt' parameters must be specified when initializing KalmanPredictor")

        self.horizon = horizon
        self.dt = dt
        self.gpu_id = gpu_id
        self.m = 6
        self.r2 = torch.tensor([0.1]).float()
        self.q2 = torch.tensor([0.1]).float()

        if device is None:
            if torch.cuda.is_available():
                if self.gpu_id is not None:
                    self.device = torch.device(f"cuda:{self.gpu_id}")
                else:
                    self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(f"KalmanPredictor using device: {self.device}")
        if self.device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(self.device)
                print(f"Using GPU Model: {gpu_name}")
            except RuntimeError as e:
                print(f"Could not get GPU model name: {e}")

        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(current_dir, model_path)
        self.F_gen, self.Q_gen = SystemModel.create_CA_matrices(self.m, 5e-2, self.q2)
        self.H_obs, self.R_obs = self._get_observation_params("position_velocity")
        self._load_model()

        self.obstacle_states = {}
        self.obstacle_count = 0
        self.is_first_call = True
        self._setup_transition_matrix(dt)

    def _get_observation_params(self, observation_type):
        """Build observation matrices for the selected type."""
        h_position_velocity = torch.zeros(4, self.m)
        h_position_velocity[0, 0] = 1.0
        h_position_velocity[1, 1] = 1.0
        h_position_velocity[2, 3] = 1.0
        h_position_velocity[3, 4] = 1.0

        h_only_pos_both = torch.zeros(2, self.m)
        h_only_pos_both[0, 0] = 1.0
        h_only_pos_both[1, 3] = 1.0

        r_4 = self.r2 * torch.eye(4)
        r_2 = self.r2 * torch.eye(2)
        configs = {
            "position_both": (h_only_pos_both, r_2),
            "position_velocity": (h_position_velocity, r_4),
            "all_states": (torch.eye(self.m), self.r2 * torch.eye(self.m)),
        }
        return configs.get(observation_type, (h_only_pos_both, r_2))

    def _setup_transition_matrix(self, dt):
        """Update the constant-acceleration transition matrix."""
        new_t = torch.eye(self.m, device=self.device)
        new_t[0, 1] = dt
        new_t[0, 2] = 0.5 * dt ** 2
        new_t[3, 4] = dt
        new_t[3, 5] = 0.5 * dt ** 2
        new_t[1, 2] = dt
        new_t[4, 5] = dt
        self.T = new_t

    def _load_model(self):
        """Load model weights and move model to runtime device."""
        t = 100
        self.sys_model = SystemModel(self.F_gen, self.Q_gen, self.H_obs, self.R_obs, t, t)

        args = SimpleNamespace()
        args.use_cuda = self.device.type == "cuda"
        args.gpu_id = self.gpu_id
        args.n_batch = 32
        args.in_mult_KNet = 5
        args.out_mult_KNet = 40

        self.knet_model = KalmanNetNN()
        self.knet_model.NNBuild(self.sys_model, args)

        try:
            print(f"Loading model from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.knet_model.load_state_dict(state_dict)
            print("Successfully loaded model")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.knet_model.eval()
        self.knet_model.to(self.device)

    def _match_obstacles(self, state_tensors):
        """Map current detections to tracked obstacle ids."""
        if self.is_first_call or len(self.obstacle_states) == 0:
            index_to_id = {}
            new_obstacles = list(range(len(state_tensors)))
            for i in new_obstacles:
                obstacle_id = self.obstacle_count
                index_to_id[i] = obstacle_id
                self.obstacle_count += 1
            return index_to_id, new_obstacles

        index_to_id = {}
        new_obstacles = []
        used_ids = set()

        for i, state in enumerate(state_tensors):
            pos = torch.tensor([state[0].item(), state[3].item()])
            best_match = None
            min_distance = float("inf")

            for obstacle_id, prev_state in self.obstacle_states.items():
                if obstacle_id in used_ids:
                    continue
                prev_pos = torch.tensor([prev_state[0].item(), prev_state[3].item()])
                distance = torch.norm(pos - prev_pos).item()
                if distance < min_distance and distance < 10.0:
                    min_distance = distance
                    best_match = obstacle_id

            if best_match is not None:
                index_to_id[i] = best_match
                used_ids.add(best_match)
            else:
                new_obstacles.append(i)
                obstacle_id = self.obstacle_count
                index_to_id[i] = obstacle_id
                self.obstacle_count += 1

        return index_to_id, new_obstacles

    def _predict_with_transition_matrix_batch(self, states, original_positions=None):
        """Generate trajectories from updated states."""
        batch_size = states.shape[0]
        states = states.to(self.device)
        all_trajectories = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            all_trajectories[i].append(states[i].clone().cpu())

        current_states = states.unsqueeze(-1)
        t_expanded = self.T.unsqueeze(0).expand(batch_size, -1, -1)

        for _ in range(self.horizon - 1):
            next_states = torch.bmm(t_expanded, current_states)
            current_states = next_states
            for i in range(batch_size):
                all_trajectories[i].append(current_states[i].squeeze().cpu())

        if original_positions is None:
            return all_trajectories

        restored_trajectories = []
        for i, trajectory in enumerate(all_trajectories):
            orig_x, orig_y = original_positions[i]
            restored_traj = []
            for state in trajectory:
                restored_state = state.clone()
                restored_state[0] += orig_x
                restored_state[3] += orig_y
                restored_traj.append(restored_state)
            restored_trajectories.append(restored_traj)
        return restored_trajectories

    def _update_with_kalmannet_batch(self, prev_states, observations, batch_size=None):
        """Run KalmanNet updates on a batch of obstacles."""
        if batch_size is None:
            batch_size = prev_states.shape[0]

        with torch.no_grad():
            updated_states = torch.zeros_like(prev_states)
            for i in range(batch_size):
                prev_state = prev_states[i].to(self.device)
                observation = observations[i].to(self.device)

                self.knet_model.batch_size = 1
                x_init = prev_state.unsqueeze(0).unsqueeze(-1)
                self.knet_model.InitSequence(x_init, 2)
                self.knet_model.init_hidden_KNet()

                h = self.sys_model.H.to(self.device)
                y_prev = torch.matmul(h, prev_state.unsqueeze(-1)).squeeze(-1)
                y_prev = y_prev.unsqueeze(0).unsqueeze(-1)
                _ = self.knet_model(y_prev)

                y_curr = torch.matmul(h, observation.unsqueeze(-1)).squeeze(-1)
                y_curr = y_curr.unsqueeze(0).unsqueeze(-1)
                x_posterior = self.knet_model(y_curr)
                updated_states[i] = x_posterior.squeeze()
            return updated_states

    def __call__(self, state_tensors):
        """Predict trajectories for all tracked obstacles."""
        with torch.no_grad():
            index_to_id, new_obstacles = self._match_obstacles(state_tensors)
            batch_size = len(state_tensors)
            if batch_size == 0:
                return []

            batch_states = torch.stack([state.clone() for state in state_tensors])
            original_positions = [(state[0].item(), state[3].item()) for state in state_tensors]
            centered_states = batch_states.clone()
            centered_states[:, 0] = 0.0
            centered_states[:, 3] = 0.0
            updated_states = torch.zeros_like(centered_states)

            if self.is_first_call:
                updated_states = centered_states
            else:
                prev_states = torch.zeros_like(centered_states)
                for i, _ in enumerate(state_tensors):
                    obstacle_id = index_to_id[i]
                    if i in new_obstacles:
                        updated_states[i] = centered_states[i]
                    else:
                        prev_states[i] = self.obstacle_states[obstacle_id]

                non_new_indices = [i for i in range(batch_size) if i not in new_obstacles]
                if non_new_indices:
                    non_new_prev_states = prev_states[non_new_indices]
                    non_new_observations = centered_states[non_new_indices]
                    non_new_updated = self._update_with_kalmannet_batch(
                        non_new_prev_states,
                        non_new_observations,
                        len(non_new_indices),
                    )
                    for idx, orig_idx in enumerate(non_new_indices):
                        updated_states[orig_idx] = non_new_updated[idx]

            predictions = self._predict_with_transition_matrix_batch(
                updated_states,
                original_positions=original_positions,
            )

            updated_states_dict = {}
            for i, state in enumerate(updated_states):
                obstacle_id = index_to_id[i]
                updated_states_dict[obstacle_id] = state.clone()
            self.obstacle_states = updated_states_dict

            if self.is_first_call:
                self.is_first_call = False
            return predictions

    def update_parameters(self, horizon=None, dt=None):
        """Update runtime prediction parameters."""
        if horizon is not None:
            self.horizon = horizon
        if dt is not None:
            self.dt = dt
            self._setup_transition_matrix(dt)
