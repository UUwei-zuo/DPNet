from _dpnet_import_paths import ensure_dpnet_import_paths
from math import cos, inf, pi, sin, sqrt, tan
import threading

import numpy as np
import rospy
import torch

ensure_dpnet_import_paths(__file__)

from action_solver import DPNetSolver
from d_kalmannet.predictor import KalmanPredictor
from utils.geometry_utils import create_rectangle_vertex


def _sat_intersect(poly_a, poly_b):
    """Check for intersection between two convex polygons using the Separating Axis Theorem."""
    for poly in (poly_a, poly_b):
        n_pts = poly.shape[0]
        for idx in range(n_pts):
            p = poly[idx]
            q = poly[(idx + 1) % n_pts]
            edge = q - p
            axis = np.array([-edge[1], edge[0]])
            proj_a = poly_a @ axis
            proj_b = poly_b @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True

def _point_segment_distance(point, seg_start, seg_end):
    """Compute the shortest distance from a point to a segment in 2D."""
    segment = seg_end - seg_start
    denom = float(np.dot(segment, segment)) + 1e-12
    t = np.clip(float(np.dot(point - seg_start, segment)) / denom, 0.0, 1.0)
    closest = seg_start + t * segment
    return float(np.linalg.norm(point - closest))

def boxes_min_distance(vertices_a, vertices_b):
    """Calculate minimum distance between two convex rectangles given by their vertices."""
    poly_a = np.asarray(vertices_a, dtype=float)
    poly_b = np.asarray(vertices_b, dtype=float)

    if poly_a.shape == (2, 4):
        poly_a = poly_a.T
    if poly_b.shape == (2, 4):
        poly_b = poly_b.T

    if _sat_intersect(poly_a, poly_b):
        return 0.0

    def edge_iter(poly):
        for idx in range(poly.shape[0]):
            yield poly[idx], poly[(idx + 1) % poly.shape[0]]

    min_dist = float('inf')
    for point in poly_a:
        for start, end in edge_iter(poly_b):
            min_dist = min(min_dist, _point_segment_distance(point, start, end))
    for point in poly_b:
        for start, end in edge_iter(poly_a):
            min_dist = min(min_dist, _point_segment_distance(point, start, end))
    return min_dist

class MpcPathTracking:
    def __init__(self, car_tuple, receding=5, sample_time=0.1, iter_num=1, max_num_obs=20,
                 **kwargs) -> None:

        self.car_tuple = car_tuple
        self.L = car_tuple.wheelbase
        self.receding = receding
        self.sample_time = sample_time

        self.control_sequence = kwargs.get('init_vel', np.zeros((2, self.receding)))
        self.cur_index = 0

        self.use_precise_collision = kwargs.get('use_precise_collision', True)
        self.ego_length = kwargs.get('ego_length', 4.5)
        self.ego_width = kwargs.get('ego_width', 2.2)
        self._ego_rect_local = create_rectangle_vertex(self.ego_length, self.ego_width)
        control_cfg = rospy.get_param('~DPNet_planner', rospy.get_param('~control', {}))
        ref_cfg = control_cfg.get('ref_generation', {})
        self.ref_interval = kwargs.get(
            'ref_interval',
            ref_cfg.get('interval', rospy.get_param('~ref_interval', 0.2)),
        )
        d_kalman_cfg = control_cfg.get('d-kalmannet', {})
        predictor_dt = float(d_kalman_cfg.get('dt', 0.2))
        self.predictor_dt = predictor_dt
        self.predictor = KalmanPredictor(
            horizon=self.receding,
            dt=predictor_dt,
            gpu_id=0
        )
        self.os = DPNetSolver(
            receding, car_tuple, iter_num, sample_time, max_num_obs, **kwargs
        )
        rospy.loginfo('DTMPC solver backend: action')

        # DT-MPC hyperparameters (can be overridden via ROS params)
        dt_cfg = control_cfg.get('dt_mpc', {})
        self.dt_mpc_params = {
            'd1': dt_cfg.get('d1', rospy.get_param('~dt_mpc_d1', 0.10)),
            'd2': dt_cfg.get('d2', rospy.get_param('~dt_mpc_d2', 0.40)),
            'd0': dt_cfg.get('d0', rospy.get_param('~dt_mpc_d0', 2.0)),
            'alpha': dt_cfg.get('alpha', rospy.get_param('~dt_mpc_alpha', 0.20)),
            'beta': dt_cfg.get('beta', rospy.get_param('~dt_mpc_beta', 0.05)),
            'tau1_min': dt_cfg.get('tau1_min', rospy.get_param('~dt_mpc_tau1_min', 0.30)),
            'tau2_min': dt_cfg.get('tau2_min', rospy.get_param('~dt_mpc_tau2_min', 0.30)),
            'kappa_init': dt_cfg.get('kappa_init', rospy.get_param('~dt_mpc_kappa_init', 1.00)),
            'delta_kappa': dt_cfg.get('delta_kappa', rospy.get_param('~dt_mpc_delta_kappa', 1.00)),
        }
        self.prev_opt_states = None
        self._prediction_lock = threading.Lock()
        self._latest_prediction_by_id = {}
        self._latest_moving_ids = set()

    def _apply_velocity_estimates(self, velocity_estimate):
        """Write velocity estimates and classify moving obstacles."""
        id_to_index = {}
        for i, obs_state in enumerate(self.os.obstacle_states):
            if 'id' in obs_state:
                id_to_index[obs_state['id']] = i

        if velocity_estimate:
            for obs_id, velocity in velocity_estimate.items():
                if obs_id in id_to_index:
                    index = id_to_index[obs_id]
                    self.os.obstacle_states[index]['vx'] = velocity[0]
                    self.os.obstacle_states[index]['vy'] = velocity[1]

        velocity_threshold = 0.1
        moving_indices = []
        for i, obs_state in enumerate(self.os.obstacle_states):
            speed = np.hypot(obs_state.get('vx', 0.0), obs_state.get('vy', 0.0))
            if speed >= velocity_threshold:
                moving_indices.append(i)

        return moving_indices

    @staticmethod
    def _apply_velocity_estimates_to_states(obstacle_states, velocity_estimate):
        """Apply velocity estimates to a local obstacle-state snapshot."""
        id_to_index = {
            obs_state['id']: i
            for i, obs_state in enumerate(obstacle_states)
            if 'id' in obs_state
        }
        if velocity_estimate:
            for obs_id, velocity in velocity_estimate.items():
                index = id_to_index.get(obs_id)
                if index is None:
                    continue
                obstacle_states[index]['vx'] = velocity[0]
                obstacle_states[index]['vy'] = velocity[1]

        velocity_threshold = 0.1
        moving_indices = []
        for i, obs_state in enumerate(obstacle_states):
            speed = np.hypot(obs_state.get('vx', 0.0), obs_state.get('vy', 0.0))
            if speed >= velocity_threshold:
                moving_indices.append(i)

        return moving_indices

    def update_obstacle_snapshot(self, obstacle_info):
        """Update the latest obstacle snapshot used by predictor/control loops."""
        with self._prediction_lock:
            self.os.receive_obstacles(obstacle_info)

    def update_prediction_buffer(self, velocity_estimate):
        """Run one predictor cycle and cache latest predictions by obstacle id."""
        with self._prediction_lock:
            obstacle_states = [dict(obs) for obs in self.os.obstacle_list_buffer]

        if not obstacle_states:
            with self._prediction_lock:
                self._latest_prediction_by_id = {}
                self._latest_moving_ids = set()
            return

        moving_indices = self._apply_velocity_estimates_to_states(obstacle_states, velocity_estimate)
        moving_predictions_by_id = {}

        if moving_indices:
            state_matrix = np.array([
                [obs['x'], obs['vx'], 0.0, obs['y'], obs['vy'], 0.0]
                for obs in obstacle_states
            ], dtype=np.float32)
            state_tensor = torch.from_numpy(state_matrix)
            moving_tensors = [state_tensor[i] for i in moving_indices]
            moving_prediction_list = list(self.predictor(moving_tensors))

            for original_index, pred in zip(moving_indices, moving_prediction_list):
                obs_id = obstacle_states[original_index].get('id')
                if obs_id is not None:
                    moving_predictions_by_id[obs_id] = pred

        with self._prediction_lock:
            self._latest_prediction_by_id = moving_predictions_by_id
            self._latest_moving_ids = set(moving_predictions_by_id.keys())

    def _prepare_obstacles_from_cached_prediction(self):
        """Build solver obstacle inputs using latest obstacle and prediction caches."""
        with self._prediction_lock:
            current_obstacle_states = [dict(obs) for obs in self.os.obstacle_list_buffer]
            prediction_by_id = dict(self._latest_prediction_by_id)

        if not current_obstacle_states:
            self.os.obstacle_states = []
            self.os.obstacle_list = []
            return {}, [], None

        moving_predictions = {}
        moving_indices = []
        moving_prediction_list = []
        for idx, obs in enumerate(current_obstacle_states):
            obs_id = obs.get('id')
            if obs_id in prediction_by_id:
                pred = prediction_by_id[obs_id]
                moving_predictions[idx] = pred
                moving_indices.append(idx)
                moving_prediction_list.append(pred)

        self.os.obstacle_states = current_obstacle_states
        self.os.obstacle_list = self.os.convert_to_solver_obstacles(moving_predictions)
        return moving_predictions, moving_prediction_list, moving_indices

    def _predict_moving_obstacles(self, moving_indices):
        """Predict moving obstacles and convert predictions for solver."""
        moving_predictions = {}
        moving_prediction_list = None
        phi_list = None

        if self.os.obstacle_states:
            try:
                if moving_indices:
                    state_matrix = np.array([
                        [obs['x'], obs['vx'], 0.0, obs['y'], obs['vy'], 0.0]
                        for obs in self.os.obstacle_states
                    ], dtype=np.float32)
                    state_tensor = torch.from_numpy(state_matrix)
                    moving_tensors = [state_tensor[i] for i in moving_indices]
                    moving_prediction_list = list(self.predictor(moving_tensors))
                    moving_predictions = {
                        original_index: pred
                        for original_index, pred in zip(moving_indices, moving_prediction_list)
                    }
                phi_list = self.compute_doppler_phi(
                    self.prev_opt_states, self.os.obstacle_states, moving_predictions, self.receding
                )

                self.os.obstacle_list = self.os.convert_to_solver_obstacles(moving_predictions)
            except Exception as e:
                rospy.logwarn(f"Prediction/conversion failed: {e}. Using static obstacles.")

        return moving_predictions, moving_prediction_list, moving_indices, phi_list

    def _set_phi_with_fallback(self, phi_list):
        """Set phi values on solver; fallback to constant profile if unavailable."""
        d1 = self.dt_mpc_params['d1']
        num_obs = len(self.os.obstacle_list)
        phi_fallback = [np.full((self.receding,), d1, dtype=float) for _ in range(num_obs)]
        self.os.set_phi(phi_list if phi_list is not None else phi_fallback)

    def control(self, state, ref_path, ref_speed=None, **kwargs):

        ego_predicted_states, ref_traj_list, self.cur_index = self.pre_process(
            state, ref_path, self.cur_index, ref_speed, **kwargs
        )

        (
            moving_predictions,
            moving_prediction_list,
            moving_indices,
        ) = self._prepare_obstacles_from_cached_prediction()
        phi_list = self.compute_doppler_phi(
            self.prev_opt_states, self.os.obstacle_states, moving_predictions, self.receding
        )

        self.os.update_obstacles()
        self._set_phi_with_fallback(phi_list)

        optimal_actions, info = self.os.iterative_solve(
            ego_predicted_states,
            self.control_sequence,
            ref_traj_list,
            ref_speed,
            **kwargs,
        )

        if self.cur_index == len(ref_path) - 1:
            optimal_actions = np.zeros((2, self.receding))
            info['arrive'] = True
        else:
            info['arrive'] = False

        info['predictions'] = moving_prediction_list if moving_prediction_list else None
        info['moving_indices'] = moving_indices if moving_indices else None
        info['prediction_horizon'] = self.receding

        self.control_sequence = optimal_actions

        if 'opt_state_list' in info:
            self.prev_opt_states = info['opt_state_list']

        return optimal_actions[:, 0:1], info

    def pre_process(self, current_state, ref_path, current_index, ref_speed, **kwargs):

        min_dis, min_index = self.closest_point(current_state, ref_path, current_index, **kwargs)
        traj_point = self._state_to_column(ref_path[min_index])[0:3, :]
        ref_traj_list = [traj_point.copy()]
        state_pre_list = [current_state]
        current_ref_index = min_index
        move_len = max(float(ref_speed), 0.0) * self.sample_time

        for i in range(self.receding):
            current_state = self.motion_predict_model(
                current_state,
                self.control_sequence[:, i:i + 1],
                self.L,
                self.sample_time,
            )
            state_pre_list.append(current_state)

            traj_point, current_ref_index = self.inter_point(
                traj_point, ref_path, current_ref_index, move_len
            )
            traj_point = self._state_to_column(traj_point)[0:3, :]

            diff = traj_point[2, 0] - current_state[2, 0]
            traj_point[2, 0] = current_state[2, 0] + MpcPathTracking.normalize_angle(diff)
            ref_traj_list.append(traj_point.copy())

        state_pre_array = np.hstack(state_pre_list)

        return state_pre_array, ref_traj_list, min_index

    def motion_predict_model(self, car_state, vel, wheel_base, sample_time):

        assert car_state.shape == (3, 1) and vel.shape == (2, 1)

        phi = car_state[2, 0]

        v = vel[0, 0]
        psi = vel[1, 0]

        ds = np.array([[v * cos(phi)], [v * sin(phi)], [v * tan(psi) / wheel_base]])

        next_state = car_state + ds * sample_time

        return next_state

    def closest_point(self, state, ref_path, start_ind, threshold=0.1, ind_range=10, **kwargs):

        min_dis = inf
        min_ind = start_ind

        for i, waypoint in enumerate(ref_path[start_ind:start_ind + ind_range]):
            dis = MpcPathTracking.distance(state[0:2], waypoint[0:2])
            if dis < min_dis:
                min_dis = dis
                min_ind = start_ind + i
                if dis < threshold: break

        return min_dis, min_ind

    def inter_point(self, traj_point, ref_path, current_index, length):

        circle = np.squeeze(traj_point[0:2])

        while True:

            if current_index + 1 > len(ref_path) - 1:
                end_point = ref_path[-1][:, 0]
                end_point[2] = MpcPathTracking.normalize_angle(end_point[2])

                return end_point, current_index

            cur_point = ref_path[current_index]
            next_point = ref_path[current_index + 1]

            segment = [np.squeeze(cur_point[0:2]), np.squeeze(next_point[0:2])]
            int_point = self.range_cir_seg(circle, length, segment)

            if int_point is None:
                current_index = current_index + 1
            else:
                diff = MpcPathTracking.normalize_angle(next_point[2, 0] - cur_point[2, 0])
                theta = MpcPathTracking.normalize_angle(cur_point[2, 0] + diff / 2)
                traj_point = np.append(int_point, theta)

                return traj_point, current_index

    @staticmethod
    def _state_to_column(point):
        """Normalize point arrays to shape (N, 1)."""
        point_array = np.asarray(point)
        if point_array.ndim == 1:
            return point_array[:, np.newaxis]
        return point_array

    def range_cir_seg(self, circle, r, segment):

        assert circle.shape == (2,) and segment[0].shape == (2,) and segment[1].shape == (2,)

        sp = segment[0]
        ep = segment[1]

        d = ep - sp

        if np.linalg.norm(d) == 0:
            return None

        f = sp - circle

        a = d @ d
        b = 2 * f @ d
        c = f @ f - r ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None
        else:

            t1 = (-b - sqrt(discriminant)) / (2 * a)
            t2 = (-b + sqrt(discriminant)) / (2 * a)

            if t2 >= 0 and t2 <= 1:
                int_point = sp + t2 * d
                return int_point

            return None

    @staticmethod
    def distance(point1, point2):
        return sqrt((point1[0, 0] - point2[0, 0]) ** 2 + (point1[1, 0] - point2[1, 0]) ** 2)

    @staticmethod
    def normalize_angle(radian):
        while radian > pi:
            radian = radian - 2 * pi
        while radian < -pi:
            radian = radian + 2 * pi

        return radian

    def compute_doppler_phi(self, prev_states, obstacle_states, moving_predictions, T):
        """Compute per-obstacle, per-time φ using Doppler-inferred collision check.
        Returns a list of length N_obstacles; each element is a vector of length T.
        """
        params = self.dt_mpc_params
        d1 = params['d1']; d2 = params['d2']; d0 = params['d0']
        alpha = params['alpha']; beta = params['beta']
        tau1_min = params['tau1_min']; tau2_min = params['tau2_min']
        kappa_init = params['kappa_init']; delta_kappa = params['delta_kappa']

        num_obs = len(obstacle_states)
        # Temporal factor τ2(h), h=1..T
        tau2 = np.array([max(1.0 - beta * (h - 1), tau2_min) for h in range(1, T + 1)], dtype=float)

        # If no previous states, return constant φ=d1
        if not prev_states or len(prev_states) < 3:
            return [np.full((T,), d1, dtype=float) for _ in range(num_obs)]

        # Historical potential states S_{t-1}^∘ ≈ prev_states[2:] -> length T-1
        hist_states = prev_states[2:]
        if len(hist_states) == 0:
            return [np.full((T,), d1, dtype=float) for _ in range(num_obs)]

        ego_centers = [(float(s[0, 0]), float(s[1, 0])) for s in hist_states]

        # Build per-obstacle predicted centers for h=1..T
        obs_pred_centers = []
        for i, obs in enumerate(obstacle_states):
            if (moving_predictions is not None) and (i in moving_predictions):
                pred = moving_predictions[i]
                centers = []
                for h in range(1, T + 1):
                    idx = min(h, len(pred) - 1) if len(pred) > 0 else 0
                    ps = pred[idx]
                    # predictor format: [x, vx, ax, y, vy, ay]
                    px = float(ps[0].item()) if hasattr(ps[0], 'item') else float(ps[0])
                    py = float(ps[3].item()) if hasattr(ps[3], 'item') else float(ps[3])
                    centers.append((px, py))
                obs_pred_centers.append(centers)
            else:
                # Static or no prediction: repeat current position
                centers = [(float(obs['x']), float(obs['y'])) for _ in range(T)]
                obs_pred_centers.append(centers)

        # Prepare obstacle geometry if precise collision is enabled
        if self.use_precise_collision:
            obstacle_bases = [create_rectangle_vertex(float(obs.get('length', 5.0)),
                                                      float(obs.get('width', 2.5)))
                              for obs in obstacle_states]
            obstacle_yaws = [float(obs.get('yaw', 0.0)) for obs in obstacle_states]

        # Doppler-inferred collision priority ρ_n
        rho = [np.inf for _ in range(num_obs)]
        kappa = float(kappa_init)
        Hm1 = len(ego_centers)  # T-1 typically
        for h_idx in range(Hm1):
            # Compare G^∘_{h+1} with O_{h+1|t}
            ego_xy = ego_centers[h_idx]
            obs_step = min(h_idx + 1, T - 1)
            for n in range(num_obs):
                if np.isfinite(rho[n]):
                    continue
                ox, oy = obs_pred_centers[n][obs_step]

                if self.use_precise_collision:
                    ego_yaw = float(hist_states[h_idx][2, 0]) if hist_states[h_idx].shape[0] >= 3 else 0.0
                    rot_ego = np.array([[np.cos(ego_yaw), -np.sin(ego_yaw)],
                                        [np.sin(ego_yaw),  np.cos(ego_yaw)]])
                    ego_vertices = (rot_ego @ self._ego_rect_local + np.array([[ego_xy[0]], [ego_xy[1]]])).T

                    rot_obs = np.array([[np.cos(obstacle_yaws[n]), -np.sin(obstacle_yaws[n])],
                                        [np.sin(obstacle_yaws[n]),  np.cos(obstacle_yaws[n])]])
                    obs_vertices = (rot_obs @ obstacle_bases[n] + np.array([[ox], [oy]])).T

                    dist = boxes_min_distance(ego_vertices, obs_vertices)
                else:
                    dist = np.hypot(ego_xy[0] - ox, ego_xy[1] - oy)

                if dist <= d0:
                    rho[n] = kappa
            kappa += float(delta_kappa)

        # Spatial factor τ1(n)
        tau1 = []
        for n in range(num_obs):
            if np.isfinite(rho[n]):
                tau1_n = max(1.0 - alpha * (rho[n] - kappa_init), tau1_min)
            else:
                tau1_n = tau1_min
            tau1.append(tau1_n)

        # Compose φ_{h,n} = d1 + τ1(n) τ2(h) d2
        phi_list = []
        for n in range(num_obs):
            phi_n = d1 + (tau1[n] * tau2) * d2
            phi_list.append(phi_n.astype(float))
        return phi_list