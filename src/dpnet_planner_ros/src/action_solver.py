from _dpnet_import_paths import ensure_dpnet_import_paths
from collections import namedtuple
from math import cos, inf, sin, tan
from multiprocessing import Pool

import cvxpy as cp
import numpy as np

ensure_dpnet_import_paths(__file__)

from utils.geometry_utils import create_rectangle_vertex, generate_inequalities

pool = None
MAX_OBSTACLES = 10
OBSTACLE_SIZE_SCALE = 1.0
Obstacle = namedtuple('Obstacle', 'A b cone_type')

class DPNetSolver:

    def __init__(self, receding, car_tuple, iter_num=2, step_time=0.1, max_num_obs=10,
                 iter_threshold=0.1, process_num=12, use_ecos=True, **kwargs) -> None:

        # setting
        self.T = receding
        self.car_tuple = car_tuple
        self.L = car_tuple.wheelbase
        self.max_num_obs = max_num_obs
        self.iter_num = iter_num
        self.step_time = step_time
        self.acce_bound = np.array([[car_tuple.abs_acce], [car_tuple.abs_acce_steer]]) * self.step_time
        self.iter_threshold = iter_threshold
        
        # **SOLVER CONFIGURATION**
        self.use_ecos = use_ecos
        
        # independ variable
        self.indep_s = cp.Variable((3, receding + 1), name='state')
        self.indep_u = cp.Variable((2, receding), name='vel')
        self.phi_list = []

        self.dummy_obstacle_vertex = create_rectangle_vertex(5, 2.5)
        placeholder_obstacle_params = self._create_obstacle_params(
            1000.0, 1000.0, 0.0, self.dummy_obstacle_vertex
        )
        self.obstacle_list = [placeholder_obstacle_params] * MAX_OBSTACLES
        self.obstacle_meta = []

        self.update_obstacles()

        self.obstacle_list_buffer = []

        self.obstacle_states = []

        global pool
        if process_num > 1:
            pool = Pool(process_num)
    
    def set_phi(self, phi_values_list):
        """Set per-obstacle, per-time safety distances φ for the current horizon."""
        if not isinstance(phi_values_list, list):
            raise ValueError("phi_values_list must be a list")
        num_obs = len(self.obstacle_list)
        if len(phi_values_list) != num_obs:
            raise ValueError(f"phi length {len(phi_values_list)} != number of obstacles {num_obs}")
        if (not hasattr(self, 'phi_list')) or (len(self.phi_list) != num_obs):
            self.phi_list = [cp.Parameter((self.T,)) for _ in range(num_obs)]
        for i in range(num_obs):
            vec = np.asarray(phi_values_list[i]).reshape((self.T,))
            self.phi_list[i].value = vec

    def receive_obstacles(self, obstacle_info):
        """Process obstacle information and update the obstacle list."""
        self.obstacle_list_buffer = []

        for obstacle in obstacle_info:
            obstacle_state = {
                'id': obstacle['id'],
                'x': obstacle['x'],
                'y': obstacle['y'],
                'yaw': obstacle['yaw'],
                'vx': 0,
                'vy': 0,
                'length': obstacle.get('length', 5.0),
                'width': obstacle.get('width', 2.5),
                'height': obstacle.get('height', 2.0)
            }
            self.obstacle_list_buffer.append(obstacle_state)

    def convert_to_solver_obstacles(self, moving_predictions=None):
        """Convert obstacles to solver format using moving obstacle predictions."""
        solver_obstacles = []
        
        for i, obs_state in enumerate(self.obstacle_states):
            x, y = obs_state['x'], obs_state['y']
            yaw = obs_state['yaw']
            length = obs_state.get('length', 5.0) * OBSTACLE_SIZE_SCALE
            width = obs_state.get('width', 2.5) * OBSTACLE_SIZE_SCALE

            obstacle_vertex = create_rectangle_vertex(length, width)

            if moving_predictions and i in moving_predictions:
                predictions = moving_predictions[i]
                vertex_list = []
                
                for t in range(min(len(predictions), self.T + 1)):
                    if t < len(predictions):
                        pred_state = predictions[t]
                        pred_x = pred_state[0].item()
                        pred_y = pred_state[3].item()
                        pred_yaw = yaw
                        rot_matrix = np.array([[cos(pred_yaw), -sin(pred_yaw)], [sin(pred_yaw), cos(pred_yaw)]])
                        tran_matrix = np.array([[pred_x], [pred_y]])
                        vertices_t = rot_matrix @ obstacle_vertex + tran_matrix
                        vertex_list.append(vertices_t)
                    else:
                        vertex_list.append(vertex_list[-1])

                while len(vertex_list) < self.T + 1:
                    vertex_list.append(vertex_list[-1])

                A_list = []
                b_list = []
                for vertices_t in vertex_list:
                    A_t, b_t = generate_inequalities(vertices_t)
                    A_list.append(A_t)
                    b_list.append(b_t)
                
                solver_obstacles.append(Obstacle(A_list, b_list, 'Rpositive'))
                
            else:
                static_obstacle = self._create_obstacle_params(x, y, yaw, obstacle_vertex)
                solver_obstacles.append(static_obstacle)
        
        return solver_obstacles

    def update_obstacles(self):
        self.obstacle_meta = [self._build_obstacle_meta(obs) for obs in self.obstacle_list]
        if not hasattr(self, 'indep_lam_list') or len(self.indep_lam_list) != len(
            self.obstacle_list
        ):
            self.indep_lam_list = [
                cp.Variable((meta['edge_num'], self.T + 1))
                for meta in self.obstacle_meta
            ]
            self.indep_mu_list = [
                cp.Variable((self.car_tuple.G.shape[0], self.T + 1))
                for obs in self.obstacle_list
            ]

            self.nom_lam_list = [
                np.ones((meta['edge_num'], self.T + 1))
                for meta in self.obstacle_meta
            ]
            self.nom_mu_list = [
                np.ones((self.car_tuple.G.shape[0], self.T + 1))
                for obs in self.obstacle_list
            ]
            self.xi_list = [
                np.zeros((self.T + 1, 2))
                for obs in self.obstacle_list
            ]

    def iterative_solve(self, ego_predicted_states, control_sequence, ref_traj_list, ref_speed, **kwargs):

        for i in range(self.iter_num):
            (
                optimal_states,
                optimal_actions,
                resi_dual,
                resi_pri,
            ) = self.rda_prob(
                ref_traj_list, ref_speed, ego_predicted_states, control_sequence, **kwargs
            )

            if resi_dual < 0.1 or resi_pri < 0.1:
                break

        opt_state_list = [state[:, np.newaxis] for state in optimal_states.T]
        info = {'ref_traj_list': ref_traj_list, 'opt_state_list': opt_state_list}

        return optimal_actions, info

    def rda_prob(self, ref_state, ref_speed, ego_predicted_states, control_sequence, **kwargs):

        s = ego_predicted_states
        u = control_sequence
        s, u = self.update_su_prob(ref_state, ref_speed, s, u, **kwargs)

        input_args = [
            (self, s, obs_index, kwargs)
            for obs_index in range(len(self.obstacle_list))
        ]
        lam_mu_list = pool.map(DPNetSolver.update_LamMu_prob, input_args)

        safe_LamMU_list = []
        for obs_index, lam_mu in enumerate(lam_mu_list):
            if lam_mu is None:
                safe_LamMU_list.append(
                    (self.nom_lam_list[obs_index], self.nom_mu_list[obs_index], inf)
                )
            else:
                safe_LamMU_list.append(lam_mu)

        self.nom_lam_list = [lam_mu[0] for lam_mu in safe_LamMU_list]
        self.nom_mu_list = [lam_mu[1] for lam_mu in safe_LamMU_list]

        if len(safe_LamMU_list) != 0:
            resi_dual = max(
                [lam_mu[2] for lam_mu in safe_LamMU_list]
            )
        else:
            resi_dual = 0

        self.xi_list, resi_pri = self.update_xi(s)

        return s, u, resi_dual, resi_pri

    def nav_cost_cons(self, ref_state, ref_speed, nom_s, nom_u, **kwargs):
        cost = 0
        constraints = []
        states_by_ackermann = []

        nav_weights = kwargs.get(
            'nav_weights',
            {'wu': 1.0, 'wut': 1.0, 'ws': 1.0, 'wst': 1.0},
        )

        cost += nav_weights['wu'] * cp.sum_squares(self.indep_u[0, 0:self.T - 1] - ref_speed)
        cost += nav_weights['wut'] * cp.square(self.indep_u[0, self.T - 1] - ref_speed)
        ref_s = np.hstack(ref_state)

        cost += nav_weights['ws'] * cp.sum_squares(self.indep_s[:, 0:self.T - 1] - ref_s[:, 0:self.T - 1])
        cost += nav_weights['wst'] * cp.sum_squares(self.indep_s[:, self.T - 1] - ref_s[:, self.T - 1])

        for t in range(self.T):
            indep_st = self.indep_s[:, t:t + 1]
            indep_st1 = self.indep_s[:, t + 1:t + 2]
            indep_ut = self.indep_u[:, t:t + 1]

            nom_st = nom_s[:, t:t + 1]
            nom_ut = nom_u[:, t:t + 1]

            ref_st1 = ref_state[t + 1]
            ref_ut = ref_speed

            A, B, C = self.linear_ackermann_model(nom_st, nom_ut, self.step_time, self.L)
            states_by_ackermann.append(A @ indep_st + B @ indep_ut + C)

        ackermann_states_array = cp.hstack(states_by_ackermann)

        constraints += [self.indep_s[:, 1:] == ackermann_states_array]
        constraints += [cp.abs(self.indep_u[:, 1:] - self.indep_u[:, :-1]) <= self.acce_bound]
        constraints += [cp.abs(self.indep_u[0, :]) <= self.car_tuple.abs_speed]
        constraints += [cp.abs(self.indep_u[1, :]) <= self.car_tuple.abs_steer]
        constraints += [self.indep_s[:, 0:1] == nom_s[:, 0:1]]

        return cost, constraints

    def update_su_prob(self, ref_state, ref_speed, nom_s, nom_u, **kwargs):
        nav_cost, nav_constraints = self.nav_cost_cons(ref_state, ref_speed, nom_s, nom_u, **kwargs)
        su_cost, su_constraints = self.update_su_cost_cons(nom_s, **kwargs)

        prob_su = cp.Problem(cp.Minimize(nav_cost + su_cost), su_constraints + nav_constraints)

        self._solve_problem(prob_su, verbose=False)

        if prob_su.status == cp.OPTIMAL:
            return self.indep_s.value, self.indep_u.value

        else:
            return nom_s, nom_u

    def _solve_problem(self, problem, verbose=False):
        if self.use_ecos:
            return problem.solve(solver=cp.ECOS, verbose=verbose)

        return problem.solve(verbose=verbose)

    @staticmethod
    def update_LamMu_prob(input_args):
        self, nom_s, obs_index, kwargs = input_args
        rot_cache = self._rotation_cache_from_state(nom_s)

        ro = kwargs.get('ro', 1.0)

        cost = 0
        constraints = []
        Hmt_list = []

        obs = self.obstacle_list[obs_index]
        obs_meta = self.obstacle_meta[obs_index]
        time_varying = obs_meta['time_varying']
        indep_lam_array = self.indep_lam_list[obs_index]
        indep_mu_array = self.indep_mu_list[obs_index]
        nom_y_array = self.xi_list[obs_index]  
        nom_lam_array = self.nom_lam_list[obs_index]
        nom_mu_array = self.nom_mu_list[obs_index]

        for t in range(self.T):
            indep_lam = indep_lam_array[:, t+1:t+2]
            indep_mu = indep_mu_array[:, t+1:t+2]
            nom_rot = rot_cache[t + 1]

            if time_varying:
                obs_A_t = obs.A[t+1]
                Hmt = indep_mu.T @ self.car_tuple.G + indep_lam.T @ obs_A_t @ nom_rot
            else:
                Hmt = indep_mu.T @ self.car_tuple.G + indep_lam.T @ obs.A @ nom_rot
            
            Hmt_list.append(Hmt)

        Hm_array = cp.vstack(Hmt_list)
        cost += 0.5 * ro * cp.sum_squares(Hm_array + nom_y_array[1:])

        if time_varying:
            obs_dis_list_t = []
            for t in range(self.T):
                indep_lam_t = indep_lam_array[:, t+1:t+2]
                indep_mu_t = indep_mu_array[:, t+1:t+2]
                nom_s_t = nom_s[:, t+1:t+2]
                
                obs_A_t = obs.A[t+1]
                obs_b_t = obs.b[t+1]
                obs_dis_t = (
                    indep_lam_t.T @ obs_A_t @ nom_s_t[0:2]
                    - indep_lam_t.T @ obs_b_t
                    - indep_mu_t.T @ self.car_tuple.g
                )
                obs_dis_list_t.append(obs_dis_t)

            obs_dis_constraints = cp.hstack(obs_dis_list_t)
            constraints += [obs_dis_constraints[0, :] >= self.phi_list[obs_index]]
        else:
            obs_dis_constraints = cp.diag(
                indep_lam_array.T @ obs.A @ nom_s[0:2]
                - indep_lam_array.T @ obs.b
                - indep_mu_array.T @ self.car_tuple.g
            )
            constraints += [obs_dis_constraints[1:] >= self.phi_list[obs_index]]

        if time_varying:
            dual_constraints = []
            for t in range(self.T + 1):
                if t < len(obs.A):
                    obs_A_t = obs.A[t]
                else:
                    obs_A_t = obs.A[-1]
                
                lam_t = indep_lam_array[:, t:t+1]
                dual_constraints.append(cp.norm(obs_A_t.T @ lam_t) <= 1)
            
            constraints += dual_constraints
        else:
            constraints += [cp.norm(obs.A.T @ indep_lam_array, axis=0) <= 1]
            
        constraints += [self.cone_cp_array(-indep_lam_array, obs.cone_type)]
        constraints += [self.cone_cp_array(-indep_mu_array, self.car_tuple.cone_type)]
 
        prob = cp.Problem(cp.Minimize(cost), constraints)

        try:
            self._solve_problem(prob, verbose=False)

            if prob.status == cp.OPTIMAL:
                lam_diff = np.linalg.norm(indep_lam_array.value - nom_lam_array, axis=0)
                mu_diff = np.linalg.norm(indep_mu_array.value - nom_mu_array, axis=0)
                max_diff = np.max(lam_diff) + np.max(mu_diff)

                return indep_lam_array.value, indep_mu_array.value, max_diff
            else:
                return nom_lam_array, nom_mu_array, inf

        except Exception as e:
            print(f"Solver crashed: {e}")
            return nom_lam_array, nom_mu_array, inf


    def update_xi(self, nom_s):
        new_xi_list = []
        H_norm_list = []
        rot_cache = self._rotation_cache_from_state(nom_s)
        for obs_index, obs in enumerate(self.obstacle_list):
            obs_meta = self.obstacle_meta[obs_index]
            time_varying = obs_meta['time_varying']
            H_list = []
            for t in range(self.T + 1):
                lam_t = self.nom_lam_list[obs_index][:, t:t + 1]
                mu_t = self.nom_mu_list[obs_index][:, t:t + 1]
                rot_t = rot_cache[t]

                if time_varying:
                    obs_A_t = obs.A[t]
                    H_n_t = mu_t.T @ self.car_tuple.G + lam_t.T @ obs_A_t @ rot_t
                else:
                    H_n_t = mu_t.T @ self.car_tuple.G + lam_t.T @ obs.A @ rot_t
                
                H_list.append(H_n_t)

            H_array = np.vstack(H_list)
            H_norm = np.linalg.norm(H_array, axis=1)
            H_norm_list.append(np.max(H_norm))

            new_xi_list.append(self.xi_list[obs_index] + H_array)

        if len(H_norm_list) == 0:
            resi_pri = 0
        else:
            resi_pri = max(H_norm_list)

        return new_xi_list, resi_pri

    def update_su_cost_cons(self, nom_s, ro=1, **kwargs):

        cost = 0
        constraints = []

        if hasattr(self, 'obstacle_list') and self.obstacle_list and len(self.obstacle_list) > 0:
            num_real_obstacles = len(self.obstacle_list)
            
            for obs_index in range(min(num_real_obstacles, len(self.obstacle_list))):
                obs = self.obstacle_list[obs_index]
                obs_meta = self.obstacle_meta[obs_index]
                time_varying = obs_meta['time_varying']
                nom_lam_array = self.nom_lam_list[obs_index]
                indep_trans_array = self.indep_s[0:2]
                nom_mu_array = self.nom_mu_list[obs_index]

                if time_varying:
                    obs_dis_list_t = []
                    for t in range(self.T):
                        nom_lam_t = nom_lam_array[:, t+1:t+2]
                        nom_mu_t = nom_mu_array[:, t+1:t+2]
                        indep_trans_t = indep_trans_array[:, t+1:t+2]
                        
                        obs_A_t = obs.A[t+1]
                        obs_b_t = obs.b[t+1]
                        obs_dis_t = (
                            nom_lam_t.T @ obs_A_t @ indep_trans_t
                            - nom_lam_t.T @ obs_b_t
                            - nom_mu_t.T @ self.car_tuple.g
                        )
                        obs_dis_list_t.append(obs_dis_t)

                    obs_dis_constraints = cp.hstack(obs_dis_list_t)
                    constraints += [obs_dis_constraints[0, :] >= self.phi_list[obs_index]]
                else:
                    obs_dis_constraints = cp.diag(
                        nom_lam_array.T @ obs.A @ indep_trans_array
                        - nom_lam_array.T @ obs.b
                        - nom_mu_array.T @ self.car_tuple.g
                    )
                    constraints += [obs_dis_constraints[1:] >= self.phi_list[obs_index]]

        return cost, constraints

    def linear_ackermann_model(self, nom_state, nom_u, dt, L):

        phi = nom_state[2, 0]
        v = nom_u[0, 0]
        psi = nom_u[1, 0]

        A = np.array([[1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1]])

        B = np.array([[cos(phi) * dt, 0], [sin(phi) * dt, 0],
                      [tan(psi) * dt / L, v * dt / (L * (cos(psi)) ** 2)]])

        C = np.array([[phi * v * sin(phi) * dt], [-phi * v * cos(phi) * dt],
                      [-psi * v * dt / (L * (cos(psi)) ** 2)]])

        return A, B, C

    def cone_cp_array(self, array, cone='Rpositive'):
        if cone == 'Rpositive':
            return cp.constraints.nonpos.NonPos(array)
        elif cone == 'norm2':
            return cp.constraints.nonpos.NonPos(cp.norm(array[0:-1], axis=0) - array[-1])

    def _rotation_cache_from_state(self, state_array):
        """Precompute 2x2 rotation matrices for each state in horizon."""
        angles = state_array[2, :self.T + 1]
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)
        return [np.array([[c, -s], [s, c]]) for c, s in zip(cos_vals, sin_vals)]

    @staticmethod
    def _build_obstacle_meta(obs):
        """Cache obstacle structure information to avoid repeated hot-path checks."""
        time_varying = isinstance(obs.A, list)
        edge_num = obs.A[0].shape[0] if time_varying else obs.A.shape[0]
        return {'time_varying': time_varying, 'edge_num': edge_num}

    def _create_obstacle_params(self, x, y, yaw, obstacle_vertex):
        """Create obstacle parameters tuple from local-frame obstacle vertices."""
        rot_matrix = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])
        tran_matrix = np.array([[x], [y]])
        new_vertex = rot_matrix @ obstacle_vertex + tran_matrix
        A, b = generate_inequalities(new_vertex)

        return Obstacle(A, b, 'Rpositive')