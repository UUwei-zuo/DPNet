#! /usr/bin/env python3

import os
import sys
import threading
from collections import namedtuple
from math import atan2, cos, pi, sin, sqrt

import numpy as np
import rospy
from _dpnet_import_paths import ensure_dpnet_import_paths

# ROS message imports
from carla_msgs.msg import CarlaCollisionEvent, CarlaEgoVehicleControl
from derived_object_msgs.msg import ObjectArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float64

ensure_dpnet_import_paths(__file__)

from DTMPC import MpcPathTracking
from utils.geometry_utils import create_rectangle_vertex, generate_inequalities
from visualization import PlannerVisualizer

Car = namedtuple('Car', 'G g cone_type wheelbase abs_speed abs_steer abs_acce abs_acce_steer')

# Constants
VELOCITY_SCALING = 35
PLANNER_MODE_ACTIVE = 1
DOPPLER_POINT_DTYPE = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('azimuth', np.float64),
    ('elevation', np.float64),
    ('velocity', np.float64),
])

class DPNetPlanner:
    """Main class for optimal path planning and control of a Limo vehicle."""

    def __init__(self, robot_id):
        # ROS parameters
        control_cfg = rospy.get_param('~DPNet_planner', rospy.get_param('~control', {}))
        self.receding = control_cfg.get('receding', rospy.get_param('receding', 10))
        self.sample_time = control_cfg.get('sample_time', rospy.get_param('sample_time', 0.1))
        self.nav_weights = control_cfg.get(
            'nav_weights',
            {'wu': 1.0, 'wut': 1.0, 'ws': 1.0, 'wst': 1.0},
        )
        self.iter_num = rospy.get_param('iter_num', 2)
        self.iter_threshold = rospy.get_param('iter_threshold', 0.1)

        # Robot parameter
        self.ref_speed = 30
        self.robot_id = robot_id
        # ego [length, width, wheelbase] (m); optional 4th: RViz wireframe height (m)
        self.shape = list(rospy.get_param('shape', [4.8, 1.9, 2.5]))
        self.car_params = self._create_car_params()

        # State variables
        self.robot_state = np.zeros((3, 1))
        self.x = 0
        self.y = 0
        self.z = 0
        self.angle = 0
        self.planner = PLANNER_MODE_ACTIVE
        self.collided = False
        self._arrival_published = False

        # Doppler LiDAR
        self.position_offset = [0, 0, 1.8]
        self.doppler_points_buffer = np.empty(0, dtype=DOPPLER_POINT_DTYPE)
        self._sensor_lock = threading.Lock()

        # Obstacle data for Doppler LiDAR association
        self.obstacle_callback_buffer = []
        self.obstacle_cache = {}
        self.visualizer = PlannerVisualizer(robot_id, self.shape)
        self.ref_path_list = self._load_reference_path()
        self.path = self.visualizer.create_reference_path(self.ref_path_list)

        # Initialize MPC controller
        self.mpc_controller = MpcPathTracking(
            car_tuple=self.car_params,
            receding=self.receding,
            sample_time=self.sample_time,
            iter_num=self.iter_num,
            iter_threshold=self.iter_threshold,
            process_num=8,
            init_vel=0.01 * np.ones((2, self.receding)),
            ego_length=self.shape[0],
            ego_width=self.shape[1],
            use_precise_collision=rospy.get_param('~use_precise_collision', False),
        )

        self._setup_ros()
        predictor_rate_hz = 1.0 / max(self.mpc_controller.predictor_dt, 1e-3)
        self._predictor_timer = rospy.Timer(
            rospy.Duration(max(self.mpc_controller.predictor_dt, 1e-3)),
            self._predictor_timer_callback,
        )
        rospy.loginfo(
            "Started standalone predictor loop at %.2f Hz (dt=%.3f s).",
            predictor_rate_hz,
            self.mpc_controller.predictor_dt,
        )

    def _setup_ros(self):
        """Set up ROS subscribers and publishers."""
        rospy.Subscriber("/carla/" + self.robot_id + "/odometry", Odometry, self.robot_state_callback)
        rospy.Subscriber("/carla/" + self.robot_id + "/objects", ObjectArray, self.obstacle_callback)
        rospy.Subscriber("/carla/" + self.robot_id + "/planner", Float64, self.planner_callback)
        rospy.Subscriber("/carla/" + self.robot_id + "/doppler_lidar", PointCloud2, self.doppler_lidar_callback)
        rospy.Subscriber("/carla/" + self.robot_id + "/collision", CarlaCollisionEvent, self.collision_callback)
        self.vel = Twist()
        self.output = CarlaEgoVehicleControl()
        self.pub_vel = rospy.Publisher(
            '/carla/' + self.robot_id + '/vehicle_control_cmd',
            CarlaEgoVehicleControl,
            queue_size=10,
        )
        self.pub_arrival = rospy.Publisher(
            '/carla/' + self.robot_id + '/arrival',
            Bool,
            queue_size=1,
        )

    def _set_zero_control(self):
        """Set control outputs to zero."""
        self.vel.linear.x = 0
        self.vel.angular.z = 0
        self.output.throttle = 0
        self.output.steer = 0

    def _load_reference_path(self):
        """Load the reference path from file."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(base_dir, 'utils', 'reference_path.txt'),
            os.path.join(sys.path[0], 'utils', 'reference_path.txt'),
        ]

        txt_path = next((path for path in candidate_paths if os.path.isfile(path)), None)
        if txt_path is None:
            raise FileNotFoundError(
                "reference_path.txt not found. Checked: "
                + ", ".join(candidate_paths)
            )

        point_array = np.atleast_2d(np.loadtxt(txt_path))
        point_list = [point.reshape(-1, 1) for point in point_array]
        return self._generate_line_path(point_list, step_size=0.5)

    @staticmethod
    def _generate_line_path(point_list, step_size=0.1):
        """Generate dense line segments between consecutive waypoints."""
        if len(point_list) == 0:
            return []
        if len(point_list) == 1:
            return [point_list[0]]

        curve = [point_list[0]]

        for i in range(len(point_list) - 1):
            start_point = point_list[i]
            end_point = point_list[i + 1]

            start_xy = np.asarray(start_point[0:2]).reshape(2, 1)
            end_xy = np.asarray(end_point[0:2]).reshape(2, 1)

            diff = end_xy - start_xy
            length = np.linalg.norm(diff)
            if length == 0.0:
                continue

            direction = diff / length
            theta = atan2(diff[1, 0], diff[0, 0])
            cur_len = 0.0

            while (cur_len + step_size) < length:
                cur_len += step_size
                new_xy = start_xy + cur_len * direction
                curve.append(
                    np.array([[new_xy[0, 0]], [new_xy[1, 0]], [theta]], dtype=float)
                )

            if end_point.shape == (3, 1):
                curve.append(end_point)
            else:
                curve.append(np.array([[end_point[0]], [end_point[1]], [theta]]))

        return curve

    def _run_mpc_step(self):
        """Run one MPC step and return control and planner info."""
        opt_vel, info = self.mpc_controller.control(
            self.robot_state, self.ref_path_list,
            ref_speed=self.ref_speed, ro=1.0,
            nav_weights=self.nav_weights
        )
        return opt_vel, info

    def _predictor_timer_callback(self, _event):
        """Standalone predictor loop at fixed frequency 1/dt."""
        try:
            if self.collided:
                return
            doppler_estimates = self.doppler_to_lidar()
            self.mpc_controller.update_prediction_buffer(doppler_estimates)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, f"Predictor loop failed: {exc}")

    def _build_visualization_paths(self, info):
        """Build paths and update prediction visualization."""
        if info.get('predictions') and info.get('moving_indices'):
            self.visualizer.update_prediction_visualization(
                info['predictions'],
                info['moving_indices'],
                info['prediction_horizon']
            )

        if self.planner == PLANNER_MODE_ACTIVE:
            opt_state = self.visualizer.create_optimal_horizon_trajectory(
                info['opt_state_list'], self.robot_state, self.z, skip_initial_steps=3
            )
        else:
            opt_state = self.visualizer.create_hidden_path(info['opt_state_list'], self.robot_state)

        return {
            'reference': self.path if self.planner == PLANNER_MODE_ACTIVE else None,
            'action': opt_state,
        }

    def _update_control_output(self, opt_vel, arrived):
        """Update control output from MPC result."""
        if arrived:
            self._set_zero_control()
            return

        self.vel.linear.x = round(opt_vel[0, 0], 2)
        self.vel.angular.z = round(opt_vel[1, 0], 2)
        self.output.throttle = round(opt_vel[0, 0], 2) / VELOCITY_SCALING
        self.output.steer = -round(opt_vel[1, 0], 2)

    def calculate_velocity(self, freq=50):
        """Main control loop that calculates and publishes velocity commands."""
        rate = rospy.Rate(freq)

        while not rospy.is_shutdown():
            if self.collided:
                self._set_zero_control()
                if self.planner == PLANNER_MODE_ACTIVE:
                    self.pub_vel.publish(self.output)
                rate.sleep()
                continue

            opt_vel, info = self._run_mpc_step()

            paths_dict = self._build_visualization_paths(info)
            arrived = bool(info['arrive'])
            self._update_control_output(opt_vel, arrived)
            if arrived and not self._arrival_published:
                self.pub_arrival.publish(Bool(data=True))
                self._arrival_published = True

            self.visualizer.publish_visualizations(paths_dict, show_car=True)
            if self.planner == PLANNER_MODE_ACTIVE:
                self.pub_vel.publish(self.output)

            rate.sleep()

    def _create_car_params(self):
        """Create car parameters tuple."""
        cone_type = rospy.get_param('cone_type', 'Rpositive')
        wheelbase = float(self.shape[2])
        abs_speed = rospy.get_param('abs_speed', 15)
        abs_steer = rospy.get_param('abs_steer', 0.6)
        abs_acce = rospy.get_param('abs_acce', 2)
        abs_acce_steer = rospy.get_param('abs_acce_steer', 0.03)

        vertex = DPNetPlanner._calculate_vehicle_vertices(self.shape[:3])
        G, h = generate_inequalities(vertex)

        return Car(G, h, cone_type, wheelbase, abs_speed, abs_steer, abs_acce, abs_acce_steer)

    def robot_state_callback(self, data):
        """Callback for robot state updates from odometry."""
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        z = data.pose.pose.position.z
        quat = data.pose.pose.orientation

        yaw = self.visualizer.quaternion_to_yaw(quat)
        yaw_degrees = yaw * 180 / pi
        self.x = x
        self.y = y
        self.z = z
        self.angle = yaw_degrees
        offset = self.shape[2] / 2
        self.robot_state[0] = x - offset * cos(yaw)
        self.robot_state[1] = y - offset * sin(yaw)
        self.robot_state[2] = yaw

        self.visualizer.update_car_marker(x, y, z, quat)

    def obstacle_callback(self, data):
        """Callback for obstacle detection updates."""
        obstacle_snapshot = []

        for object_data in data.objects:
            pose_x = object_data.pose.position.x
            pose_y = object_data.pose.position.y
            pose_z = object_data.pose.position.z
            quat = object_data.pose.orientation
            yaw = self.visualizer.quaternion_to_yaw(quat)
            vel_world_x = object_data.twist.linear.x
            vel_world_y = object_data.twist.linear.y
            obstacle_id = object_data.id

            if obstacle_id in self.obstacle_cache:
                cached_data = self.obstacle_cache[obstacle_id]
                length = cached_data['length']
                width = cached_data['width']
                height = cached_data['height']
                exact_block_vertex = cached_data['exact_block_vertex']
            else:
                raw_length = object_data.shape.dimensions[0]
                raw_width = object_data.shape.dimensions[1]
                raw_height = object_data.shape.dimensions[2]
                length = raw_length
                width = raw_width
                height = raw_height
                
                exact_block_vertex = create_rectangle_vertex(length, width)

                self.obstacle_cache[obstacle_id] = {
                    'length': length,
                    'width': width,
                    'height': height,
                    'exact_block_vertex': exact_block_vertex
                }

            rot_matrix = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])
            tran_matrix = np.array([[pose_x], [pose_y]])
            vertices = rot_matrix @ exact_block_vertex + tran_matrix
            A, b = generate_inequalities(vertices)

            obstacle = {
                'A': A,
                'b': b,
                'id': object_data.id,
                'x': pose_x,
                'y': pose_y,
                'z': pose_z,
                'yaw': yaw,
                'vel_world_x': vel_world_x,
                'vel_world_y': vel_world_y,
                'length': length,
                'width': width,
                'height': height,
                'z_min': pose_z + 0.05,
                'z_max': pose_z + height,
                'time': data.header.stamp.to_sec(),
            }

            obstacle_snapshot.append(obstacle)

        with self._sensor_lock:
            self.obstacle_callback_buffer = obstacle_snapshot

        self.mpc_controller.update_obstacle_snapshot(obstacle_snapshot)
        current_obstacle_ids = {obj['id'] for obj in obstacle_snapshot}
        cached_ids = set(self.obstacle_cache.keys())
        removed_ids = cached_ids - current_obstacle_ids
        
        for removed_id in removed_ids:
            del self.obstacle_cache[removed_id]

    def planner_callback(self, msg):
        """Callback for planner mode updates."""
        self.planner = msg.data

    def collision_callback(self, msg):
        """Callback for collision events."""
        if self.collided:
            return

        self.collided = True
        rospy.logwarn(f"Collision detected for {self.robot_id}.")

    def doppler_lidar_callback(self, msg):
        """Callback for Doppler LiDAR data processing."""
        try:
            point_iter = pc2.read_points(
                msg,
                field_names=('x', 'y', 'z', 'azimuth', 'elevation', 'velocity'),
                skip_nans=True
            )
            points_array = np.fromiter(point_iter, dtype=DOPPLER_POINT_DTYPE)
            if points_array.size == 0:
                with self._sensor_lock:
                    self.doppler_points_buffer = np.empty(0, dtype=DOPPLER_POINT_DTYPE)
                return

            x = points_array['x']
            y = points_array['y']
            z = points_array['z']
            azimuth = -points_array['azimuth']
            elevation = points_array['elevation']
            velocity = points_array['velocity']

            vehicle_x = self.x
            vehicle_y = self.y
            vehicle_yaw = self.robot_state[2]

            in_vehicle_x = x + self.position_offset[0]
            in_vehicle_y = y + self.position_offset[1]
            in_vehicle_z = z + self.position_offset[2]

            cos_yaw = np.cos(vehicle_yaw)
            sin_yaw = np.sin(vehicle_yaw)
            
            world_x = vehicle_x + in_vehicle_x * cos_yaw - in_vehicle_y * sin_yaw
            world_y = vehicle_y + in_vehicle_x * sin_yaw + in_vehicle_y * cos_yaw
            world_z = in_vehicle_z
            
            num_points = len(world_x)
            points_buffer = np.empty(num_points, dtype=DOPPLER_POINT_DTYPE)
            points_buffer['x'] = world_x
            points_buffer['y'] = world_y
            points_buffer['z'] = world_z
            points_buffer['azimuth'] = azimuth
            points_buffer['elevation'] = elevation
            points_buffer['velocity'] = velocity
            with self._sensor_lock:
                self.doppler_points_buffer = points_buffer
                    
        except Exception as e:
            rospy.logwarn(f"Error processing Doppler LiDAR data: {e}")
            with self._sensor_lock:
                self.doppler_points_buffer = np.empty(0, dtype=DOPPLER_POINT_DTYPE)

    @staticmethod
    def _apply_mask(mask, *arrays):
        """Apply a boolean mask to aligned arrays; return None if empty."""
        if not np.any(mask):
            return None
        return tuple(arr[mask] for arr in arrays)

    def find_points_in_obstacle(self):
        """Get Doppler LiDAR points associated with each detected obstacle using vectorized operations"""
        with self._sensor_lock:
            current_obstacles_snapshot = self.obstacle_callback_buffer.copy()
            doppler_points = self.doppler_points_buffer.copy()

        if len(doppler_points) == 0 or not current_obstacles_snapshot:
            return {}
        obstacle_points = {}

        def _build_obstacle_entry(obstacle, obstacle_x, obstacle_y):
            """Create per-obstacle payload with shared structure."""
            return {
                'points': np.empty(0, dtype=DOPPLER_POINT_DTYPE),
                'obstacle_info': {
                    'x': obstacle_x,
                    'y': obstacle_y,
                    'z': obstacle['z'],
                    'yaw': obstacle['yaw'],
                    'vel_world_x': obstacle['vel_world_x'],
                    'vel_world_y': obstacle['vel_world_y'],
                },
            }

        for obstacle in current_obstacles_snapshot:
            obstacle_id = obstacle['id']
            obstacle_x = obstacle['x']
            obstacle_y = obstacle['y']
            obstacle_points[obstacle_id] = _build_obstacle_entry(
                obstacle, obstacle_x, obstacle_y
            )

        point_x = doppler_points['x']
        point_y = doppler_points['y']
        point_z = doppler_points['z']
        points_2d = np.column_stack((point_x, point_y))
        point_indices = np.arange(len(doppler_points))

        for obstacle in current_obstacles_snapshot:
            obstacle_id = obstacle['id']
            z_mask = (point_z >= obstacle['z_min']) & (point_z <= obstacle['z_max'])
            if not np.any(z_mask):
                continue
            z_filtered_indices = point_indices[z_mask]
            z_filtered_points_2d = points_2d[z_mask]

            A, b = obstacle['A'], obstacle['b']
            constraint_checks = A @ z_filtered_points_2d.T <= b
            inside_bbox_mask = np.all(constraint_checks, axis=0)

            if not np.any(inside_bbox_mask):
                continue
            inside_indices = z_filtered_indices[inside_bbox_mask]

            if len(inside_indices) > 0:
                obstacle_points[obstacle_id]['points'] = doppler_points[inside_indices].copy()

        return obstacle_points

    def radial_to_linear(self, obstacle_data):
        """Doppler velocity rectification"""
        points = obstacle_data['points']
        obs_info = obstacle_data['obstacle_info']

        if points is None or len(points) == 0:
            return 0.0, 0.0, float('inf')

        yaw = obs_info['yaw']
        v_dir = np.array([np.cos(yaw), np.sin(yaw), 0])
        ego_pose = np.array([self.x, self.y, 0])
        obs_pose = np.array([obs_info['x'], obs_info['y'], 0])
        lidar_pose = np.array([
            self.x + self.position_offset[0],
            self.y + self.position_offset[1],
            self.position_offset[2]
        ])

        e2o_vec = ego_pose - obs_pose
        e2o_norm = np.linalg.norm(e2o_vec)
        if e2o_norm < 1e-3:
            return 0.0, 0.0, float('inf')
        e2o_unit = e2o_vec / e2o_norm

        cos_varpi = float(np.clip(np.dot(e2o_unit, v_dir), -1.0, 1.0))
        if abs(cos_varpi) < 1e-3:
            return 0.0, 0.0, float('inf')

        if isinstance(points, np.ndarray) and points.dtype.fields is not None:
            point_x = points['x'].astype(float, copy=False)
            point_y = points['y'].astype(float, copy=False)
            point_z = points['z'].astype(float, copy=False)
            velocities = points['velocity'].astype(float, copy=False)
            elevations = points['elevation'].astype(float, copy=False)
        else:
            point_x = np.array([p['x'] for p in points], dtype=float)
            point_y = np.array([p['y'] for p in points], dtype=float)
            point_z = np.array([p['z'] for p in points], dtype=float)
            velocities = np.array([p['velocity'] for p in points], dtype=float)
            elevations = np.array([p['elevation'] for p in points], dtype=float)

        l2p_x = point_x - lidar_pose[0]
        l2p_y = point_y - lidar_pose[1]
        l2p_z = point_z - lidar_pose[2]
        l2p_norms = np.hypot(l2p_x, l2p_y)

        initial_valid_mask = (
            (l2p_norms >= 1e-3) &
            (np.abs(elevations) <= 90.0) &
            (np.abs(velocities) <= 100.0)
        )
        masked = self._apply_mask(initial_valid_mask, l2p_x, l2p_y, l2p_z, l2p_norms, velocities, elevations)
        if masked is None:
            return 0.0, 0.0, float('inf')
        l2p_x, l2p_y, l2p_z, l2p_norms, velocities, elevations = masked

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_norm = 1.0 / l2p_norms
            l2p_unit_x = l2p_x * inv_norm
            l2p_unit_y = l2p_y * inv_norm
            l2p_unit_z = l2p_z * inv_norm
        l2p_projected_norms = np.hypot(l2p_unit_x, l2p_unit_y)

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_proj_norm = 1.0 / l2p_projected_norms
            l2p_projected_unit_x = l2p_unit_x * inv_proj_norm
            l2p_projected_unit_y = l2p_unit_y * inv_proj_norm

        cos_iotas = np.clip(
            l2p_projected_unit_x * v_dir[0] + l2p_projected_unit_y * v_dir[1],
            -1.0,
            1.0
        )
        geometric_valid_mask = (
            (l2p_projected_norms >= 1e-3) &
            np.isfinite(l2p_projected_unit_x) &
            np.isfinite(l2p_projected_unit_y) &
            np.isfinite(l2p_unit_z) &
            (np.abs(cos_iotas) >= 0.01)
        )
        masked = self._apply_mask(geometric_valid_mask, cos_iotas, velocities, elevations)
        if masked is None:
            return 0.0, 0.0, float('inf')
        cos_iotas, velocities, elevations = masked

        with np.errstate(invalid='ignore', over='ignore'):
            cos_elevations = np.cos(np.deg2rad(elevations))
            mu_breves = velocities / cos_elevations
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            radial_estimates = mu_breves * cos_varpi / cos_iotas

        final_valid_mask = (
            np.isfinite(mu_breves) &
            np.isfinite(radial_estimates) &
            (np.abs(mu_breves) <= 100.0) &
            (np.abs(radial_estimates) <= 100.0)
        )
        masked = self._apply_mask(final_valid_mask, radial_estimates)
        if masked is None:
            return 0.0, 0.0, float('inf')
        radial_estimates = masked[0]

        if len(radial_estimates) >= 3:
            clipped_estimates = np.clip(radial_estimates, -50.0, 50.0)
            with np.errstate(over='ignore', invalid='ignore'):
                estimate_variance = float(np.std(clipped_estimates))
            if not np.isfinite(estimate_variance):
                estimate_variance = float('inf')
        else:
            estimate_variance = float('inf')

        with np.errstate(over='ignore', invalid='ignore'):
            avg_radial_magnitude = float(np.median(radial_estimates))

        max_reasonable_velocity = 50.0
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            linear_estimate_magnitude = avg_radial_magnitude / cos_varpi
            
        linear_estimate_magnitude = np.clip(linear_estimate_magnitude, 
                                          -max_reasonable_velocity, 
                                          max_reasonable_velocity)

        with np.errstate(over='ignore', invalid='ignore'):
            v_x = linear_estimate_magnitude * v_dir[0]
            v_y = linear_estimate_magnitude * v_dir[1]
        
        if not np.all(np.isfinite([avg_radial_magnitude, linear_estimate_magnitude, v_x, v_y])):
            return 0.0, 0.0, float('inf')

        if abs(v_x) > 0.1 or abs(v_y) > 0.1:
            return float(v_x), float(v_y), estimate_variance
        else:
            return 0.0, 0.0, estimate_variance

    def doppler_to_lidar(self):
        """estimate obstacle linear velocity from doppler point cloud"""
        with np.errstate(over='ignore', invalid='ignore'):
            obs_linear_estimate = {}
            obstacle_points = self.find_points_in_obstacle()
            if not obstacle_points:
                return obs_linear_estimate

            for obstacle_id, obstacle_and_points in obstacle_points.items():
                points = obstacle_and_points['points']
                if len(points) > 10:
                    vel_x, vel_y, _ = self.radial_to_linear(obstacle_and_points)
                    obs_linear_estimate[obstacle_id] = (vel_x, vel_y) if abs(vel_x) > 0.05 or abs(
                        vel_y) > 0.05 else (0, 0)
                else:
                    obs_linear_estimate[obstacle_id] = (0, 0)

            return obs_linear_estimate

    @staticmethod
    def _calculate_vehicle_vertices(shape):
        """Calculate the vertices of the vehicle footprint in the body frame."""
        start_x = -(shape[0] - shape[2]) / 2
        start_y = -shape[1] / 2

        point0 = np.array([[start_x], [start_y]])  # left bottom point
        point1 = np.array([[start_x + shape[0]], [start_y]])
        point2 = np.array([[start_x + shape[0]], [start_y + shape[1]]])
        point3 = np.array([[start_x], [start_y + shape[1]]])

        return np.hstack((point0, point1, point2, point3))

if __name__ == '__main__':
    planner = None
    try:
        rospy.init_node('dpnet_planner_node', anonymous=True)
        robot_id = rospy.get_param('~robot_id', 'agent_1')
        planner = DPNetPlanner(robot_id)
        planner.calculate_velocity()

    except rospy.ROSInterruptException as exc:
        rospy.loginfo("DPNet planner interrupted: %s", exc)
    except KeyboardInterrupt:
        print("Shutting down...")
