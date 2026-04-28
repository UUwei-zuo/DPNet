#! /usr/bin/env python3

from math import cos, sin

import numpy as np
import rospy

# ROS message imports
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray


class PlannerVisualizer:
    """Handles visualization for the optimal planner."""

    def __init__(self, robot_id, vehicle_shape):
        """Initialize visualizer state and publishers."""
        self.robot_id = robot_id
        self.shape = vehicle_shape
        self.marker_array_car = MarkerArray()
        self._carla_reference_drawn = False

        self._setup_publishers()

    def _setup_publishers(self):
        """Set up ROS publishers."""
        self.pub_marker_car = rospy.Publisher(f'car_marker_{self.robot_id}', MarkerArray, queue_size=10)
        self.pub_ref_path = rospy.Publisher(f'ref_path__{self.robot_id}', Path, queue_size=10)
        self.pub_action_path = rospy.Publisher(f'action__{self.robot_id}', Path, queue_size=10)
        self.pub_prediction_markers = rospy.Publisher(f'prediction_markers_{self.robot_id}', MarkerArray, queue_size=10)
        self.pub_carla_debug_markers = rospy.Publisher('/carla/debug_marker', MarkerArray, queue_size=1)

    @staticmethod
    def create_carla_reference_marker(path_msg, marker_id_base=9000):
        """Sparse CARLA ``Marker.ARROW`` markers along the reference path (red gradient)."""
        poses = path_msg.poses
        marker_array = MarkerArray()
        if len(poses) < 2:
            return marker_array

        z_arrow = 0.16
        r0, g0, b0, a0 = 0.42, 0.03, 0.04, 0.15
        r1, g1, b1, a1 = 0.98, 0.48, 0.42, 0.05
        n_seg = len(poses) - 1

        ARROW_ID_OFFSET = 50000
        max_arrows = 16
        arrow_stride = max(1, n_seg // max_arrows)
        reach = max(1, min(10, max(2, n_seg // 8)))

        def _path_point(idx, z_off):
            p = Point()
            ps = poses[idx].pose.position
            p.x = ps.x
            p.y = ps.y
            p.z = ps.z + z_off
            return p

        arrow_indices = sorted(set(list(range(0, n_seg, arrow_stride)) + ([n_seg - 1] if n_seg else [])))
        aid = 0
        for i in arrow_indices:
            j = min(i + reach, len(poses) - 1)
            if j <= i:
                j = i + 1
            if j >= len(poses):
                continue

            t = i / float(max(n_seg - 1, 1))
            r = r0 + (r1 - r0) * t
            g = g0 + (g1 - g0) * t
            b = b0 + (b1 - b0) * t
            a = min(0.85, a0 + (a1 - a0) * t + 0.12)

            am = Marker()
            am.header.frame_id = 'map'
            am.header.stamp = rospy.get_rostime()
            am.ns = "dpnet_reference_flow"
            am.id = marker_id_base + ARROW_ID_OFFSET + aid
            am.type = Marker.ARROW
            am.action = Marker.ADD
            am.scale.x = 0.10
            am.scale.y = 0.20
            am.color.r = r
            am.color.g = g
            am.color.b = b
            am.color.a = a
            am.lifetime = rospy.Duration(0)
            am.points.append(_path_point(i, z_arrow))
            am.points.append(_path_point(j, z_arrow))
            marker_array.markers.append(am)
            aid += 1

        return marker_array

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        """Convert a quaternion into a 3x3 rotation matrix."""
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return np.eye(3)

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ]
        )

    def create_car_marker(self, x, y, z, quat):
        """Create RGB 3D coordinate axes markers for the ego vehicle."""
        rotation_matrix = self.quaternion_to_rotation_matrix(quat)
        axis_length = max(1.2, float(self.shape[0]) * 0.8)

        axis_specs = [
            ("ego_axes_x", 100, (1.0, 0.0, 0.0), rotation_matrix[:, 0]),
            ("ego_axes_y", 101, (0.0, 1.0, 0.0), rotation_matrix[:, 1]),
            ("ego_axes_z", 102, (0.0, 0.0, 1.0), rotation_matrix[:, 2]),
        ]

        markers = []
        for namespace, marker_id, color, axis_direction in axis_specs:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.get_rostime()
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.30
            marker.scale.y = 0.50
            marker.scale.z = 0.40
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0)

            start_point = Point()
            start_point.x = x
            start_point.y = y
            start_point.z = z

            end_point = Point()
            end_point.x = x + float(axis_direction[0]) * axis_length
            end_point.y = y + float(axis_direction[1]) * axis_length
            end_point.z = z + float(axis_direction[2]) * axis_length

            marker.points = [start_point, end_point]
            markers.append(marker)

        return markers

    def update_car_marker(self, x, y, z, quat):
        """Update car marker data."""
        self.marker_array_car.markers = self.create_car_marker(x, y, z, quat)

    def create_reference_path(self, path_points, frame_id='map'):
        """Create a path message from path points."""
        path = Path()
        path.header.seq = 0
        path.header.stamp = rospy.get_rostime()
        path.header.frame_id = frame_id

        for i in range(len(path_points)):
            ps = PoseStamped()
            ps.pose.position.x = path_points[i][0, 0]
            ps.pose.position.y = path_points[i][1, 0]
            ps.pose.position.z = 0
            path.poses.append(ps)

        return path

    def create_optimal_horizon_trajectory(self, path_points, robot_state, z_height, skip_initial_steps=2):
        """Create a horizon trajectory path for visualization."""
        path = Path()
        path.header.seq = 0
        path.header.stamp = rospy.get_rostime()
        path.header.frame_id = 'map'

        yaw = robot_state[2]
        offset = self.shape[2] / 2
        start_idx = min(skip_initial_steps, len(path_points) - 1) if skip_initial_steps > 0 else 0

        for i in range(start_idx, len(path_points)):
            ps = PoseStamped()
            ps.pose.position.x = path_points[i][0, 0] + offset * cos(yaw)
            ps.pose.position.y = path_points[i][1, 0] + offset * sin(yaw)
            ps.pose.position.z = z_height + 1.5
            ps.pose.orientation.w = 1

            path.poses.append(ps)

        return path

    def create_hidden_path(self, path_points, robot_state):
        """Create an off-map path to hide visualization."""
        path = Path()
        path.header.seq = 0
        path.header.stamp = rospy.get_rostime()
        path.header.frame_id = 'map'

        yaw = robot_state[2]
        offset = self.shape[2]
        ps = PoseStamped()
        ps.pose.position.x = 1000 + path_points[0][0] + offset * cos(yaw)
        ps.pose.position.y = 1000 + path_points[1][0] + offset * sin(yaw)
        ps.pose.position.z = 0.5
        ps.pose.orientation.w = 1

        path.poses.append(ps)
        return path

    def create_prediction_markers(self, prediction, obs_idx, prediction_horizon, frame_id='map'):
        """Create markers for predicted obstacle states."""
        markers = []
        
        if not prediction or len(prediction) == 0:
            return markers

        marker_id = obs_idx * 1000
        for step_idx, predicted_state in enumerate(prediction):
            if step_idx == 0:
                continue
                
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.get_rostime()
            marker.ns = f"predictions_obs_{obs_idx}"
            marker.id = marker_id + step_idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            x = predicted_state[0].item()
            y = predicted_state[3].item()
            
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.5 + step_idx * 0.1
            marker.pose.orientation.w = 1.0

            base_scale = 0.9
            scale_factor = max(0.3, 1.0 - (step_idx / prediction_horizon) * 0.5)
            marker.scale.x = base_scale * scale_factor
            marker.scale.y = base_scale * scale_factor
            marker.scale.z = base_scale * scale_factor
            
            marker.color.r = 1.0
            marker.color.g = 0.3
            marker.color.b = 0.0
            marker.color.a = max(0.2, 1.0 - (step_idx / prediction_horizon) * 0.9)
            marker.lifetime = rospy.Duration(0.5)
            
            markers.append(marker)
        
        return markers

    def update_prediction_visualization(self, predictions, moving_indices, prediction_horizon):
        """Update prediction markers from predicted trajectories."""
        if not predictions or not moving_indices:
            self.prediction_markers = MarkerArray()
            return

        all_markers = []

        for obs_idx, (prediction, _moving_idx) in enumerate(
            zip(predictions, moving_indices)
        ):
            sphere_markers = self.create_prediction_markers(
                prediction, obs_idx, prediction_horizon
            )
            all_markers.extend(sphere_markers)

        self.prediction_markers = MarkerArray()
        self.prediction_markers.markers = all_markers

    def publish_visualizations(self, paths_dict, show_car=True):
        """Publish all configured paths and markers."""
        if 'reference' in paths_dict and paths_dict['reference'] is not None:
            self.pub_ref_path.publish(paths_dict['reference'])
            if not self._carla_reference_drawn:
                carla_marker = self.create_carla_reference_marker(paths_dict['reference'])
                self.pub_carla_debug_markers.publish(carla_marker)
                self._carla_reference_drawn = True

        if 'action' in paths_dict and paths_dict['action'] is not None:
            self.pub_action_path.publish(paths_dict['action'])

        if show_car:
            self.pub_marker_car.publish(self.marker_array_car)

        if hasattr(self, 'prediction_markers') and self.prediction_markers.markers:
            self.pub_prediction_markers.publish(self.prediction_markers)

    @staticmethod
    def quaternion_to_yaw(quaternion):
        """Convert a quaternion to yaw."""
        w = quaternion.w
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z

        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (z ** 2 + y ** 2))
