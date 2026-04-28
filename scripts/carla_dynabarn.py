#!/usr/bin/env python

"""Spawn and control dynamic CARLA obstacle vehicles."""

import argparse
import atexit
import logging
import math
import random
import signal
import sys
import threading
import time

import carla
import numpy as np
import scipy.stats as ss

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_AREA_BOUNDS = [-310, -220, -290, -140]
DEFAULT_SPAWN_RANDOM_AREA_BOUNDS = [-360, -230, -300, -160]
DEFAULT_SPECTATOR_ROT = [-50, 90, 0]

_MAIN_STOP_EVENT = threading.Event()
_VEHICLE_CONTROLLERS = []
_CLEANUP_STARTED = False

def _signal_handler(signum, frame):
    global _CLEANUP_STARTED
    try:
        logger.info(f"Received signal {signum}. Stopping spawn_obs_random...")
        _MAIN_STOP_EVENT.set()
        if not _CLEANUP_STARTED:
            _cleanup_all_vehicles()
    except Exception:
        logger.error("Error while handling termination signal")
    finally:
        sys.exit(0)

def _cleanup_all_vehicles():
    """Stop and clean up all active vehicle controllers."""
    global _CLEANUP_STARTED, _VEHICLE_CONTROLLERS
    if _CLEANUP_STARTED:
        return
    _CLEANUP_STARTED = True
    
    logger.info("Cleaning up all vehicles...")
    for controller in _VEHICLE_CONTROLLERS:
        try:
            controller.stop()
        except Exception:
            logger.warning("Error stopping a vehicle controller")
    
    _VEHICLE_CONTROLLERS.clear()
    logger.info("All vehicles cleaned up.")

def _register_shutdown_handlers():
    """Register cleanup handlers for direct script execution."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup_all_vehicles)


def set_global_random_seed(seed):
    """Set process-wide random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Global random seed set")


def ros_to_carla(x, y, z):
    """Convert ROS coordinates to CARLA coordinates."""
    return x, -y, z


def carla_to_ros(x, y, z):
    """Convert CARLA coordinates to ROS coordinates."""
    return x, -y, z


def generate_random_point_in_area(area_bounds):
    """Generate a random point within area bounds."""
    x_min, y_min, x_max, y_max = area_bounds
    
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)
    
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = 0.5
    
    return [x, y, z]


def calculate_distance(point1, point2):
    """Compute Euclidean distance between two 2D points."""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])


def is_spawn_position_valid(new_position, existing_positions, min_distance):
    """Check whether a new spawn position is sufficiently separated."""
    for existing_pos in existing_positions:
        distance = calculate_distance(new_position[:2], existing_pos[:2])
        if distance < min_distance:
            return False
    return True


def gen_points(n, area_bounds):
    """Generate random points used for polynomial fitting."""
    x_min, y_min, x_max, y_max = area_bounds
    x_s = []
    y_s = []
    for i in range(n):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        x_s.append(x)
        y_s.append(y)
    return x_s, y_s


def get_random_polynomial_trajectory(order, area_bounds, num_waypoints=10):
    """Generate a polynomial trajectory and sampled waypoints."""
    x_min, y_min, x_max, y_max = area_bounds
    
    n = order
    x_s, y_s = gen_points(n + 1, area_bounds)
    
    z = np.polyfit(x_s, y_s, n)
    p = np.poly1d(z)
    
    xp = np.linspace(x_min, x_max, num_waypoints)
    pxp = p(xp)
    
    x_max_boundary = (p - x_max).roots
    x_min_boundary = (p - x_min).roots
    y_max_boundary = p(x_max) if not np.isnan(p(x_max)) else None
    y_min_boundary = p(x_min) if not np.isnan(p(x_min)) else None
    
    if np.any(~np.iscomplex(x_max_boundary)):
        for x_val in x_max_boundary.real:
            if x_min <= x_val <= x_max:
                y_val = p(x_val)
                if y_min <= y_val <= y_max:
                    x_s = np.append(x_s, x_val)
                    y_s = np.append(y_s, y_max)
    
    if np.any(~np.iscomplex(x_min_boundary)):
        for x_val in x_min_boundary.real:
            if x_min <= x_val <= x_max:
                y_val = p(x_val)
                if y_min <= y_val <= y_max:
                    x_s = np.append(x_s, x_val)
                    y_s = np.append(y_s, y_min)
    
    if y_max_boundary is not None and y_min <= y_max_boundary <= y_max:
        x_s = np.append(x_s, x_max)
        y_s = np.append(y_s, y_max_boundary)
    
    if y_min_boundary is not None and y_min <= y_min_boundary <= y_max:
        x_s = np.append(x_s, x_min)
        y_s = np.append(y_s, y_min_boundary)
    
    xp = np.append(xp, x_s)
    pxp = np.append(pxp, y_s)

    points = []
    for i in range(len(xp)):
        if (x_min <= xp[i] <= x_max) and (y_min <= pxp[i] <= y_max):
            points.append((xp[i], pxp[i]))

    seen = set()
    uniq = [x for x in points if x not in seen and not seen.add(x)]
    uniq = sorted(uniq, key=lambda x: x[0])

    border_points = []
    normal_points = []
    for point in uniq:
        if (point[0] == x_min) or (point[0] == x_max) or (point[1] == y_min) or (point[1] == y_max):
            border_points.append(point)
        else:
            normal_points.append(point)

    border_points = sorted(border_points, key=lambda x: x[0])
    if len(border_points) == 0:
        for _ in range(2):
            x_border = random.choice([x_min, x_max])
            y_val = random.uniform(y_min, y_max)
            border_points.append((x_border, y_val))
    elif len(border_points) == 1:
        x_border = random.choice([x_min, x_max])
        y_val = random.uniform(y_min, y_max)
        border_points.append((x_border, y_val))

    start_border_point = border_points.pop(0)
    end_border_point = border_points.pop(-1) if border_points else border_points[0] if border_points else start_border_point

    if not border_points:
        uniq = normal_points + [start_border_point, end_border_point]
    else:
        uniq = normal_points + border_points

    uniq = sorted(uniq, key=lambda x: x[0])
    uniq.insert(0, start_border_point)
    uniq.append(end_border_point)

    if random.choice([0, 1]) < 0.5:
        uniq.reverse()

    new_x = [point[0] for point in uniq]
    new_y = [point[1] for point in uniq]
    
    return new_x, new_y, uniq


class PolynomialTrajectoryVehicleController:
    """Polynomial waypoint navigation inside a bounded ROS-area."""

    def __init__(self, world, vehicle_id, blueprint_library, client, area_bounds,
                 avg_throttle=5.0, min_throttle=2.0, max_throttle=10.0, throttle_std=2.0,
                 start_delay=0.0, blueprint='vehicle.tesla.model3',
                 polynomial_order=3, num_waypoints=10,
                 ego_role_name='agent_1', min_respawn_distance_to_ego=20.0,
                 max_respawn_sampling_attempts=100):
        """area_bounds: [x_min,y_min,x_max,y_max] ROS; client is kept for compatibility."""
        self.world = world
        self.vehicle_id = vehicle_id
        self.blueprint_library = blueprint_library
        self.vehicle = None
        self.stop_event = threading.Event()
        self.control_thread = None

        self.area_bounds = area_bounds
        self.polynomial_order = polynomial_order
        self.num_waypoints = num_waypoints
        self.current_waypoint = None
        self.trajectory_waypoints = []
        self.trajectory_index = 0

        self.spawn_point = generate_random_point_in_area(area_bounds)

        self.avg_throttle = avg_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.throttle_std = throttle_std
        self.current_throttle = self.generate_random_throttle()

        self.start_delay = start_delay
        self.blueprint = blueprint
        self.throttle_scale = 10.0
        self.ego_role_name = ego_role_name
        self.min_respawn_distance_to_ego = max(0.0, float(min_respawn_distance_to_ego))
        self.max_respawn_sampling_attempts = max(1, int(max_respawn_sampling_attempts))

        self.initial_heading = self.calculate_initial_heading()

        self.stuck_duration_threshold_s = 3.0
        self.movement_distance_epsilon_m = 0.2
        self.last_position_xy = None
        self.last_movement_time_s = time.time()
        self.last_respawn_time_s = 0.0
        self.respawn_cooldown_s = 1.0


    def get_ego_position_ros(self):
        """Return ego [x,y,z] in ROS frame, or None."""
        if not self.ego_role_name:
            return None

        try:
            for actor in self.world.get_actors().filter('vehicle.*'):
                role_name = actor.attributes.get('role_name', '')
                if role_name == self.ego_role_name:
                    location = actor.get_location()
                    return list(carla_to_ros(location.x, location.y, location.z))
        except Exception:
            logger.warning(f"{self.vehicle_id}: Failed to query ego vehicle '{self.ego_role_name}'")
        return None

    def sample_respawn_point_away_from_ego(self):
        """Sample a spawn point at least min distance from ego, or None."""
        ego_position = self.get_ego_position_ros()
        if ego_position is None:
            logger.warning(
                f"{self.vehicle_id}: Ego '{self.ego_role_name}' not found for respawn check"
            )
            return None

        for _ in range(self.max_respawn_sampling_attempts):
            candidate = generate_random_point_in_area(self.area_bounds)
            if calculate_distance(candidate[:2], ego_position[:2]) >= self.min_respawn_distance_to_ego:
                return candidate

            logger.warning(f"{self.vehicle_id}: Respawn search limit reached")
        return None

    @staticmethod
    def _normalize_angle_pi(angle_rad):
        """Wrap an angle into [-pi, pi]."""
        return (angle_rad + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _normalize_xy_vector(direction_x, direction_y):
        """Normalize a 2D vector to unit length."""
        direction_length = math.hypot(direction_x, direction_y)
        if direction_length <= 0:
            return 0.0, 0.0
        return direction_x / direction_length, direction_y / direction_length

    def _wait_for_start_delay(self):
        """Apply configured start delay with stop-aware sleep."""
        if self.start_delay <= 0:
            return True

        logger.info(f"Vehicle {self.vehicle_id} waiting {self.start_delay} seconds before starting movement...")
        if self.stop_event.wait(self.start_delay):
            logger.info(f"Vehicle {self.vehicle_id} received stop signal during delay period")
            return False
        return True

    def _navigation_step(self, log_waypoint_reached=False):
        """Get one navigation step state."""
        if self.current_waypoint is None:
            if self.stop_event.wait(0.1):
                return None, True
            return None, False

        carla_waypoint = ros_to_carla(*self.current_waypoint)
        current_location = self.vehicle.get_location()

        if self.update_and_check_stuck(current_location):
            if self.stop_event.wait(0.2):
                return None, True
            return None, False

        distance = calculate_distance(
            (carla_waypoint[0], carla_waypoint[1]),
            (current_location.x, current_location.y),
        )
        if distance < 5.0:
            if log_waypoint_reached:
                logger.info(f"Vehicle {self.vehicle_id} reached waypoint, generating next one...")
            self.generate_next_waypoint()
            self.last_movement_time_s = time.time()
            if self.stop_event.wait(1.0):
                return None, True
            return None, False

        direction_x = carla_waypoint[0] - current_location.x
        direction_y = carla_waypoint[1] - current_location.y

        return (current_location, carla_waypoint, direction_x, direction_y), False

    def _stop_vehicle_motion(self, brake_duration_s):
        """Stop the vehicle with a brake control command."""
        if not self.vehicle:
            return

        try:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            if brake_duration_s > 0:
                time.sleep(brake_duration_s)
        except Exception:
            logger.debug(f"{self.vehicle_id}: Error while applying stop command")

    def calculate_initial_heading(self):
        """Heading (deg) toward first temp trajectory waypoint, ±30° jitter."""
        try:
            temp_waypoints = get_random_polynomial_trajectory(
                self.polynomial_order, self.area_bounds, min(5, self.num_waypoints)
            )[2]

            if temp_waypoints and len(temp_waypoints) > 0:
                first_waypoint = temp_waypoints[0]
                dx = first_waypoint[0] - self.spawn_point[0]
                dy = first_waypoint[1] - self.spawn_point[1]

                target_angle = math.degrees(math.atan2(dy, dx))

                if target_angle < 0:
                    target_angle += 360

                offset = random.uniform(-30, 30)
                initial_heading = (target_angle + offset) % 360

                return initial_heading
            else:
                logger.warning(f"Vehicle {self.vehicle_id}: Could not generate waypoint for heading calculation, using random heading")
                return random.uniform(0, 360)

        except Exception:
            logger.warning(f"Vehicle {self.vehicle_id}: Initial heading error, using random heading")
            return random.uniform(0, 360)

    def generate_random_throttle(self):
        """Sample throttle level from truncated normal over [min_throttle, max_throttle)."""
        throttles = np.arange(self.min_throttle, self.max_throttle, 0.1)
        if len(throttles) == 0:
            return self.avg_throttle

        prob = ss.norm.pdf(throttles, loc=self.avg_throttle, scale=self.throttle_std)
        prob = prob / prob.sum()

        throttle = 0
        while throttle == 0 or throttle < 0:
            throttle = np.random.choice(throttles, 1, p=prob)[0]

        return throttle

    def generate_next_waypoint(self):
        """Advance along polynomial waypoints; start a new trajectory when exhausted."""
        if not self.trajectory_waypoints or self.trajectory_index >= len(self.trajectory_waypoints):
            try:
                _, _, uniq = get_random_polynomial_trajectory(self.polynomial_order, self.area_bounds, self.num_waypoints)
                self.trajectory_waypoints = uniq
                self.trajectory_index = 0
                self.current_throttle = self.generate_random_throttle()
            except Exception:
                logger.error(f"Vehicle {self.vehicle_id} failed to generate polynomial trajectory")
                raise

        if self.trajectory_index < len(self.trajectory_waypoints):
            waypoint_tuple = self.trajectory_waypoints[self.trajectory_index]
            self.current_waypoint = [waypoint_tuple[0], waypoint_tuple[1], 0.5]
            self.trajectory_index += 1
            return self.current_waypoint
        else:
            return self.generate_next_waypoint()

    def spawn_vehicle(self):
        """Spawn this controller's vehicle at spawn_point with initial_heading."""
        if self.vehicle is None:
            try:
                carla_coords = ros_to_carla(*self.spawn_point)

                transform = carla.Transform(
                    carla.Location(x=carla_coords[0], y=carla_coords[1], z=carla_coords[2]),
                    carla.Rotation(yaw=self.initial_heading)
                )

                vehicle_blueprints = self.blueprint_library.filter(self.blueprint)

                if not vehicle_blueprints:
                    logger.warning(f"No '{self.blueprint}' blueprint found for {self.vehicle_id}. Trying generic vehicle.")
                    vehicle_blueprints = self.blueprint_library.filter('vehicle.*')

                vehicle_bp = random.choice(vehicle_blueprints)

                if vehicle_bp.has_attribute('role_name'):
                    vehicle_bp.set_attribute('role_name', self.vehicle_id)

                self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)

                if self.vehicle:
                    logger.info(f"Vehicle {self.vehicle_id} spawned with blueprint '{self.blueprint}' at {self.spawn_point}")
                    return True
                else:
                    logger.error(f"Failed to spawn vehicle {self.vehicle_id} at {transform.location}")
                    return False
            except Exception:
                logger.error(f"Error spawning vehicle {self.vehicle_id}")
                return False
        return True

    def start_control(self):
        """Start background control thread."""
        if self.vehicle and not self.control_thread:
            self.control_thread = threading.Thread(target=self.manual_control_vehicle)
            logger.info(f"Starting manual throttle control for vehicle {self.vehicle_id}")
            self.control_thread.daemon = True
            self.control_thread.start()
            return True
        elif not self.vehicle:
            logger.error(f"Cannot start control for vehicle {self.vehicle_id}: vehicle not spawned")
            return False
        return True

    def respawn_vehicle_random(self):
        """Destroy current vehicle and respawn at a new random position within the area."""
        try:
            safe_spawn_point = self.sample_respawn_point_away_from_ego()
            if safe_spawn_point is None:
                return False

            if self.vehicle:
                self._stop_vehicle_motion(0.2)
                try:
                    self.vehicle.destroy()
                except Exception:
                    logger.warning(f"{self.vehicle_id}: Failed to destroy vehicle during respawn")
                finally:
                    self.vehicle = None

            self.spawn_point = safe_spawn_point
            self.initial_heading = self.calculate_initial_heading()
            logger.info(f"{self.vehicle_id}: Respawning at new point {self.spawn_point} with heading {self.initial_heading:.1f}°")

            if self.spawn_vehicle() and self.vehicle:
                try:
                    self.vehicle.set_autopilot(False)
                except Exception:
                    pass
                self.current_waypoint = None
                self.trajectory_waypoints = []
                self.trajectory_index = 0
                self.generate_next_waypoint()
                self.last_position_xy = None
                self.last_movement_time_s = time.time()
                return True
            else:
                logger.error(f"{self.vehicle_id}: Respawn failed")
                return False
        except Exception:
            logger.error(f"{self.vehicle_id}: Respawn failed")
            return False

    def update_and_check_stuck(self, current_location):
        """Return True if stuck handling triggered a respawn."""
        try:
            now_s = time.time()
            current_xy = (current_location.x, current_location.y)

            if self.last_position_xy is None:
                self.last_position_xy = current_xy
                self.last_movement_time_s = now_s
                return False
            
            distance_moved_m = math.hypot(current_xy[0] - self.last_position_xy[0],
                                           current_xy[1] - self.last_position_xy[1])
            
            if distance_moved_m >= self.movement_distance_epsilon_m:
                self.last_position_xy = current_xy
                self.last_movement_time_s = now_s
                return False

            time_without_movement_s = now_s - self.last_movement_time_s
            if time_without_movement_s > self.stuck_duration_threshold_s:
                if (now_s - self.last_respawn_time_s) >= self.respawn_cooldown_s:
                    self.last_respawn_time_s = now_s
                    logger.warning(f"{self.vehicle_id}: Detected stuck for {time_without_movement_s:.2f}s; respawning...")
                    self.respawn_vehicle_random()
                    return True
            return False
        except Exception:
            logger.debug(f"{self.vehicle_id}: Stuck detection failed")
            return False

    def manual_control_vehicle(self):
        """Drive with throttle/steer toward waypoints."""
        try:
            if not self.vehicle:
                logger.error(f"Vehicle {self.vehicle_id} does not exist, cannot control")
                return

            if not self._wait_for_start_delay():
                return

            self.generate_next_waypoint()

            while not self.stop_event.is_set() and not _MAIN_STOP_EVENT.is_set():
                navigation_step, should_break = self._navigation_step(log_waypoint_reached=False)
                if should_break:
                    break
                if navigation_step is None:
                    continue

                _, _, direction_x, direction_y = navigation_step

                transform = self.vehicle.get_transform()
                forward_vector = transform.get_forward_vector()

                heading_angle = math.atan2(forward_vector.y, forward_vector.x)
                target_angle = math.atan2(direction_y, direction_x)
                angle_diff = self._normalize_angle_pi(target_angle - heading_angle)

                steer = max(min(angle_diff / math.pi, 1.0), -1.0)

                throttle = min(self.current_throttle / self.throttle_scale, 1.0)
                if abs(steer) > 0.3:
                    throttle = throttle * 0.6

                control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0, 
                                             hand_brake=False, reverse=False)
                self.vehicle.apply_control(control)

                if self.stop_event.wait(0.1):
                    break

        except Exception:
            logger.error(f"Error in manual control for {self.vehicle_id}")
        finally:
            logger.info(f"Vehicle {self.vehicle_id} manual control thread finished")

    def stop(self):
        """Signal control thread to exit and destroy the vehicle."""
        try:
            logger.info(f"Stopping vehicle {self.vehicle_id}...")
            self.stop_event.set()

            if self.control_thread and self.control_thread.is_alive():
                logger.info(f"Waiting for control thread of {self.vehicle_id} to finish...")
                self.control_thread.join(timeout=3.0)
                if self.control_thread.is_alive():
                    logger.warning(f"Control thread for {self.vehicle_id} did not finish in time")

            if self.vehicle:
                try:
                    self._stop_vehicle_motion(0.3)
                    self.vehicle.destroy()
                    logger.info(f"Vehicle {self.vehicle_id} destroyed")
                    self.vehicle = None
                except Exception:
                    logger.warning(f"Failed to destroy vehicle {self.vehicle_id}")
        except Exception:
            logger.error(f"Error stopping vehicle {self.vehicle_id}")



def set_spectator_camera(world, position, rotation):
    """Move spectator to ROS position with pitch/yaw/roll (deg); yaw negated for CARLA."""
    try:
        spectator = world.get_spectator()
        carla_coords = ros_to_carla(*position)
        transform = carla.Transform(
            carla.Location(x=carla_coords[0], y=carla_coords[1], z=carla_coords[2]),
            carla.Rotation(pitch=rotation[0], yaw=-rotation[1], roll=rotation[2])
        )
        spectator.set_transform(transform)
        logger.info(f"Spectator camera set to position: {position}, rotation: {rotation}")
        return True
    except Exception:
        logger.error("Error setting spectator camera")
        return False


def spawn_random_vehicles(num_vehicles=3, area_bounds=None,
                         blueprint='vehicle.tesla.model3', avg_throttle=5.0, min_throttle=2.0,
                         max_throttle=10.0, throttle_std=2.0, delays=0.0,
                         polynomial_order=3, num_waypoints=10, spectator_config=None,
                         random_seed=None, min_spawn_distance=10.0,
                         ego_role_name='agent_1', min_respawn_distance_to_ego=20.0):
    """Connect to CARLA, spawn separated dynamic vehicles, run until interrupted."""
    if area_bounds is None:
        area_bounds = list(DEFAULT_SPAWN_RANDOM_AREA_BOUNDS)

    if random_seed is not None:
        set_global_random_seed(random_seed)

    client = None
    vehicle_controllers = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        try:
            world = client.get_world()
            logger.info("Connected to CARLA server successfully")
        except Exception:
            logger.error("Failed to connect to CARLA server")
            return

        if spectator_config:
            set_spectator_camera(
                world,
                spectator_config.get("position", [-330, -200, 50]),
                spectator_config.get("rotation", [-60, 0, 0])
            )
            time.sleep(0.2)

        blueprint_library = world.get_blueprint_library()

        for i in range(num_vehicles):
            vehicle_id = f"random_vehicle_{i+1}"

            start_delay = delays if delays else 0.0
            
            controller = PolynomialTrajectoryVehicleController(
                world=world,
                vehicle_id=vehicle_id,
                blueprint_library=blueprint_library,
                client=client,
                area_bounds=area_bounds,
                avg_throttle=avg_throttle,
                min_throttle=min_throttle,
                max_throttle=max_throttle,
                throttle_std=throttle_std,
                start_delay=start_delay,
                blueprint=blueprint,
                polynomial_order=polynomial_order,
                num_waypoints=num_waypoints,
                ego_role_name=ego_role_name,
                min_respawn_distance_to_ego=min_respawn_distance_to_ego
            )
            vehicle_controllers.append(controller)
            _VEHICLE_CONTROLLERS.append(controller)

        spawned_count = 0
        spawned_positions = []
        max_spawn_attempts = 50

        for controller in vehicle_controllers:
            valid_position_found = False
            attempts = 0

            while not valid_position_found and attempts < max_spawn_attempts:
                attempts += 1

                if is_spawn_position_valid(controller.spawn_point, spawned_positions, min_spawn_distance):
                    success = controller.spawn_vehicle()
                    if success:
                        controller.vehicle.set_autopilot(False)
                        spawned_positions.append(controller.spawn_point)
                        spawned_count += 1
                        logger.info(f"Vehicle {controller.vehicle_id} spawned successfully at {controller.spawn_point} (attempt {attempts})")
                        valid_position_found = True
                    else:
                        logger.debug(f"Vehicle {controller.vehicle_id} spawn failed at {controller.spawn_point}, trying new position")
                        controller.spawn_point = generate_random_point_in_area(area_bounds)
                else:
                    logger.debug(f"Vehicle {controller.vehicle_id} spawn position {controller.spawn_point} too close to existing vehicles, trying new position")
                    controller.spawn_point = generate_random_point_in_area(area_bounds)

            if not valid_position_found:
                logger.warning(f"Could not find valid spawn position for vehicle {controller.vehicle_id} after {max_spawn_attempts} attempts")

        if spawned_count == 0:
            logger.error("No vehicles were spawned successfully")
            return

        logger.info(f"Successfully spawned {spawned_count}/{len(vehicle_controllers)} vehicles with minimum {min_spawn_distance}m separation")

        time.sleep(1.2)
        logger.info("Waiting for ROS bridge to detect entities...")
        time.sleep(1.0)

        for controller in vehicle_controllers:
            if controller.vehicle:
                controller.start_control()

        try:
            logger.info("Press Ctrl+C to stop the simulation")
            while not _MAIN_STOP_EVENT.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in main loop")
        finally:
            _MAIN_STOP_EVENT.set()

    except Exception:
        logger.error("Error in spawn control loop")
    finally:
        if not _CLEANUP_STARTED:
            _cleanup_all_vehicles()
        logger.info("Script finished")


if __name__ == "__main__":
    try:
        _register_shutdown_handlers()
        parser = argparse.ArgumentParser(description='Spawn vehicles with random waypoint navigation in CARLA')
        
        parser.add_argument('--vehicles', type=int, default=4,
                            help='Number of vehicles to spawn')
        
        parser.add_argument('--area', type=float, nargs=4, default=DEFAULT_AREA_BOUNDS,
                            metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
                            help='Area bounds [x_min, y_min, x_max, y_max] in ROS coordinates')

        parser.add_argument('--blueprint', type=str, default='vehicle.tesla.model3',
                            help='Vehicle blueprint name')
        
        parser.add_argument('--avg-throttle', type=float, default=4.0,
                            help='Average throttle command used for random target speed sampling')

        parser.add_argument('--min-throttle', type=float, default=3.5,
                            help='Minimum throttle command for random target speed sampling')

        parser.add_argument('--max-throttle', type=float, default=4.5,
                            help='Maximum throttle command for random target speed sampling')

        parser.add_argument('--throttle-std', type=float, default=1.0,
                            help='Standard deviation for throttle distribution')

        parser.add_argument('--delays', type=float, default=0.0,
                            help='Common start delay (seconds) for all vehicles')
        
        parser.add_argument('--poly-order', type=int, default=3,
                            help='Polynomial order for trajectory generation')

        parser.add_argument('--num-waypoints', type=int, default=12,
                            help='Number of waypoints per polynomial trajectory')

        parser.add_argument('--min-spawn-distance', type=float, default=10.0,
                            help='Minimum distance between spawned vehicles in meters for feasibility')

        parser.add_argument('--ego-role-name', type=str, default='agent_1',
                            help='CARLA ego vehicle role_name used for safe respawn checks')

        parser.add_argument('--min-respawn-distance-to-ego', type=float, default=20.0,
                            help='Minimum respawn distance from ego vehicle in meters')

        parser.add_argument('--spectator-pos', type=float, nargs=3, default=[-330.61,-225.46,65.16],
                            help='Spectator camera position [x, y, z] in ROS coordinates')
        
        parser.add_argument('--spectator-rot', type=float, nargs=3, default=[-59.43,35.72,0.00],
                            help='Spectator camera rotation [pitch, yaw, roll] in degrees')
        
        parser.add_argument('--seed', type=int, default=6,
                            help='Global random seed')

        args = parser.parse_args()

        if args.spectator_rot is None:
            args.spectator_rot = list(DEFAULT_SPECTATOR_ROT)

        if args.vehicles <= 0:
            logger.error("Number of vehicles must be positive")
            sys.exit(1)
            
        if args.min_throttle < 0.0:
            logger.error("Minimum throttle must be non-negative")
            sys.exit(1)

        if args.max_throttle < args.min_throttle:
            logger.error("Maximum throttle must be greater than or equal to minimum throttle")
            sys.exit(1)

        if args.throttle_std <= 0.0:
            logger.error("Throttle standard deviation must be positive")
            sys.exit(1)
            
        if args.poly_order < 1:
            logger.error("Polynomial order must be at least 1")
            sys.exit(1)

        if args.num_waypoints < 3:
            logger.error("Number of waypoints must be at least 3")
            sys.exit(1)

        if args.min_spawn_distance < 0.0:
            logger.error("Minimum spawn distance must be non-negative")
            sys.exit(1)

        if args.min_respawn_distance_to_ego < 0.0:
            logger.error("Minimum respawn distance to ego must be non-negative")
            sys.exit(1)

        if args.delays < 0.0:
            logger.error("Start delay must be non-negative")
            sys.exit(1)

        if args.spectator_pos is None:
            center_x = (args.area[0] + args.area[2]) / 2
            center_y = (args.area[1] + args.area[3]) / 2
            args.spectator_pos = [center_x, center_y, 50]

        spectator_config = {
            "position": args.spectator_pos,
            "rotation": args.spectator_rot
        }

        logger.info(f"Configuration:")
        logger.info(f"  Vehicles: {args.vehicles}")
        logger.info(f"  Area: {args.area}")
        logger.info(f"  Blueprint: {args.blueprint}")
        logger.info(
            f"  Throttle: avg={args.avg_throttle:.1f}, range={args.min_throttle:.1f}-{args.max_throttle:.1f}, std={args.throttle_std:.1f}"
        )
        logger.info(f"  Start delay: {args.delays}")
        logger.info("  Control mode: Throttle")
        logger.info(f"  Spectator position: {args.spectator_pos}")
        logger.info(f"  Spectator rotation: {args.spectator_rot}")
        logger.info(f"  Polynomial order: {args.poly_order}")
        logger.info(f"  Waypoints per trajectory: {args.num_waypoints}")
        logger.info(f"  Minimum spawn distance: {args.min_spawn_distance}m")
        logger.info(f"  Ego role name: {args.ego_role_name}")
        logger.info(f"  Minimum respawn distance to ego: {args.min_respawn_distance_to_ego}m")

        logger.info("Starting polynomial trajectory vehicle simulation...")
        spawn_random_vehicles(
            num_vehicles=args.vehicles,
            area_bounds=args.area,
            blueprint=args.blueprint,
            avg_throttle=args.avg_throttle,
            min_throttle=args.min_throttle,
            max_throttle=args.max_throttle,
            throttle_std=args.throttle_std,
            delays=args.delays,
            polynomial_order=args.poly_order,
            num_waypoints=args.num_waypoints,
            spectator_config=spectator_config,
            random_seed=args.seed,
            min_spawn_distance=args.min_spawn_distance,
            ego_role_name=args.ego_role_name,
            min_respawn_distance_to_ego=args.min_respawn_distance_to_ego
        )

    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception:
        logger.error("Unhandled runtime exception")
    finally:
        logger.info("Script finished")