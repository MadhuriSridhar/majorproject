"""
CARLA integration context for the Neuro-Symbolic Driver Attention
Monitoring System.

This module:
- Connects to the CARLA simulator.
- Finds or spawns a vehicle.
- Follows the vehicle with a smooth CARLA spectator camera.
- Moves the vehicle using direct vehicle control.
- Detects moving/stationary state.
- Tracks driving sessions.
- Reads vehicle speed.
- Converts vehicle speed into a contextual speed-risk value.
- Reads CARLA weather.
- Determines day/night state.
- Provides a single context dictionary for the main application.
"""

import math
import time
from typing import Any, Dict, Optional

import carla


class CarlaContext:
    """Handles vehicle and environment context from CARLA."""

    # ---------------------------------------------------------------
    # BASIC THRESHOLDS
    # ---------------------------------------------------------------

    MOVING_THRESHOLD_KMH = 1.0

    # Speed-risk thresholds.
    SPEED_LOW_MAX_KMH = 30.0
    SPEED_MODERATE_MAX_KMH = 60.0
    SPEED_HIGH_MAX_KMH = 90.0

    # ---------------------------------------------------------------
    # CAMERA SETTINGS
    # ---------------------------------------------------------------

    # Camera position relative to the vehicle.
    CAMERA_DISTANCE = 10.0
    CAMERA_HEIGHT = 5.0

    # Lower = smoother/slower camera movement.
    # Higher = camera catches up faster.
    CAMERA_SMOOTHING = 0.15

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
    ):
        self.host = host
        self.port = port

        # -----------------------------------------------------------
        # CARLA CONNECTION
        # -----------------------------------------------------------

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()

        # -----------------------------------------------------------
        # VEHICLE
        # -----------------------------------------------------------

        self.vehicle: Optional[carla.Vehicle] = None

        # -----------------------------------------------------------
        # DRIVING SESSION
        # -----------------------------------------------------------

        self.driving_session_active = False
        self.driving_session_id = 0
        self.session_start_time: Optional[float] = None

        # -----------------------------------------------------------
        # LATEST VEHICLE VALUES
        # -----------------------------------------------------------

        self.speed_kmh = 0.0
        self.vehicle_state = "NO_VEHICLE"

        # -----------------------------------------------------------
        # CAMERA STATE
        # -----------------------------------------------------------

        self.camera_following = False

        # -----------------------------------------------------------
        # MANUAL CONTROL STATE
        # -----------------------------------------------------------

        self.manual_control_active = False

        # Route-following state, built lazily from the configured spawn lane.
        self.route_waypoints = []
        self.route_index = 0
        self.route_complete = False
        self.current_throttle = 0.0
        self.current_steer = 0.0
        self.current_brake = 0.0
        self.last_control_time: Optional[float] = None
        self.last_control_status = "INITIALIZING"

        # Weather demo state; transitions are elapsed-time based and applied
        # only when the active scenario changes.
        self.weather_demo_started_at: Optional[float] = None
        self.weather_demo_index = -1
        self.weather_demo_complete = False

    # ===============================================================
    # CARLA CONNECTION
    # ===============================================================

    def is_connected(self) -> bool:
        """Check whether the CARLA server is reachable."""

        try:
            self.world.get_map()
            return True

        except Exception:
            return False

    def get_map_name(self) -> str:
        """Return the currently loaded CARLA map name."""

        try:
            return self.world.get_map().name

        except Exception:
            return "Unknown"

    # ===============================================================
    # VEHICLE
    # ===============================================================

    def find_vehicle(self) -> bool:
        """
        Find the first vehicle currently present in the CARLA world.

        Returns:
            True if a vehicle was found.
            False if no vehicle exists.
        """

        try:
            vehicles = self.world.get_actors().filter("vehicle.*")

            if not vehicles:
                self.vehicle = None
                return False

            self.vehicle = vehicles[0]

            return True

        except Exception:
            self.vehicle = None
            return False

    def spawn_vehicle(self) -> bool:
        """
        Find an existing vehicle or spawn a generic vehicle.

        If a vehicle already exists, it is reused.
        """

        # Reuse an existing vehicle.
        if self.find_vehicle():

            print(
                f"Using existing vehicle: "
                f"{self.vehicle.type_id}"
            )

            print(
                f"Vehicle ID: "
                f"{self.vehicle.id}"
            )

            return True

        try:

            blueprints = (
                self.world
                .get_blueprint_library()
                .filter("vehicle.*")
            )

            if not blueprints:
                print(
                    "ERROR: No vehicle blueprints found."
                )

                return False

            spawn_points = (
                self.world
                .get_map()
                .get_spawn_points()
            )

            if not spawn_points:
                print(
                    "ERROR: No spawn points available."
                )

                return False

            # Generic vehicle.
            vehicle_bp = blueprints[0]

            self.vehicle = (
                self.world.try_spawn_actor(
                    vehicle_bp,
                    spawn_points[0],
                )
            )

            if self.vehicle is None:

                print(
                    "ERROR: Could not spawn vehicle."
                )

                return False

            print(
                "Vehicle spawned successfully."
            )

            print(
                f"Vehicle: "
                f"{self.vehicle.type_id}"
            )

            print(
                f"Vehicle ID: "
                f"{self.vehicle.id}"
            )

            return True

        except Exception as exc:

            print(
                f"ERROR while spawning vehicle: "
                f"{exc}"
            )

            self.vehicle = None

            return False

    # ===============================================================
    # VEHICLE SPEED
    # ===============================================================

    def get_speed_kmh(self) -> float:
        """
        Calculate the current vehicle speed in km/h.

        CARLA provides velocity in metres per second.
        """

        if self.vehicle is None:
            return 0.0

        try:

            velocity = self.vehicle.get_velocity()

            speed_ms = math.sqrt(
                velocity.x ** 2
                + velocity.y ** 2
                + velocity.z ** 2
            )

            speed_kmh = speed_ms * 3.6

            return round(
                speed_kmh,
                2,
            )

        except Exception:
            return 0.0

    def get_speed_context(self) -> Dict[str, Any]:
        """
        Convert vehicle speed into a contextual speed signal.

        Returns:
            speed_kmh
            speed_risk
            speed_level
        """

        speed_kmh = self.get_speed_kmh()

        # -----------------------------------------------------------
        # NO VEHICLE
        # -----------------------------------------------------------

        if self.vehicle is None:

            return {
                "speed_kmh": 0.0,
                "speed_risk": 0.0,
                "speed_level": "NO_VEHICLE",
            }

        # -----------------------------------------------------------
        # SPEED RISK
        # -----------------------------------------------------------

        if speed_kmh < self.SPEED_LOW_MAX_KMH:

            speed_risk = 0.0
            speed_level = "LOW"

        elif speed_kmh < self.SPEED_MODERATE_MAX_KMH:

            speed_risk = 25.0
            speed_level = "MODERATE"

        elif speed_kmh < self.SPEED_HIGH_MAX_KMH:

            speed_risk = 50.0
            speed_level = "HIGH"

        else:

            speed_risk = 75.0
            speed_level = "VERY_HIGH"

        return {
            "speed_kmh": round(
                speed_kmh,
                2,
            ),
            "speed_risk": speed_risk,
            "speed_level": speed_level,
        }

    # ===============================================================
    # VEHICLE STATE / DRIVING SESSION
    # ===============================================================

    def update_vehicle_state(self) -> str:
        """
        Update vehicle movement state and driving-session state.

        Possible states:
            NO_VEHICLE
            MOVING
            STATIONARY
        """

        if self.vehicle is None:
            self.find_vehicle()

        # -----------------------------------------------------------
        # NO VEHICLE
        # -----------------------------------------------------------

        if self.vehicle is None:

            self.speed_kmh = 0.0
            self.vehicle_state = "NO_VEHICLE"

            if self.driving_session_active:

                self.driving_session_active = False
                self.session_start_time = None

            return self.vehicle_state

        # -----------------------------------------------------------
        # UPDATE SPEED
        # -----------------------------------------------------------

        self.speed_kmh = self.get_speed_kmh()

        # -----------------------------------------------------------
        # MOVING
        # -----------------------------------------------------------

        if self.speed_kmh >= self.MOVING_THRESHOLD_KMH:

            if not self.driving_session_active:

                self.driving_session_active = True

                self.driving_session_id += 1

                self.session_start_time = time.time()

                print(
                    f"\nNEW DRIVING SESSION: "
                    f"{self.driving_session_id}"
                )

            self.vehicle_state = "MOVING"

        # -----------------------------------------------------------
        # STATIONARY
        # -----------------------------------------------------------

        else:

            if self.driving_session_active:

                print(
                    f"\nDRIVING SESSION "
                    f"{self.driving_session_id} ENDED"
                )

            self.driving_session_active = False

            self.session_start_time = None

            self.vehicle_state = "STATIONARY"

        return self.vehicle_state

    def get_driving_duration_seconds(self) -> float:
        """Return the duration of the current driving session."""

        if (
            not self.driving_session_active
            or self.session_start_time is None
        ):

            return 0.0

        return max(
            0.0,
            time.time()
            - self.session_start_time,
        )

    # ===============================================================
    # WEATHER
    # ===============================================================

    def get_weather_context(self) -> Dict[str, Any]:
        """
        Read CARLA weather parameters and convert them into
        a contextual weather value.
        """

        try:

            weather = self.world.get_weather()

            cloudiness = float(
                weather.cloudiness
            )

            precipitation = float(
                weather.precipitation
            )

            precipitation_deposits = float(
                weather.precipitation_deposits
            )

            fog_density = float(
                weather.fog_density
            )

            wetness = float(
                weather.wetness
            )

            # -------------------------------------------------------
            # WEATHER CONDITION
            # -------------------------------------------------------

            if precipitation >= 60.0:

                condition = "HEAVY_RAIN"

            elif precipitation >= 20.0:

                condition = "RAIN"

            elif fog_density >= 50.0:

                condition = "FOG"

            elif cloudiness >= 70.0:

                condition = "CLOUDY"

            else:

                condition = "CLEAR"

            # -------------------------------------------------------
            # WEATHER RISK
            # -------------------------------------------------------

            risk = 0.0

            if precipitation >= 60.0:

                risk += 60.0

            elif precipitation >= 20.0:

                risk += 35.0

            if fog_density >= 50.0:

                risk += 30.0

            elif fog_density >= 25.0:

                risk += 15.0

            if wetness >= 60.0:

                risk += 10.0

            risk = min(
                100.0,
                risk,
            )

            return {
                "condition": condition,

                "risk": round(
                    risk,
                    2,
                ),

                "cloudiness": round(
                    cloudiness,
                    2,
                ),

                "precipitation": round(
                    precipitation,
                    2,
                ),

                "precipitation_deposits": round(
                    precipitation_deposits,
                    2,
                ),

                "fog_density": round(
                    fog_density,
                    2,
                ),

                "wetness": round(
                    wetness,
                    2,
                ),
            }

        except Exception as exc:

            return {
                "condition": "UNKNOWN",
                "risk": 0.0,
                "cloudiness": 0.0,
                "precipitation": 0.0,
                "precipitation_deposits": 0.0,
                "fog_density": 0.0,
                "wetness": 0.0,
                "error": str(exc),
            }

    def _weather_demo_parameters(self, index: int) -> Any:
        """Return the CARLA weather preset for one demo scenario."""

        presets = ("ClearNoon", "SoftRainNoon", "HardRainNoon", None, "ClearNoon")
        if index == 3:
            return carla.WeatherParameters(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                70.0,
                20.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        preset = getattr(carla.WeatherParameters, presets[index])
        return preset() if callable(preset) else preset

    def update_weather_demo(self, now_s: Optional[float] = None) -> bool:
        """Advance the five-stage, nonblocking 30-second CARLA weather demo."""

        if self.world is None:
            return False

        now = time.monotonic() if now_s is None else float(now_s)
        if self.weather_demo_started_at is None:
            self.weather_demo_started_at = now

        elapsed = max(0.0, now - self.weather_demo_started_at)
        index = min(4, int(elapsed // 30.0))
        if self.weather_demo_complete and index == self.weather_demo_index:
            return True
        if index == self.weather_demo_index:
            return True

        try:
            self.world.set_weather(self._weather_demo_parameters(index))
            self.weather_demo_index = index
            self.weather_demo_complete = index == 4 and elapsed >= 120.0
            labels = ("CLEAR", "RAIN", "HEAVY_RAIN", "FOG", "CLEAR")
            start = index * 30
            end = start + 30
            print(f"CARLA weather demo -> {labels[index]} ({start}-{end}s)")
            return True
        except Exception:
            return False

    def get_carla_location_context(self) -> Dict[str, Any]:
        """Return virtual CARLA geography classified from current x/y."""

        try:
            from location_context import carla_sim_location_context

            position = self.get_vehicle_position()
            context = carla_sim_location_context(
                position.get("x"),
                position.get("y"),
                position.get("z"),
            )
            return {
                "latitude": context.latitude,
                "longitude": context.longitude,
                "zone_type": context.zone_type,
                "zone_name": context.zone_name,
                "risk_level": context.risk_level,
                "source": context.source,
                "available": context.available,
                "mode": context.mode,
            }
        except Exception:
            return {
                "latitude": None,
                "longitude": None,
                "zone_type": "none",
                "zone_name": "Normal Road",
                "risk_level": 5.0,
                "source": "CARLA_SIM",
                "available": False,
                "mode": "carla_sim",
            }

    # ===============================================================
    # DAY / NIGHT
    # ===============================================================

    def get_time_context(self) -> Dict[str, Any]:
        """Determine whether CARLA is currently in day or night."""

        try:

            weather = self.world.get_weather()

            sun_altitude = float(
                weather.sun_altitude_angle
            )

            if sun_altitude < 0:

                period = "NIGHT"
                risk = 20.0

            else:

                period = "DAY"
                risk = 0.0

            return {
                "period": period,

                "sun_altitude_angle": round(
                    sun_altitude,
                    2,
                ),

                "risk": risk,
            }

        except Exception as exc:

            return {
                "period": "UNKNOWN",
                "sun_altitude_angle": 0.0,
                "risk": 0.0,
                "error": str(exc),
            }

    # ===============================================================
    # VEHICLE POSITION
    # ===============================================================

    def get_vehicle_position(
        self,
    ) -> Dict[str, float]:
        """Return the vehicle's CARLA world position."""

        if self.vehicle is None:

            return {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }

        try:

            location = (
                self.vehicle.get_location()
            )

            return {
                "x": round(
                    location.x,
                    3,
                ),

                "y": round(
                    location.y,
                    3,
                ),

                "z": round(
                    location.z,
                    3,
                ),
            }

        except Exception:

            return {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }

    # ===============================================================
    # SMOOTH CAMERA FOLLOW
    # ===============================================================

    def follow_vehicle(
        self,
        distance: float = CAMERA_DISTANCE,
        height: float = CAMERA_HEIGHT,
    ) -> bool:
        """
        Smoothly follow the vehicle using the CARLA spectator.

        The camera interpolates toward the desired position instead
        of instantly snapping to it.
        """

        if self.vehicle is None:

            if not self.find_vehicle():

                self.camera_following = False

                return False

        try:

            spectator = (
                self.world.get_spectator()
            )

            vehicle_transform = (
                self.vehicle.get_transform()
            )

            # -------------------------------------------------------
            # DESIRED CAMERA POSITION
            # -------------------------------------------------------

            desired_location = (
                vehicle_transform.transform(
                    carla.Location(
                        x=-distance,
                        z=height,
                    )
                )
            )

            # -------------------------------------------------------
            # CURRENT CAMERA POSITION
            # -------------------------------------------------------

            current_transform = (
                spectator.get_transform()
            )

            current_location = (
                current_transform.location
            )

            # -------------------------------------------------------
            # SMOOTH INTERPOLATION
            # -------------------------------------------------------

            smoothing = (
                self.CAMERA_SMOOTHING
            )

            new_location = carla.Location(

                x=(
                    current_location.x
                    + (
                        desired_location.x
                        - current_location.x
                    ) * smoothing
                ),

                y=(
                    current_location.y
                    + (
                        desired_location.y
                        - current_location.y
                    ) * smoothing
                ),

                z=(
                    current_location.z
                    + (
                        desired_location.z
                        - current_location.z
                    ) * smoothing
                ),
            )

            # Keep the camera oriented in the vehicle's direction.
            new_rotation = (
                vehicle_transform.rotation
            )

            spectator.set_transform(
                carla.Transform(
                    new_location,
                    new_rotation,
                )
            )

            self.camera_following = True

            return True

        except Exception:

            self.camera_following = False

            return False

    # ===============================================================
    # VEHICLE CONTROL
    # ===============================================================

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        return (first - second + 180.0) % 360.0 - 180.0

    def _choose_next_waypoint(self, waypoint: Any) -> Optional[Any]:
        """Choose the most forward, unvisited driving waypoint."""

        candidates = list(waypoint.next(2.0))
        if not candidates:
            return None

        heading = float(waypoint.transform.rotation.yaw)
        candidates.sort(
            key=lambda candidate: abs(
                self._angle_difference(
                    float(candidate.transform.rotation.yaw),
                    heading,
                )
            )
        )

        for candidate in candidates:
            location = candidate.transform.location
            if all(
                location.distance(existing.transform.location) > 1.0
                for existing in self.route_waypoints[-80:]
            ):
                return candidate

        return candidates[0]

    def _ensure_route(self) -> bool:
        """Build a forward route from the vehicle's current road lane."""

        if self.vehicle is None:
            return False

        if self.route_waypoints:
            return True

        try:
            current = self.world.get_map().get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if current is None:
                return False

            self.route_waypoints = [current]
            self.route_index = 0
            self.route_complete = False

            for _ in range(150):
                next_waypoint = self._choose_next_waypoint(
                    self.route_waypoints[-1]
                )
                if next_waypoint is None:
                    self.route_complete = True
                    break
                self.route_waypoints.append(next_waypoint)

            return True

        except Exception:
            return False

    def _advance_route(self) -> None:
        """Advance monotonically once the vehicle reaches its target."""

        if self.vehicle is None:
            return

        vehicle_location = self.vehicle.get_location()
        while self.route_index < len(self.route_waypoints) - 1:
            target = self.route_waypoints[self.route_index]
            if vehicle_location.distance(target.transform.location) > 3.0:
                break
            self.route_index += 1

        if self.route_complete or len(self.route_waypoints) - self.route_index > 20:
            return

        for _ in range(80):
            next_waypoint = self._choose_next_waypoint(
                self.route_waypoints[-1]
            )
            if next_waypoint is None:
                self.route_complete = True
                break
            self.route_waypoints.append(next_waypoint)

    def _route_geometry(self) -> tuple[float, str, Any]:
        """Return upcoming curvature, label, and distance-based target."""

        self._advance_route()
        current = self.route_waypoints[self.route_index]
        target_index = self.route_index
        lookahead = max(8.0, min(24.0, 8.0 + self.get_speed_kmh() * 0.25))
        travelled = 0.0

        while target_index < len(self.route_waypoints) - 1:
            next_waypoint = self.route_waypoints[target_index + 1]
            travelled += next_waypoint.transform.location.distance(
                self.route_waypoints[target_index].transform.location
            )
            target_index += 1
            if travelled >= lookahead:
                break

        curvature = 0.0
        if target_index > self.route_index:
            curvature = abs(
                self._angle_difference(
                    float(
                        self.route_waypoints[target_index].transform.rotation.yaw
                    ),
                    float(current.transform.rotation.yaw),
                )
            )

        if curvature >= 45.0:
            label = "SHARP_TURN"
        elif curvature >= 15.0:
            label = "TURN"
        else:
            label = "STRAIGHT"

        return curvature, label, self.route_waypoints[target_index]

    def _apply_route_control(self, dt: float) -> bool:
        """Compute and smoothly apply one route-following update."""

        if not self._ensure_route():
            return False

        if self.route_complete and self.route_index >= len(self.route_waypoints) - 1:
            target_speed = 0.0
            label = "ROUTE_END"
            target_steer = 0.0
        else:
            curvature, label, target = self._route_geometry()
            relative = self.vehicle.get_transform().inverse_transform(
                target.transform.location
            )
            target_steer = self._clamp(
                math.atan2(relative.y, max(0.1, relative.x)) / math.radians(45.0),
                -1.0,
                1.0,
            )

            target_speed = 42.0
            if curvature >= 45.0:
                target_speed = 16.0
            elif curvature >= 15.0:
                target_speed = 25.0

            weather_risk = self.get_weather_context().get("risk", 0.0)
            target_speed -= min(8.0, float(weather_risk) * 0.08)

        speed = self.get_speed_kmh()
        target_throttle = self._clamp((target_speed - speed) / 12.0, 0.0, 0.75)
        target_brake = 0.0
        if speed > target_speed + 2.0:
            target_brake = self._clamp((speed - target_speed) / 12.0, 0.0, 0.65)
            target_throttle = 0.0
        if target_speed == 0.0:
            target_brake = self._clamp(speed / 8.0, 0.0, 1.0)
            target_throttle = 0.0

        self.current_steer += self._clamp(
            target_steer - self.current_steer,
            -2.0 * dt,
            2.0 * dt,
        )
        self.current_throttle += self._clamp(
            target_throttle - self.current_throttle,
            -0.7 * dt,
            0.7 * dt,
        )
        self.current_brake += self._clamp(
            target_brake - self.current_brake,
            -1.5 * dt,
            1.5 * dt,
        )

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=self._clamp(self.current_throttle, 0.0, 1.0),
                steer=self._clamp(self.current_steer, -1.0, 1.0),
                brake=self._clamp(self.current_brake, 0.0, 1.0),
            )
        )
        self.last_control_status = label
        return True

    def start_moving(
        self,
        throttle: float = 0.4,
        steer: float = 0.0,
    ) -> bool:
        """Start moving the vehicle."""

        if self.vehicle is None:

            if not self.find_vehicle():

                return False

        try:

            self.vehicle.set_autopilot(
                False
            )

            now = time.monotonic()
            if self.last_control_time is None:
                self.last_control_time = now
                self.vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=0.0,
                    )
                )
            else:
                dt = self._clamp(now - self.last_control_time, 0.02, 0.2)
                self.last_control_time = now
                self._apply_route_control(dt)

            self.manual_control_active = True

            return True

        except Exception:

            return False

    def stop_vehicle(self) -> bool:
        """Stop the vehicle using the brake."""

        if self.vehicle is None:

            return False

        try:

            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                )
            )

            self.manual_control_active = False
            self.current_throttle = 0.0
            self.current_steer = 0.0
            self.current_brake = 1.0
            self.last_control_time = None

            return True

        except Exception:

            return False

    # ===============================================================
    # COMPLETE CARLA CONTEXT
    # ===============================================================

    def get_context(self) -> Dict[str, Any]:
        """
        Return the complete current CARLA context.

        Context includes:
        - vehicle presence
        - vehicle speed
        - speed risk
        - movement state
        - driving session
        - driving duration
        - vehicle position
        - weather
        - day/night
        """

        vehicle_state = (
            self.update_vehicle_state()
        )

        speed_context = (
            self.get_speed_context()
        )

        self.update_weather_demo()

        weather = (
            self.get_weather_context()
        )

        time_context = (
            self.get_time_context()
        )

        position = (
            self.get_vehicle_position()
        )

        location = (
            self.get_carla_location_context()
        )

        return {

            # -------------------------------------------------------
            # CARLA
            # -------------------------------------------------------

            "carla_connected": (
                self.is_connected()
            ),

            "map": (
                self.get_map_name()
            ),

            # -------------------------------------------------------
            # SIMULATED LOCATION
            # -------------------------------------------------------

            "location": location,

            # -------------------------------------------------------
            # VEHICLE
            # -------------------------------------------------------

            "vehicle_present": (
                self.vehicle is not None
            ),

            "vehicle_type": (
                self.vehicle.type_id
                if self.vehicle is not None
                else None
            ),

            "vehicle_id": (
                self.vehicle.id
                if self.vehicle is not None
                else None
            ),

            # -------------------------------------------------------
            # SPEED CONTEXT
            # -------------------------------------------------------

            "speed_kmh": (
                speed_context["speed_kmh"]
            ),

            "speed_risk": (
                speed_context["speed_risk"]
            ),

            "speed_level": (
                speed_context["speed_level"]
            ),

            # -------------------------------------------------------
            # VEHICLE STATE
            # -------------------------------------------------------

            "vehicle_state": (
                vehicle_state
            ),

            # -------------------------------------------------------
            # DRIVING SESSION
            # -------------------------------------------------------

            "driving_session_active": (
                self.driving_session_active
            ),

            "driving_session_id": (
                self.driving_session_id
            ),

            "driving_duration_seconds": round(
                self.get_driving_duration_seconds(),
                2,
            ),

            # -------------------------------------------------------
            # POSITION
            # -------------------------------------------------------

            "position": position,

            # -------------------------------------------------------
            # ENVIRONMENT
            # -------------------------------------------------------

            "weather": weather,

            "time": time_context,

            # -------------------------------------------------------
            # CAMERA
            # -------------------------------------------------------

            "camera_following": (
                self.camera_following
            ),
        }

    # ===============================================================
    # CLEANUP
    # ===============================================================

    def cleanup(self) -> None:
        """Safely stop the vehicle."""

        if self.vehicle is None:

            return

        try:

            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                )
            )

        except Exception:
            pass


# ====================================================================
# STANDALONE TEST
# ====================================================================

def main() -> None:
    """
    Standalone CARLA context test.

    The test:
    - Connects to CARLA.
    - Finds/spawns a vehicle.
    - Starts the vehicle.
    - Smoothly follows the vehicle.
    - Displays speed and speed risk.
    - Displays vehicle state.
    - Displays driving session.
    - Displays weather.
    - Displays day/night.
    - Stops safely with Ctrl+C.
    """

    print("=" * 70)
    print("CARLA CONTEXT TEST")
    print("=" * 70)

    context = CarlaContext()

    # ---------------------------------------------------------------
    # CONNECTION
    # ---------------------------------------------------------------

    if not context.is_connected():

        print(
            "ERROR: Could not connect to CARLA."
        )

        return

    print(
        "CARLA connection successful."
    )

    print(
        "Map:",
        context.get_map_name(),
    )

    # ---------------------------------------------------------------
    # VEHICLE
    # ---------------------------------------------------------------

    print(
        "\nFinding/spawning vehicle..."
    )

    if not context.spawn_vehicle():

        print(
            "ERROR: Vehicle could not be created."
        )

        return

    print(
        "\nVehicle ready."
    )

    print(
        "Smooth spectator-camera follow enabled."
    )

    print(
        "Starting vehicle movement."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    # ---------------------------------------------------------------
    # START
    # ---------------------------------------------------------------

    context.follow_vehicle()

    context.start_moving(
        throttle=0.4
    )

    try:

        while True:

            # Smooth camera update.
            context.follow_vehicle()

            # Tick the existing route controller.
            context.start_moving()

            # Get complete context.
            data = context.get_context()

            weather = data["weather"]

            time_data = data["time"]

            print(
                f"Speed: "
                f"{data['speed_kmh']:6.2f} km/h | "

                f"Speed Risk: "
                f"{data['speed_risk']:5.1f} | "

                f"Speed Level: "
                f"{data['speed_level']:10} | "

                f"State: "
                f"{data['vehicle_state']:10} | "

                f"Session: "
                f"{data['driving_session_id']} | "

                f"Weather: "
                f"{weather['condition']:12} | "

                f"Time: "
                f"{time_data['period']}"
            )

            # Approximately 20 updates per second.
            time.sleep(0.05)

    except KeyboardInterrupt:

        print(
            "\n\nStopping CARLA test..."
        )

    finally:

        context.stop_vehicle()

        print(
            "Vehicle stopped."
        )

        print(
            "CARLA context test finished."
        )


if __name__ == "__main__":
    main()