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

import argparse
import glob
import importlib
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import carla

DEFAULT_TARGET_SPEED_KMH = 35.0

_CARLA_NAV_CACHE: Optional[
    Tuple[Any, Any, Any]
] = None


def _normalize_carla_pythonapi_root(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def _carla_pythonapi_search_paths() -> list[str]:
    """Return candidate CARLA PythonAPI/carla roots, newest preference first."""

    candidates: list[str] = []

    def add(path: Optional[str]) -> None:
        if not path:
            return
        normalized = _normalize_carla_pythonapi_root(path)
        if normalized not in candidates:
            candidates.append(normalized)

    # Standard install layout used by this project (C:\CARLA\PythonAPI\carla).
    add(os.path.join(os.path.expandvars(r"%SystemDrive%"), "CARLA", "PythonAPI", "carla"))
    add(r"C:\CARLA\PythonAPI\carla")

    carla_root = os.environ.get("CARLA_ROOT")
    if carla_root:
        add(os.path.join(carla_root, "PythonAPI", "carla"))

    try:
        agents_mod = importlib.import_module("agents")
        agents_file = getattr(agents_mod, "__file__", "") or ""
        if agents_file:
            add(os.path.join(os.path.dirname(agents_file), ".."))
    except ImportError:
        pass

    # Trust the interpreter's own environment: when "import agents" already
    # works, the root that provides it is on sys.path (PYTHONPATH, a .pth file
    # or the current directory), so such a root must be a candidate too, even
    # if it resolves through a package that is not the official agent package.
    for entry in list(sys.path):
        if not entry:
            continue
        if os.path.isfile(
            os.path.join(entry, "agents", "navigation", "basic_agent.py")
        ):
            add(entry)

    try:
        carla_file = getattr(carla, "__file__", "") or ""
        if carla_file:
            add(os.path.join(os.path.dirname(carla_file), "..", "..", "PythonAPI", "carla"))
            add(os.path.join(os.path.dirname(carla_file), "..", "PythonAPI", "carla"))
    except Exception:
        pass

    for pattern in (
        os.path.expanduser("~/CARLA*/PythonAPI/carla"),
        os.path.join(os.path.expandvars(r"%SystemDrive%"), "CARLA*", "PythonAPI", "carla"),
        "C:/CARLA/PythonAPI/carla",
        "C:/CARLA*/PythonAPI/carla",
        "C:/Program Files/CARLA*/PythonAPI/carla",
        "D:/CARLA/PythonAPI/carla",
        "D:/CARLA*/PythonAPI/carla",
    ):
        for match in glob.glob(pattern):
            add(match)

    return candidates


def _load_carla_navigation_modules() -> Tuple[Any, Any, Any]:
    grp_mod = importlib.import_module("agents.navigation.global_route_planner")
    ba_mod = importlib.import_module("agents.navigation.basic_agent")
    lp_mod = importlib.import_module("agents.navigation.local_planner")
    return (
        grp_mod.GlobalRoutePlanner,
        ba_mod.BasicAgent,
        lp_mod.RoadOption,
    )


def _import_carla_navigation() -> Tuple[Any, Any, Any]:
    """Import official CARLA navigation agents from the local PythonAPI."""

    global _CARLA_NAV_CACHE
    if _CARLA_NAV_CACHE is not None:
        return _CARLA_NAV_CACHE

    try:
        _CARLA_NAV_CACHE = _load_carla_navigation_modules()
        return _CARLA_NAV_CACHE
    except ImportError as initial_error:
        last_error: Optional[BaseException] = initial_error
        found_agent_roots: list[str] = []

        for root in _carla_pythonapi_search_paths():
            agents_dir = os.path.join(root, "agents")
            if not os.path.isdir(agents_dir):
                continue

            found_agent_roots.append(root)
            if root not in sys.path:
                sys.path.insert(0, root)

            for module_name in (
                "agents",
                "agents.navigation",
                "agents.navigation.global_route_planner",
                "agents.navigation.basic_agent",
                "agents.navigation.local_planner",
            ):
                sys.modules.pop(module_name, None)

            try:
                _CARLA_NAV_CACHE = _load_carla_navigation_modules()
                return _CARLA_NAV_CACHE
            except ImportError as exc:
                last_error = exc
                continue

        if found_agent_roots:
            raise ImportError(
                "CARLA PythonAPI agents were found at "
                f"{found_agent_roots[0]}, but navigation modules failed to "
                f"import: {last_error}. Ensure CARLA agent dependencies are "
                "installed in this environment (networkx, Shapely)."
            ) from last_error

        raise ImportError(
            "CARLA PythonAPI agents were not found. Expected a CARLA install "
            "at C:\\CARLA\\PythonAPI\\carla or set CARLA_ROOT."
        ) from last_error


def _maneuver_from_option(option: Any, road_option_cls: Any) -> str:
    if option == road_option_cls.LEFT:
        return "LEFT"
    if option == road_option_cls.RIGHT:
        return "RIGHT"
    if option == road_option_cls.STRAIGHT:
        return "STRAIGHT"
    return "STRAIGHT"


def _route_length_m(route: list[tuple[Any, Any]]) -> float:
    if len(route) < 2:
        return 0.0

    total = 0.0
    previous = route[0][0].transform.location
    for waypoint, _ in route[1:]:
        current = waypoint.transform.location
        total += previous.distance(current)
        previous = current
    return total


def _detect_route_loop(
    route: list[tuple[Any, Any]],
    start_location: carla.Location,
    end_location: carla.Location,
) -> bool:
    # A drive is a loop when it ends where it started. Trips that are too short
    # are already rejected by the destination distance tiers, not by this check.
    if end_location.distance(start_location) < 150.0:
        return True

    # Re-entering a lane segment the route has already left means the route is
    # circling back. Consecutive waypoints share the same segment while simply
    # driving along a lane, so segments must be compared between runs (i.e.
    # after a segment has been left), never between sample indices.
    left_segments: set[tuple[int, int, int]] = set()
    previous_key: Optional[tuple[int, int, int]] = None
    for waypoint, _ in route:
        key = (
            int(getattr(waypoint, "road_id", 0)),
            int(getattr(waypoint, "section_id", 0)),
            int(getattr(waypoint, "lane_id", 0)),
        )
        if key == previous_key:
            continue
        if key in left_segments:
            return True
        if previous_key is not None:
            left_segments.add(previous_key)
        previous_key = key

    return False


def _analyze_global_route(
    route: list[tuple[Any, Any]],
    start_location: carla.Location,
    end_location: carla.Location,
    road_option_cls: Any,
) -> Dict[str, Any]:
    left = 0
    right = 0
    straight = 0
    uturn_count = 0
    previous_yaw: Optional[float] = None

    for _, option in route:
        if option == road_option_cls.LEFT:
            left += 1
        elif option == road_option_cls.RIGHT:
            right += 1
        elif option == road_option_cls.STRAIGHT:
            straight += 1

    for waypoint, _ in route:
        yaw = float(getattr(waypoint.transform.rotation, "yaw", 0.0))
        if previous_yaw is not None:
            delta = abs((yaw - previous_yaw + 180.0) % 360.0 - 180.0)
            if delta > 150.0:
                uturn_count += 1
        previous_yaw = yaw

    loop_detected = _detect_route_loop(route, start_location, end_location)
    length_m = _route_length_m(route)

    return {
        "length_m": length_m,
        "left": left,
        "right": right,
        "straight": straight,
        "uturn_count": uturn_count,
        "junction_count": left + right,
        "end_dist_from_start_m": end_location.distance(start_location),
        "loop_detected": loop_detected,
    }


def _route_passes_tier(analysis: Dict[str, Any], tier: Dict[str, Any]) -> bool:
    if analysis.get("loop_detected"):
        return False
    if analysis.get("uturn_count", 0) > 0:
        return False
    if analysis.get("length_m", 0.0) < tier["min_len"]:
        return False
    if analysis.get("length_m", 0.0) > tier["max_len"]:
        return False
    if analysis.get("end_dist_from_start_m", 0.0) < tier["min_end"]:
        return False
    if analysis.get("junction_count", 0) < tier["min_junctions"]:
        return False
    if tier["need_lr"] and analysis.get("left", 0) + analysis.get("right", 0) < 1:
        return False
    return True


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
        # Visited-waypoint fingerprints used only as a tie-breaker at junctions.
        # (road_id, section_id, lane_id, s) — never used to force unvisited detours.
        self._visited_waypoints: set[tuple[int, int, int, float]] = set()
        # Active junction commitment: hold the chosen branch until it is cleared.
        self._junction_commit_fp: Optional[tuple[int, int, int, float]] = None
        self._junction_commit_heading: Optional[float] = None

        # Official CARLA route navigation (--route mode).
        self.route_active = False
        self.route_completed = False
        self.route_follower = None
        self.route_agent: Any = None
        self.route_plan: list[tuple[Any, Any]] = []
        self.route_stats: Dict[str, Any] = {}
        self._global_route_planner: Any = None
        self._route_destination_location: Optional[carla.Location] = None
        self._route_start_location: Optional[carla.Location] = None
        self._route_total_length_m = 0.0
        self._navigation_method = ""
        self._route_mode = False
        self._last_progress_print_s = 0.0
        self._last_route_destination: Optional[carla.Location] = None

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

        presets = (
            "ClearNoon",
            "SoftRainNoon",
            "HardRainNight",
            "FoggyNight",
            "ClearNight",
        )
        if index < 0:
            index = 0
        preset_name = presets[index % len(presets)]
        preset = getattr(carla.WeatherParameters, preset_name, None)
        if preset is None:
            return carla.WeatherParameters()
        return preset() if callable(preset) else preset

    def update_weather_demo(self, now_s: Optional[float] = None) -> bool:
        """Advance the continuous 30-second weather demo with day/night cycling."""

        if self.world is None:
            return False

        now = time.monotonic() if now_s is None else float(now_s)
        if self.weather_demo_started_at is None:
            self.weather_demo_started_at = now

        elapsed = max(0.0, now - self.weather_demo_started_at)
        index = int(elapsed // 30.0) % 5
        if index == self.weather_demo_index:
            return True

        try:
            self.world.set_weather(self._weather_demo_parameters(index))
            self.weather_demo_index = index
            self.weather_demo_complete = False
            labels = ("CLEAR DAY", "RAIN DAY", "HEAVY RAIN NIGHT", "FOG NIGHT", "CLEAR NIGHT")
            start = (index * 30)
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

    @staticmethod
    def _waypoint_fingerprint(waypoint: Any) -> tuple[int, int, int, float]:
        """Return a stable (road_id, section_id, lane_id, rounded_s) fingerprint
        so visited-waypoint loop prevention can work across route rebuilds."""

        return (
            int(getattr(waypoint, "road_id", 0)),
            int(getattr(waypoint, "section_id", 0)),
            int(getattr(waypoint, "lane_id", 0)),
            float(round(float(getattr(waypoint, "s", 0.0)), 1)),
        )

    def _forward_driving_candidates(
        self,
        waypoint: Any,
        lookahead: float = 8.0,
    ) -> list[tuple[Any, float]]:
        """Return valid forward driving candidates from CARLA lane topology."""

        if waypoint is None:
            return []

        heading = float(getattr(waypoint.transform.rotation, "yaw", 0.0))
        try:
            raw_candidates = list(waypoint.next(lookahead))
        except Exception:
            raw_candidates = []

        valid: list[tuple[Any, float]] = []
        for candidate in raw_candidates:
            if candidate is None:
                continue

            lane_type = getattr(candidate, "lane_type", None)
            if lane_type is not None and lane_type != carla.LaneType.Driving:
                continue

            angle_delta = abs(
                self._angle_difference(
                    float(getattr(candidate.transform.rotation, "yaw", heading)),
                    heading,
                )
            )
            # Reject U-turn / reverse-direction links.
            if angle_delta > 120.0:
                continue

            valid.append((candidate, angle_delta))

        return valid

    def _is_junction(self, candidates: list[tuple[Any, float]]) -> bool:
        """True when the vehicle must choose among distinct forward branches."""

        if len(candidates) <= 1:
            return False

        angles = [angle for _, angle in candidates]
        if max(angles) < 15.0:
            return False

        road_ids = {
            self._waypoint_fingerprint(candidate)[0]
            for candidate, _ in candidates
        }
        if len(road_ids) > 1:
            return True

        turning = sum(1 for angle in angles if angle >= 15.0)
        return turning >= 2 or (turning >= 1 and len(candidates) >= 2)

    def _select_target_waypoint(
        self,
        waypoint: Any,
        candidates: Optional[list[tuple[Any, float]]] = None,
    ) -> Optional[Any]:
        """Pick the most natural forward driving branch from CARLA topology.

        Prefer a branch that leaves the current road and avoids previously visited
        lane segments. Within the same road, keep the least-angle continuation as
        the natural default.
        """

        if waypoint is None:
            return None

        if candidates is None:
            candidates = self._forward_driving_candidates(waypoint)

        if not candidates:
            return None

        current_road_id = int(getattr(waypoint, "road_id", 0))
        scored: list[tuple[tuple[int, int, float], Any]] = []
        for candidate, angle_delta in candidates:
            candidate_road_id = int(getattr(candidate, "road_id", 0))
            was_visited = (
                self._waypoint_fingerprint(candidate) in self._visited_waypoints
            )
            road_bias = 0 if candidate_road_id != current_road_id else 1
            scored.append(
                (
                    (road_bias, 1 if was_visited else 0, angle_delta),
                    candidate,
                )
            )

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _clear_junction_commit_if_done(
        self,
        current: Any,
        candidates: list[tuple[Any, float]],
    ) -> None:
        """Drop junction commitment once the vehicle is aligned on the branch."""

        if self._junction_commit_fp is None:
            return

        current_fp = self._waypoint_fingerprint(current)
        if current_fp[:3] != self._junction_commit_fp[:3]:
            return

        heading = float(getattr(current.transform.rotation, "yaw", 0.0))
        commit_heading = self._junction_commit_heading
        if commit_heading is None:
            self._junction_commit_fp = None
            return

        if abs(self._angle_difference(heading, commit_heading)) >= 20.0:
            return

        # Require a single natural forward continuation before clearing.
        low_angle = [angle for _, angle in candidates if angle < 15.0]
        if len(candidates) <= 1 or low_angle:
            self._junction_commit_fp = None
            self._junction_commit_heading = None

    def _target_matching_commit(
        self,
        candidates: list[tuple[Any, float]],
    ) -> Optional[Any]:
        """Return the candidate that honors the active junction commitment."""

        commit_fp = self._junction_commit_fp
        commit_heading = self._junction_commit_heading
        if commit_fp is None:
            return None

        for candidate, _ in candidates:
            if self._waypoint_fingerprint(candidate) == commit_fp:
                return candidate

        branch = [
            candidate
            for candidate, _ in candidates
            if self._waypoint_fingerprint(candidate)[:3] == commit_fp[:3]
        ]
        if branch and commit_heading is not None:
            return min(
                branch,
                key=lambda candidate: abs(
                    self._angle_difference(
                        float(
                            getattr(candidate.transform.rotation, "yaw", 0.0)
                        ),
                        commit_heading,
                    )
                ),
            )

        if commit_heading is not None and candidates:
            return min(
                (candidate for candidate, _ in candidates),
                key=lambda candidate: abs(
                    self._angle_difference(
                        float(
                            getattr(candidate.transform.rotation, "yaw", 0.0)
                        ),
                        commit_heading,
                    )
                ),
            )

        return None

    def _resolve_route_target(self, current: Any) -> Any:
        """Resolve the steering target, honoring any active junction commitment."""

        candidates = self._forward_driving_candidates(current)
        if not candidates:
            self._mark_visited(current)
            return current

        self._clear_junction_commit_if_done(current, candidates)

        if self._junction_commit_fp is not None:
            committed = self._target_matching_commit(candidates)
            if committed is not None:
                self._mark_visited(current)
                return committed

        target = self._select_target_waypoint(current, candidates)
        if target is None:
            target = candidates[0][0]

        if self._is_junction(candidates):
            self._junction_commit_fp = self._waypoint_fingerprint(target)
            self._junction_commit_heading = float(
                getattr(target.transform.rotation, "yaw", 0.0)
            )

        self._mark_visited(current)
        return target

    def _choose_next_waypoint(self, waypoint: Any) -> Optional[Any]:
        """Backward-compatible wrapper for the road-topology target selector."""
        return self._select_target_waypoint(waypoint)

    def _mark_visited(self, waypoint: Any) -> None:
        """Record a waypoint fingerprint so loop prevention can see it."""
        self._visited_waypoints.add(self._waypoint_fingerprint(waypoint))

    def _ensure_route(self) -> bool:
        """Seed the route from the vehicle's current drivable waypoint."""

        if self.vehicle is None:
            return False

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
            self._mark_visited(current)
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
            self._mark_visited(next_waypoint)

    def _route_geometry(self) -> tuple[float, str, Any]:
        """Return upcoming curvature, label, and a live road-following target."""

        current = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if current is None:
            return 0.0, "STRAIGHT", self.route_waypoints[0] if self.route_waypoints else None

        target = self._select_target_waypoint(current)
        if target is None:
            target = current

        self._mark_visited(current)
        self._mark_visited(target)

        curvature = abs(
            self._angle_difference(
                float(getattr(target.transform.rotation, "yaw", 0.0)),
                float(getattr(current.transform.rotation, "yaw", 0.0)),
            )
        )

        if curvature >= 45.0:
            label = "SHARP_TURN"
        elif curvature >= 15.0:
            label = "TURN"
        else:
            label = "STRAIGHT"

        self.route_waypoints = [current, target]
        self.route_index = 0
        return curvature, label, target

    def _apply_route_control(self, dt: float) -> bool:
        """Follow the live CARLA road topology rather than a fixed route list."""

        if not self._ensure_route():
            return False

        current = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if current is None:
            return False

        target = self._resolve_route_target(current)

        curvature = abs(
            self._angle_difference(
                float(getattr(target.transform.rotation, "yaw", 0.0)),
                float(getattr(current.transform.rotation, "yaw", 0.0)),
            )
        )

        if curvature >= 45.0:
            label = "SHARP_TURN"
            target_speed = 16.0
        elif curvature >= 15.0:
            label = "TURN"
            target_speed = 25.0
        else:
            label = "STRAIGHT"
            target_speed = 42.0

        weather_risk = self.get_weather_context().get("risk", 0.0)
        target_speed -= min(8.0, float(weather_risk) * 0.08)

        relative = self.vehicle.get_transform().inverse_transform(target.transform.location)
        target_steer = self._clamp(
            math.atan2(relative.y, max(0.1, relative.x)) / math.radians(45.0),
            -1.0,
            1.0,
        )

        speed = self.get_speed_kmh()
        target_throttle = self._clamp((target_speed - speed) / 12.0, 0.0, 0.75)
        target_brake = 0.0
        if speed > target_speed + 2.0:
            target_brake = self._clamp((speed - target_speed) / 12.0, 0.0, 0.65)
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

        if self._route_mode or self.route_active:
            return True

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
            self._stop_route_navigation()

            return True

        except Exception:

            return False

    # ===============================================================
    # OFFICIAL CARLA ROUTE NAVIGATION (--route)
    # ===============================================================

    def _ensure_global_route_planner(self) -> Any:
        GlobalRoutePlanner, _, _ = _import_carla_navigation()
        if self._global_route_planner is None:
            self._global_route_planner = GlobalRoutePlanner(
                self.world.get_map(),
                sampling_resolution=2.0,
            )
        return self._global_route_planner

    def _route_plan_from_navigation(
        self,
        start_location: carla.Location,
        destination_location: carla.Location,
    ) -> Optional[Dict[str, Any]]:
        """Trace and validate one official CARLA global route."""

        planner = self._ensure_global_route_planner()
        route = planner.trace_route(start_location, destination_location)
        if not route:
            return None

        _, _, RoadOption = _import_carla_navigation()
        analysis = _analyze_global_route(
            route,
            start_location,
            destination_location,
            RoadOption,
        )
        loop_detected = bool(analysis["loop_detected"])
        return {
            "route": route,
            "destination": destination_location,
            "analysis": analysis,
            "complete": (not loop_detected) and len(route) > 1,
            "loop_prevented": not loop_detected,
        }

    def _select_official_route(
        self,
    ) -> Optional[Tuple[list[tuple[Any, Any]], carla.Location, Dict[str, Any]]]:
        """Pick a validated finite route to a dynamic spawn-point destination."""

        if self.vehicle is None:
            return None

        start_location = self.vehicle.get_location()
        spawn_points = list(self.world.get_map().get_spawn_points())
        if not spawn_points:
            return None

        random.shuffle(spawn_points)
        tiers = (
            {
                "min_len": 500.0,
                "max_len": 2500.0,
                "min_junctions": 2,
                "need_lr": True,
                "min_end": 250.0,
            },
            {
                "min_len": 500.0,
                "max_len": 2500.0,
                "min_junctions": 1,
                "need_lr": True,
                "min_end": 250.0,
            },
            {
                "min_len": 400.0,
                "max_len": 2800.0,
                "min_junctions": 1,
                "need_lr": False,
                "min_end": 200.0,
            },
            {
                "min_len": 250.0,
                "max_len": 3500.0,
                "min_junctions": 0,
                "need_lr": False,
                "min_end": 150.0,
            },
        )

        best: Optional[
            Tuple[
                Tuple[int, int, float],
                list[tuple[Any, Any]],
                carla.Location,
                Dict[str, Any],
            ]
        ] = None

        for tier in tiers:
            for destination_transform in spawn_points[:48]:
                destination_location = destination_transform.location
                if destination_location.distance(start_location) < tier["min_end"]:
                    continue

                plan = self._route_plan_from_navigation(
                    start_location,
                    destination_location,
                )
                if plan is None or not plan.get("complete"):
                    continue

                analysis = plan["analysis"]
                if not _route_passes_tier(analysis, tier):
                    continue

                score = (
                    int(analysis.get("junction_count", 0)),
                    int(analysis.get("left", 0) + analysis.get("right", 0)),
                    -abs(float(analysis.get("length_m", 0.0)) - 1200.0),
                )
                if best is None or score > best[0]:
                    best = (score, plan["route"], destination_location, analysis)

            if best is not None:
                break

        if best is None:
            return None

        _, route, destination_location, analysis = best
        return route, destination_location, analysis

    def plan_demo_route(self) -> Optional[Dict[str, Any]]:
        """Plan a finite official CARLA route (replaces custom demo routing)."""

        selection = self._select_official_route()
        if selection is None:
            return None

        route, destination_location, analysis = selection
        return {
            "route": route,
            "destination": destination_location,
            "analysis": analysis,
            "complete": True,
            "loop_prevented": not analysis.get("loop_detected", True),
        }

    def _print_route_startup_diagnostics(
        self,
        start_location: carla.Location,
        destination_location: carla.Location,
        analysis: Dict[str, Any],
    ) -> None:
        print(f"Navigation method: {self._navigation_method}")
        print(
            f"Start: x={start_location.x:.1f} y={start_location.y:.1f}"
        )
        print(
            "Destination: "
            f"x={destination_location.x:.1f} y={destination_location.y:.1f}"
        )
        print(f"Route length: {analysis.get('length_m', 0.0):.1f} m")
        print(f"Junctions: {analysis.get('junction_count', 0)}")
        print(f"LEFT decisions: {analysis.get('left', 0)}")
        print(f"RIGHT decisions: {analysis.get('right', 0)}")
        print(f"U-TURNS: {analysis.get('uturn_count', 0)}")
        print(f"Loop detected: {analysis.get('loop_detected', True)}")

    def _peek_next_maneuver(self) -> str:
        if self.route_agent is None:
            return "STRAIGHT"

        _, _, RoadOption = _import_carla_navigation()
        local_planner = getattr(self.route_agent, "_local_planner", None)
        queue = getattr(local_planner, "_waypoints_queue", None)
        if not queue:
            return "STRAIGHT"

        for _, option in list(queue)[:8]:
            label = _maneuver_from_option(option, RoadOption)
            if label in {"LEFT", "RIGHT"}:
                return label
        return "STRAIGHT"

    def _estimate_route_progress(self) -> Tuple[float, float]:
        if (
            self.vehicle is None
            or self._route_destination_location is None
            or self._route_total_length_m <= 0.0
        ):
            return 0.0, 0.0

        remaining = self.vehicle.get_location().distance(
            self._route_destination_location
        )
        progress = 100.0 * max(
            0.0,
            min(
                1.0,
                1.0 - (remaining / max(self._route_total_length_m, 1.0)),
            ),
        )
        return progress, remaining

    def _print_route_progress_diagnostics(self) -> None:
        progress, remaining = self._estimate_route_progress()
        print(
            f"Speed: {self.get_speed_kmh():.1f} km/h | "
            f"Route progress: {progress:.1f}% | "
            f"Distance remaining: {remaining:.1f}m | "
            f"Next maneuver: {self._peek_next_maneuver()} | "
            f"Control source: {self._navigation_method}"
        )

    def start_route_following(self) -> bool:
        """Start official CARLA global-route navigation to a destination."""

        if self.vehicle is None and not self.find_vehicle():
            return False

        try:
            self.vehicle.set_autopilot(False)
        except Exception:
            return False

        try:
            _, BasicAgent, _ = _import_carla_navigation()
        except ImportError as exc:
            print(f"ERROR: {exc}")
            return False

        plan = self.plan_demo_route()
        if plan is None:
            print(
                "ERROR: Could not find a validated non-looping CARLA route."
            )
            return False

        route = plan["route"]
        destination_location = plan["destination"]
        analysis = plan["analysis"]
        start_location = self.vehicle.get_location()

        self._navigation_method = (
            "CARLA GlobalRoutePlanner + BasicAgent"
        )
        self.route_plan = route
        self.route_stats = analysis
        self._route_start_location = start_location
        self._route_destination_location = destination_location
        self._route_total_length_m = float(analysis.get("length_m", 0.0))

        self.route_agent = BasicAgent(
            self.vehicle,
            target_speed=DEFAULT_TARGET_SPEED_KMH,
            opt_dict={
                "ignore_traffic_lights": True,
                "ignore_signs": True,
            },
        )
        self.route_agent.set_global_plan(route)
        self.route_follower = None
        self.route_active = True
        self.route_completed = False
        self._route_mode = True
        self.manual_control_active = False
        self.last_control_time = None
        self._last_progress_print_s = time.monotonic()

        self._print_route_startup_diagnostics(
            start_location,
            destination_location,
            analysis,
        )
        return True

    def _tick_route_follower(self) -> bool:
        """Advance official route navigation by one control step.

        Reaching a destination does not end the demo loop; instead, it chooses a
        new validated route and keeps driving until Ctrl+C.
        """

        if not self.route_active or self.route_agent is None:
            return False

        try:
            if self.route_agent.done():
                self.vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                    )
                )
                self.route_active = False
                self.route_completed = True
                self._route_mode = False
                self.route_agent = None
                print("Destination reached.")
                print("Selecting a new validated route...")
                if self.start_route_following():
                    print("Continuing drive on the next route.")
                    return True
                print("No valid next route available. Stopping vehicle.")
                self.stop_vehicle()
                return False

            control = self.route_agent.run_step()
            self.vehicle.apply_control(control)

            now = time.monotonic()
            if now - self._last_progress_print_s >= 2.0:
                self._print_route_progress_diagnostics()
                self._last_progress_print_s = now

            return True

        except Exception as exc:
            print(f"Route follower error: {exc}")
            self._stop_route_navigation()
            return False

    def _stop_route_navigation(self) -> None:
        self.route_active = False
        self._route_mode = False
        self.route_agent = None
        self.route_follower = None
        self.route_plan = []
        self.route_stats = {}

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

    Default mode uses local road-following control.
    --route uses official CARLA GlobalRoutePlanner + BasicAgent navigation.
    """

    parser = argparse.ArgumentParser(description="CARLA context standalone test")
    parser.add_argument(
        "--route",
        action="store_true",
        help="Drive a validated finite official CARLA route to a destination",
    )
    parser.add_argument(
        "--weather-demo",
        action="store_true",
        help="Advance the CARLA weather demo while the standalone test runs",
    )
    args = parser.parse_args()

    print("=" * 70)
    if args.route:
        print("CARLA OFFICIAL ROUTE TEST")
    else:
        print("CARLA CONTEXT TEST")
    print("=" * 70)

    context = CarlaContext()

    if not context.is_connected():
        print("ERROR: Could not connect to CARLA.")
        return

    print("CARLA connection successful.")
    print("Map:", context.get_map_name())

    print("\nFinding/spawning vehicle...")
    if not context.spawn_vehicle():
        print("ERROR: Vehicle could not be created.")
        return

    print("\nVehicle ready.")
    print("Smooth spectator-camera follow enabled.")

    context.follow_vehicle()

    if args.route:
        print("Starting official CARLA route navigation.")
        print("Press Ctrl+C to stop.\n")
        if not context.start_route_following():
            return

        try:
            while context._tick_route_follower():
                context.follow_vehicle()
                if args.weather_demo:
                    context.update_weather_demo()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n\nStopping CARLA route test...")
        finally:
            context.stop_vehicle()
            print("Vehicle stopped.")
            print("CARLA route test finished.")
        return

    print("Starting vehicle movement.")
    print("Press Ctrl+C to stop.\n")

    context.start_moving(throttle=0.4)

    try:
        while True:
            context.follow_vehicle()
            context.start_moving()
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
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopping CARLA test...")

    finally:
        context.stop_vehicle()
        print("Vehicle stopped.")
        print("CARLA context test finished.")


if __name__ == "__main__":
    main()