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

            throttle = max(
                0.0,
                min(
                    1.0,
                    throttle,
                ),
            )

            steer = max(
                -1.0,
                min(
                    1.0,
                    steer,
                ),
            )

            self.vehicle.set_autopilot(
                False
            )

            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=0.0,
                )
            )

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

        weather = (
            self.get_weather_context()
        )

        time_context = (
            self.get_time_context()
        )

        position = (
            self.get_vehicle_position()
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