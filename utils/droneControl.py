"""
Module that is used to interact with the drone using MAVLINK commands.
It is based on the pymavlink module (https://github.com/ArduPilot/pymavlink).
Helpfull pymavlink and message definition examples can be found under
https://mavlink.io/en/mavgen_python/ and
https://ardupilot.org/copter/docs/common-mavlink-mission-command-messages-mav_cmd.html

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2024-01-27
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum

from pymavlink import mavutil
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox

from utils.controlsignals import DroneSignals


@dataclass
class Messages:
    """
    Defines mavlink messages that are requested (regulary) from the drone.
    """

    GPS_INFO: str = "GPS_RAW_INT"
    GPS_POS_GLOBAL: str = "GLOBAL_POSITION_INT"
    RNGFND_ALT: str = "DISTANCE_SENSOR"
    HEARTBEAT: str = "HEARTBEAT"
    CMD_RESULT: str = "MAV_RESULT"
    SYS_STATUS: str = "SYS_STATUS"
    ATTITUDE: str = "ATTITUDE"
    STATUSTEXT: str = "STATUSTEXT"
    ACKNOWLEDGED: str = "COMMAND_ACK"
    TIME: str = "SYSTEM_TIME"
    BATTERY_INFO: str = "BATTERY_STATUS"


class Mode(Enum):
    """
    Defines the flying modes that the software supports
    Supported modes are: GROUND, FLYING, LANDING, RTL
    """

    GROUND = 0
    FLYING = 1
    LANDING = 2
    RTL = 3


class DroneConnection(QThread):
    """
    Module that offers a Mavlink drone connection.
    It is based on the pymavink library.
    """

    def __init__(self, signals: DroneSignals) -> None:
        super().__init__()
        self.signals = signals
        self.connection = None
        self.serial_port = None
        self.heartbeat_thread: threading.Thread = None

    def close(self):
        try:
            if self.is_active():
                self.connection.close()
        except:
            pass

    def is_active(self):
        """
        checks if valid connection object is present.

        Returns
        -------
        active : bool
            True if connection is valid, False otherwise.
        """
        if self.connection is None:
            return False
        return True

    def init_serial_port(self, port: str):
        """
        Tries to initialize a serial connection.
        Baudrate is dafaulted for Telemetry Radio to 57600 baud/s.
        """
        try:
            self.connection = mavutil.mavlink_connection(port, baud=57600)
            self.serial_port = port
            self.request_messages(frequency=10)
            self.signals.connection_changed.emit(True)
            return True
        except Exception as e:
            QMessageBox.critical(
                None,
                "Serial port error",
                f"Serial port {port} could not be opened!",
                QMessageBox.Ok,
            )
            self.signals.status_text.emit("No telemetry radio was found")
            self.connection = None
            self.serial_port = None
            return False
            print(f"failed to establish drone connection: {e}")

    def __publish_heartbeat(self):
        """
        Publishes heartbeat from groundstation.
        If heartbeat stops the connection is considered as lost.
        """
        if not self.is_active():
            return
        system_type = mavutil.mavlink.MAV_TYPE_GCS
        autopilot_type = mavutil.mavlink.MAV_AUTOPILOT_INVALID
        while True:
            self.connection.mav.heartbeat_send(
                system_type,
                autopilot_type,
                0,  # MAV_MODE_FLAG, set to 0 for GCS
                0,  # Custom mode, not used for GCS
                0,  # System status, not used for GCS
                0,  # MAVLink version
            )
            time.sleep(1)

    def request_messages(self, frequency=0):
        """
        Every message that needs to be checked regulary must be requested
        once.
        Following messages are requested:
        Messages.ACKNOWLEDGED, Messages.GPS_INFO,
        Messages.STATUSTEXT, Messages.SYS_STATUS

        Parameters
        ----------
        frequency : int
            Frequency interval at which the messages want to be received.
        """
        if not self.is_active():
            return
        messages_to_request = [
            Messages.ACKNOWLEDGED,
            Messages.GPS_INFO,
            Messages.STATUSTEXT,
            Messages.SYS_STATUS,
            Messages.GPS_POS_GLOBAL,
            Messages.BATTERY_INFO,
            Messages.RNGFND_ALT
        ]
        for message in messages_to_request:
            message_id = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_" + message)
            if frequency:
                frequency = 1e6 / frequency
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                frequency,
                0,
                0,
                0,
                0,
                0,
            )

    def get_one_time_msg(self, message: Messages, blocking=False, timeout=2):
        """
        Request a sepcific Mavlink message one time.

        Parameters
        -----------
        msg : Messages
            message to check for.
        blocking : bool
            wait for message. Might block the current thread.
            defaults to False (not blocking).
        timeout : int
            timout in seconds after which the function returns None if no
            message was received.

        Returns
        -------
        message : str
            Message text received or None (if no message received after
            timeout)

        CAUTION
        --------
        can block the current thread when timout was chosen too long.
        """
        if not self.is_active():
            return
        message_id = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_" + message)
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        rtrn = self.get_msg(message, blocking, timeout)
        return rtrn

    def enter_guided_mode(self):
        """
        Enter Guided flight mode.
        Thus, the control is handed over to the ground station.

        Returns
        -------
        ack : str
            Mavlink CMD_ACK message result.
        """
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            1,  # enable custom modes
            4,  # guided mode (modes : https://ardupilot.org/copter/docs/parameters.html#fltmode1)
            0,
            0,
            0,
            0,
            0,
        )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True)
        return rtrn

    def takeoff(self, height=5):
        """
        Start takeoff to height.

        Parameters
        ----------
        height : int
            height in meters to which the drone will climb.
        """
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            height,  # takeoff height [m]
        )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True)
        return rtrn

    def return_to_home(self):
        """
        Enter RTL mode.
        Drone will return to launch position and land.

        Returns
        -------
        ack : str
            Mavlink CMD_ACK message result.
        """
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # no params needed
        )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True, timeout=3)
        return rtrn

    def land(self):
        """
        Tries to land drone where it is at the moment.
        """
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # latitude / longitude
            0,  # no params needed
        )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True, timeout=3)
        return rtrn

    def check_status(self):
        """
        Tries to get status text from drone.
        Is always not blocking.
        Thus, this function only MIGHT return a message

        Returns
        -------
        message : str
            Status text received or None
        """
        if not self.is_active():
            return
        current_time = None
        time_str = None
        msg = self.get_msg(Messages.STATUSTEXT, blocking=True, timeout=1)
        if msg and "PreArm" in msg.text:
            current_time = time.time()
            current_time = time.localtime(current_time)
            time_str = time.strftime("%H:%M:%S", current_time)
            rtrn = "[{}]: {}".format(time_str, msg.text)
            self.signals.status_text.emit(rtrn)

    def __receive_heartbeat(self, timeout=2):
        """
        Checks for heartbeat.
        If no heartbeat is received for timeout seconds, None is returned.
        Connection should then be interrupted!

        Parameters
        ----------
        timeout : int
            timout in seconds after which the function returns None if no
            message was received.

        Returns
        -------
        heartbeat : str
            received heartbeat message or None (after timoeout)

        CAUTION
        --------
        This function is ALWAYS thread blocking for a max of timeout seconds!
        """
        if not self.is_active():
            return None
        msg = self.get_msg(Messages.HEARTBEAT, blocking=True, timeout=timeout)
        return msg

    def get_msg(self, msg: Messages, blocking=False, timeout=2):
        """
        Get specific mavlink message.

        Parameters
        -----------
        msg : Messages
            message to check for.
        blocking : bool
            wait for message. Might block the current thread.
            defaults to False (not blocking).
        timeout : int
            timout in seconds after which the function returns None if no
            message was received.

        Returns
        -------
        message : str
            Message text received or None (if no message received after
            timeout)

        CAUTION
        --------
        can block the current thread when timout was chosen too long.
        """
        if not self.is_active():
            return
        msg = self.connection.recv_match(type=msg, blocking=blocking, timeout=timeout)
        return msg

    def ned_command(self, x, dx, d2x):
        """
        sends SET_POSITION_TARGET_LOCAL_NED command.
        This command needs to be sent at at least 0.33Hz.
        However, 1Hz is recommended for a smooth flight.

        Parameters
        ----------
        x : List-like
            position values (x,y,z)
        dx : List-like
            velocity values (vx, vy, vz)
        d2x : List-like
            acceleration values (ax, ay, az)

        INFO
        -----
        the bitmask used in the function is read from right to left.
        Moreover the mask is inverted (NAND link).
        0 means the according value is used, 1 means the value is ignored.
        Example:
                    110111000111
                             ^^^
                             position disabled
                          ^^^
                          velocity enabled
                       ^^^
                       acceleration disabled
                      ^
                      bit 10 is UNUSED!!!
                    ^^
                    yaw and yaw rate are ignored
        """
        if not self.is_active():
            return
        if not len(x) == len(dx) == len(d2x) == 3:
            return
        self.connection.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                0,
                self.connection.target_system,
                self.connection.target_component,
                12,  # MAV_FRAME_BODY_FRD
                int(0b110111000111),  # use velocity
                x[0],
                x[1],
                x[2],  # x, y, z (m)
                dx[0],
                dx[1],
                dx[2],  # vx, vy, vz (m/s)
                d2x[0],
                d2x[1],
                d2x[2],  # ax, ay, az (m/s^2)
                0,
                0,  # yaw (rad), yaw rate (rad/s)
            )
        )

    def prepare_status_message(self, gps_message, gps_raw_message,
                               rngfnd_message):
        alt = rngfnd_message.get("current_distance") / 1e2 # in m
        '''maximal range of current range finder (benewake tf mini plus) is
        6m, thus we use gps altitude above 6m'''
        if alt > 6.0:
            alt = gps_message.get("relative_alt") / 1e3 # in m
        return {
            "velocity": gps_raw_message.get("vel") / 1e2,  # in m/s
            "relative_alt": alt, # in m
            "satelites": gps_raw_message.get("satellites_visible"),
            "fix_type": gps_raw_message.get("fix_type") # 1D, 2D, 3D or 4D
        }

    def run(self):
        """
        Runs in background, when object.start() is called.
        Emits heartbeat and checks for drones' heartbeat in background
        """
        while not self.is_active():
            if not self.init_serial_port(self.serial_port):
                return
            self.signals.connection_changed.emit(False)
            self.msleep(1000)
        self.signals.connection_changed.emit(True)
        self.heartbeat_thread = threading.Thread(
            name="heartbeatThread", target=self.__publish_heartbeat
        )
        self.heartbeat_thread.start()
        while not self.isInterruptionRequested():
            msg = self.__receive_heartbeat(timeout=5)
            if not msg:
                self.signals.connection_changed.emit(False)
            gps_message = self.get_msg(Messages.GPS_INFO, blocking=False)
            gps_global = self.get_msg(Messages.GPS_POS_GLOBAL, blocking=False)
            rngfnd_alt = self.get_msg(Messages.RNGFND_ALT, blocking=False)
            if gps_message and gps_global and rngfnd_alt:
                gps_status = self.prepare_status_message(
                    gps_global.to_dict(), gps_message.to_dict(),
                    rngfnd_alt.to_dict(),
                )
                self.signals.vehicle_gps_status.emit(gps_status)
            battery_status = self.get_msg(Messages.BATTERY_INFO,
                                          blocking=False)
            if battery_status:
                self.signals.vehicle_battery_status.emit(
                    battery_status.to_dict())
            # if msg:
            #     print(msg)
            self.check_status()
            self.msleep(1000)


class DroneControl:
    """
    Module that offers simple drone control.
    Interoperates with the DroneConnection module.
    """

    def __init__(
        self, drone_connection: DroneConnection, signals: DroneSignals
    ) -> None:
        self.connection = drone_connection.connection
        self.signals = signals
        self.drone_worker = drone_connection
        self.current_status = Mode.GROUND
        self.connect_signals_and_slots()

    def connect_signals_and_slots(self):
        self.signals.connection_changed.connect(self.update_connection)

    def update_connection(self, _):
        """
        pyqtslot, called when the drone connection status changes.
        """
        self.connection = self.drone_worker.connection

    def arm(self):
        """
        Arm the drone.
        """
        if not self.connection:
            return
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        result = self.drone_worker.get_msg(Messages.ACKNOWLEDGED, blocking=True)
        print(result)

    def return_home(self):
        """
        Initiates return to home (start position) mode.
        """
        if self.current_status != Mode.FLYING:
            QMessageBox.information(
                None,
                "RTL not initiated",
                "For returning home, the drone needs to fly first :)",
                QMessageBox.Ok,
            )
            return
        rtl_ack = self.drone_worker.return_to_home()
        if rtl_ack and rtl_ack.command == 20:
            if rtl_ack.result != 0:
                QMessageBox.critical(
                    None,
                    "Landing not initiated",
                    "RTL mode command has been rejected by the drone",
                    QMessageBox.Ok,
                )
                return
            self.current_status = Mode.RTL

    def disarm(self):
        """
        Disarm the drone.
        """
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def takeoff(self, height=5):
        """
        Tries to let the drone take off

        Parameters
        ----------
        height : int
            heigth in meters to which the drone will climb.
        """
        guided_ack = self.drone_worker.enter_guided_mode()
        if guided_ack and guided_ack.command == 176 and guided_ack.result == 0:
            takeoff_ack = self.drone_worker.takeoff(height)
            if takeoff_ack and takeoff_ack.command == 22:
                if takeoff_ack.result == 0:
                    self.current_status = Mode.FLYING
                    return
                if takeoff_ack.result == 4:
                    QMessageBox.critical(
                        None,
                        "Takeoff failed",
                        """The takeoff has been rejected by the drone, please\
                        check the prearm check list""",
                        QMessageBox.Ok,
                    )
            self.current_status = Mode.FLYING

    def land(self):
        """
        Tries to land the drone at the exact position where it currently
        flies.

        """
        if self.current_status != Mode.FLYING:
            QMessageBox.information(
                None,
                "Landing not initiated",
                "For landing, the drone needs to fly first :)",
                QMessageBox.Ok,
            )
            return
        land_ack = self.drone_worker.land()
        if land_ack and land_ack.command == 21:
            if land_ack.result != 0:
                QMessageBox.critical(
                    None,
                    "Landing not initiated",
                    "The landing command has been rejected by the drone",
                    QMessageBox.Ok,
                )
                return
            self.current_status = Mode.LANDING

    def fly_forward(self, velocity: int = 1):
        """
        Fly the drone forward at a given speed (positive x direction)
        Speed is limitted between 0m/s and 8m/s

        Parameters
        ----------
        velocity : int
            velocity at which the drone should fly forward.
            Defaults to 1 m/s.
        """
        if velocity < 0 or velocity > 8:
            return
        print("flying forward...")
        pos = (0, 0, 0)
        vel = (velocity, 0, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
        time.sleep(0.33)

    def fly_backwards(self, velocity: int = 1):
        """
        Fly the drone backwards at a given speed (negative x direction)
        Speed is limitted between 0m/s and 8m/s

        Parameters
        ----------
        velocity : int
            velocity at which the drone should fly backward.
            Defaults to 1 m/s.
        """
        pos = (0, 0, 0)
        vel = (-velocity, 0, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
        time.sleep(0.33)

    def fly_right(self, velocity: int = 1):
        """
        Fly the drone to the right at a given speed (positive y direction).
        Speed is limitted between 0m/s and 8m/s

        Parameters
        ----------
        velocity : int
            velocity at which the drone should fly to the right.
            Defaults to 1 m/s.
        """
        pos = (0, 0, 0)
        vel = (0, velocity, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
        time.sleep(0.33)

    def fly_left(self, velocity: int = 1):
        """
        Fly the drone to the right at a given speed (negative y direction).
        Speed is limitted between 0m/s and 8m/s

        Parameters
        ----------
        velocity : int
            velocity at which the drone should fly to the left.
            Defaults to 1 m/s.
        """
        pos = (0, 0, 0)
        vel = (0, -velocity, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
        time.sleep(0.33)

    def climb(self, velocity: int = 1):
        """
        Climb at velocity m/s (negative z direction)

        Parameters
        ----------
        velocity : int
            velocity at which the drone should climb.
            Defaults to 1 m/s.
        """
        pos = (0, 0, 0)
        vel = (0, 0, -velocity)  # careful!!!! -velocity means velocity m/s upwards!
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
        time.sleep(0.33)

    def sink(self, velocity: int = 1):
        """
        Sink at velocity m/s (positive z direction)

        Parameters
        ----------
        velocity : int
            velocity at which the drone should sink.
            Defaults to 1 m/s.
        """
        pos = (0, 0, 0)
        vel = (0, 0, velocity)  # careful!!!! -1 means 1m/s upwards!
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
        time.sleep(0.33)

    def run_arming_checks(self):
        """
        run pre arm checks on drone manually.
        It is blocking!!!

        Thus, please run it in an extra thread.
        """
        if not self.drone_worker.is_active():
            return
        for _ in range(4):
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_RUN_PREARM_CHECKS,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            ack = self.drone_worker.get_msg(
                Messages.ACKNOWLEDGED, blocking=True, timeout=1
            )
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.drone_worker.check_status()
            time.sleep(1)
