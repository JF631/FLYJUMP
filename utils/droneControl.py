'''
Module that is used to interact with the drone using MAVLINK commands.
It is based on the pymavlink module (https://github.com/ArduPilot/pymavlink).
Helpfull pymavlink and message definition examples can be found under
https://mavlink.io/en/mavgen_python/ and
https://ardupilot.org/copter/docs/common-mavlink-mission-command-messages-mav_cmd.html

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2024-01-27
'''
import time
from dataclasses import dataclass

from pymavlink import mavutil
from PyQt5.QtCore import QThread

from utils.controlsignals import DroneSignals

@dataclass
class Messages:
    GPS_INFO = 'GPS_RAW_INT'
    HEARTBEAT = 'HEARTBEAT'
    CMD_RESULT = 'MAV_RESULT'
    SYS_STATUS = 'SYS_STATUS'
    ATTITUDE = 'ATTITUDE'   
    STATUSTEXT = 'STATUSTEXT'
    ACKNOWLEDGED = 'COMMAND_ACK'
    TIME = 'SYSTEM_TIME'


@dataclass
class Commands:
    ARM_DISARM = 'MAV_CMD_COMPONENT_ARM_DISARM'
    PREARM_CHECKS = 'MAV_CMD_RUN_PREARM_CHECKS'
    RETURN_TO_LAUNCH = 'MAV_CMD_NAV_RETURN_TO_LAUNCH'
    TAKEOFF = 'MAV_CMD_NAV_TAKEOFF'


class DroneConnection(QThread):
    '''
    Module that offers a Mavlink drone connection.
    It is based on the pymavink library.
    '''
    def __init__(self, signals: DroneSignals) -> None:
        super().__init__()
        self.signals = signals
        self.connection = self.__establish()
        self.request_messages(frequency=3)

    def is_active(self):
        '''
        checks if valid connectio object is present.

        Returns
        -------
        active : bool
            True if connection is valid, False otherwise.
        '''
        if self.connection is None:
            return False
        return True

    def __establish(self):
        '''
        Tries to initialize a serial connection.
        Baudrate is dafaulted for Telemetry Radio to 57600 baud/s.
        '''
        try:
            return mavutil.mavlink_connection('COM11', baud=57600)
        except Exception as e:
            self.signals.status_text.emit("No telemetry radio was found")
            print(f"failed to establish drone connection: {e}")

    def __publish_heartbeat(self):
        '''
        Publishes heartbeat from groundstation.
        If heartbeat stops the connection is considered as lost.
        '''
        if not self.is_active():
            return
        system_type = mavutil.mavlink.MAV_TYPE_GCS
        autopilot_type = mavutil.mavlink.MAV_AUTOPILOT_INVALID
        self.connection.mav.heartbeat_send(
            system_type,
            autopilot_type,
            0,  # MAV_MODE_FLAG, set to 0 for GCS
            0,  # Custom mode, not used for GCS
            0,  # System status, not used for GCS
            0  # MAVLink version
        )

    def request_messages(self, frequency=0):
        '''
        Every message that needs to be checked regulary must be requested
        once.
        Following messages are requested:
        Messages.ACKNOWLEDGED, Messages.GPS_INFO,
        Messages.STATUSTEXT, Messages.SYS_STATUS

        Parameters
        ----------
        frequency : int
            Frequency interval at which the messages want to be received.
        '''
        if not self.is_active():
            return
        messages_to_request = [Messages.ACKNOWLEDGED, Messages.GPS_INFO,
                               Messages.STATUSTEXT, Messages.SYS_STATUS]
        for message in messages_to_request:
            message_id = getattr(mavutil.mavlink, 'MAVLINK_MSG_ID_' + message)
            if frequency:
                frequency = 1e6 / frequency
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                message_id,
                frequency,
                0, 0, 0, 0,
                0,
            )

    def get_one_time_msg(self, message: Messages, blocking=False, timeout=2):
        '''
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
        '''
        if not self.is_active():
            return
        message_id = getattr(mavutil.mavlink, 'MAVLINK_MSG_ID_' + message)
        self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                message_id,
                0,
                0, 0, 0, 0,
                0,
            )
        rtrn = self.get_msg(message, blocking, timeout)
        return rtrn

    def enter_guided_mode(self):
        '''
        Enter Guided flight mode.
        Thus, the control is handed over to the ground station.

        Returns
        -------
        ack : str
            Mavlink CMD_ACK message result.
        '''
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                1, # enable custom modes
                4, # guided mode (modes : https://ardupilot.org/copter/docs/parameters.html#fltmode1) 
                0, 0, 0, 0, 0,
            )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True)
        return rtrn

    def takeoff(self, height=5):
        '''
        Start takeoff to height.

        Parameters
        ----------
        height : int
            height in meters to which the drone will climb.
        '''
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 
                0, 0, 0, 0, 0, 0,
                height, # takeoff height [m] 
            )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True)
        return rtrn

    def return_to_home(self):
        '''
        Enter RTL mode.
        Drone will return to launch position and land.

        Returns
        -------
        ack : str
            Mavlink CMD_ACK message result.
        '''
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 
                0, 0, 0, 0, 0, 0, 0, #no params needed 
            )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True, timeout=3)
        return rtrn

    def land(self):
        '''
        Tries to land drone where it is at the moment.
        '''
        if not self.is_active():
            return
        self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 
                0, 0, 0, 0, 
                0, 0, #latitude / longitude
                0, #no params needed 
            )
        rtrn = self.get_msg(Messages.ACKNOWLEDGED, blocking=True, timeout=3)
        return rtrn

    def __get_status(self):
        '''
        Tries to get status text from drone.
        Is always not blocking.
        Thus, this function only MIGHT return a message

        Returns
        -------
        message : str
            Status text received or None
        '''
        if not self.is_active():
            return
        msg = self.get_msg(Messages.STATUSTEXT, blocking=True, timeout=1)
        if msg and 'PreArm' in msg.text:
            self.signals.status_text.emit(msg.text)

    def __receive_heartbeat(self, timeout=2):
        '''
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
        '''
        if not self.is_active():
            return None
        msg = self.get_msg(Messages.HEARTBEAT, blocking=True,
                            timeout=timeout)
        return msg

    def get_msg(self, msg:Messages, blocking=False, timeout=2):
        '''
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
        '''
        if not self.is_active():
            return
        msg = self.connection.recv_match(type=msg, blocking=blocking,
                                        timeout=timeout)
        return msg

    def ned_command(self, x, dx, d2x):
        '''
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
        '''
        if not self.is_active():
            return
        if not len(x) == len(dx) == len(d2x) == 3:
            return
        self.connection.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                time.time() * 1e3,
                self.connection.target_system,
                self.connection.target_component,
                9, #MAV_FRAME_LOCAL_NED
                int(0b110111000111), #use velocity
                *x, # x, y, z (m)
                *dx, # vx, vy, vz (m/s)
                *d2x, # ax, ay, az (m/s^2)
                0, 0 # yaw (rad), yaw rate (rad/s) 
            ))

    def run(self):
        '''
        Runs in background, when object.start() is called.
        Emits heartbeat and checks for drones' heartbeat in background
        '''
        while not self.is_active():
            self.connection = self.__establish()
            self.signals.connection_changed.emit(False)
            self.msleep(1000)
        self.signals.connection_changed.emit(True)
        while not self.isInterruptionRequested():
            self.__publish_heartbeat()
            msg = self.__receive_heartbeat(timeout=5)
            if not msg:
                self.signals.connection_changed.emit(False)
            # if msg:
            #     print(msg)
            self.__get_status()
            self.msleep(200)


class DroneControl():
    '''
    Module that offers simple drone control.
    Interoperates with the DroneConnection module.
    '''
    def __init__(self, drone_connection: DroneConnection,
                 signals: DroneSignals) -> None:
        self.connection = drone_connection.connection
        self.signals = signals
        self.drone_worker = drone_connection
        self.connect_signals_and_slots()

    def connect_signals_and_slots(self):
        self.signals.connection_changed.connect(self.update_connection)

    def update_connection(self, _):
        '''
        pyqtslot, called when the drone connection status changes.
        '''
        self.connection = self.drone_worker.connection

    def arm(self):
        '''
        Arm the drone.
        '''
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)
        result = self.drone_worker.get_msg(Messages.ACKNOWLEDGED, blocking=True)
        print(result)
    
    def return_home(self):
        ack = self.drone_worker.return_to_home()

    def disarm(self):
        '''
        Disarm the drone.
        '''
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0)

    def takeoff(self, height=5):
        '''
        Tries to let the drone take off

        Parameters
        ----------
        height : int
            heigth in meters to which the drone will climb.
        '''
        guided_ack = self.drone_worker.enter_guided_mode()
        ack = self.drone_worker.takeoff(height)
        if not ack:
            ack = self.drone_worker.takeoff(height)
        print(ack)
    
    def land(self):
        self.drone_worker.land()

    def fly_forward(self):
        '''
        Fly the drone forward at 1 m/s (positive x direction)
        '''
        pos = (0, 0, 0)
        vel = (1, 0, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
    
    def fly_backwards(self):
        '''
        Fly the drone backwards at 1 m/s (negative x direction)
        '''
        pos = (0, 0, 0)
        vel = (-1, 0, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
    
    def fly_right(self):
        '''
        Fly the drone to the right at 1 m/s (positive y direction)
        '''
        pos = (0, 0, 0)
        vel = (0, 1, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
    
    def fly_left(self):
        '''
        Fly the drone to the left at 1 m/s (negative y direction)
        '''
        pos = (0, 0, 0)
        vel = (0, -1, 0)
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
    
    def climb(self):
        '''
        Climb at 1 m/s (negative z direction)
        '''
        pos = (0, 0, 0)
        vel = (0, 0, -1) # careful!!!! -1 means 1m/s upwards!
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)
    
    def sink(self):
        '''
        Sink at 1 m/s (positive z direction)
        '''
        pos = (0, 0, 0)
        vel = (0, 0, 1) # careful!!!! -1 means 1m/s upwards!
        acc = (0, 0, 0)
        self.drone_worker.ned_command(pos, vel, acc)

    def run_arming_checks(self):
        '''
        run pre arm checks on drone manually.
        '''
        if not self.drone_worker.is_active():
            return
        for _ in range(4):
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_RUN_PREARM_CHECKS,
                0, 1, 0, 0, 0, 0, 0, 0)
            ack = self.drone_worker.get_msg(Messages.ACKNOWLEDGED,
                                            blocking=True, timeout=1)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    status = self.drone_worker.get_one_time_msg(
                        Messages.STATUSTEXT, blocking=False)
                    if status and 'PreArm' in status.text:
                        self.signals.status_text.emit(status.text)
            
