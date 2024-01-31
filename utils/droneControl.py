'''
Module that is used to interact with the drone using MAVLINK commands.
It is based on the pymavlink module.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2024-01-27
'''
import time
import threading

from pymavlink import mavutil

class DroneControl():
    def __init__(self) -> None:
        self.connection = None

    def emit_heartbeat(self):
        system_type = mavutil.mavlink.MAV_TYPE_GCS
        autopilot_type = mavutil.mavlink.MAV_AUTOPILOT_INVALID
        for _ in range(20):
            # Craft the heartbeat message
            self.connection.mav.heartbeat_send(
                system_type,
                autopilot_type,
                0,  # MAV_MODE_FLAG, set to 0 for GCS
                0,  # Custom mode, not used for GCS
                0,  # System status, not used for GCS
                0  # MAVLink version
            )
            print("emitted")
            time.sleep(1)

    def connect_to_drone(self):
        print("trying to connect to drone...")
        self.connection = mavutil.mavlink_connection('COM11', baud=57600)
        for _ in range(1):
            self.connection.wait_heartbeat()
            print("heartbeat received from system %u" % 
                self.connection.target_component)
            time.sleep(1)
    
    def arm(self):
        print('arming device')
        self.connection.mav.command_long_send(self.connection.target_system,
                                          self.connection.target_component,
                                          mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                          0, 1, 0, 0, 0, 0, 0, 0)
        msg = self.connection.recv_match(type='GPS_RAW_INT', blocking=True)
        if not msg:
            return
        print("armed: {}".format(msg))
        heartbeat_thread = threading.Thread(target=self.emit_heartbeat)
        heartbeat_thread.start()

    def disarm(self):
        print('disarming device')
        self.connection.mav.command_long_send(self.connection.target_system,
                                          self.connection.target_component,
                                          mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                          0, 0, 0, 0, 0, 0, 0, 0)
        print("disarmed")

        

    def fly_forward(self):
        self.connection = mavutil.mavlink_connection('COM11', baud=57600)
        pos = (0, 0, 0)
        vel = (1, 0, 0)
        acc = (0, 0, 0)
        self.__ned_command(pos, vel, acc)
        self.connection.mav.set_position_target_local_ned_send(
            0, #system time in ms
            self.connection.target_system,
            self.connection.target_component,
            9, #MAV_FRAME_BODY_OFFSET_NED
            0b110111000111, #use velocity
            0, 0, 0, # x, y, z (m)
            1, 0, 0, # vx, vy, vz (m/s)
            0, 0, 0, # ax, ay, az (m/s^2)
            0, 0 # yaw (rad), yaw rate (rad/s) 
        )

    def __ned_command(self, x, dx, d2x):
        '''
        sends SET_POSITION_TARGET_LOCAL_NED command.
        This command needs to be sent at at least 3Hz.
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
        if not len(x) == len(dx) == len(d2x) == 3:
            return
        self.connection.mav.set_position_target_local_ned_send(
            time.time() * 1e3, #system time in ms (can simply be set to 0)
            self.connection.target_system,
            self.connection.target_component,
            9, #MAV_FRAME_BODY_OFFSET_NED
            int(0b110111000111), #use velocity
            *x, # x, y, z (m)
            *dx, # vx, vy, vz (m/s)
            *d2x, # ax, ay, az (m/s^2)
            0, 0 # yaw (rad), yaw rate (rad/s) 
        )