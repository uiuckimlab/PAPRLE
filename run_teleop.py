import sys
import signal
import time
import cv2
from threading import Thread
import warnings

from configs import BaseConfig
follower_config, leader_config, env_config = BaseConfig().parse()

# Import order is important , don't know fancy way to do this
if env_config.name == "ros":
    sys.path.append("/usr/lib/python3/dist-packages")
    sys.path.append("/opt/ros/noetic/lib/python3.8/site-packages/")
    import rospy
    import moveit_commander
    import pinocchio
elif env_config.name == "ros2":
    import rclpy
elif env_config.name == "isaacgym":
    import isaacgym # this needs to be imported before torch

from paprle.teleoperator import Teleoperator
from paprle.follower import Robot
from paprle.leaders import LEADERS_DICT
from paprle.envs import ENV_DICT
#from src.feedback.feedback import Feedback
from threading import Thread

TIME_DEBUG = False
class Runner:
    def __init__(self, robot_config, device_config, env_config):
        self.robot_config, self.device_config, self.env_config = robot_config, device_config, env_config

        self.TELEOP_DT = robot_config.robot_cfg.teleop_dt = device_config.teleop_dt
        self.robot = Robot(robot_config)
        self.controller = LEADERS_DICT[device_config.type](self.robot, device_config, env_config) # Get signals from teleop devices, outputs joint positions or eef poses as teleop commands.
        self.teleop = Teleoperator(self.robot, device_config, env_config, render_mode='mujoco') # Solving IK for joint positions if not already given, check collision, and output proper joint positions.
        self.env = ENV_DICT[env_config.name](self.robot, device_config, env_config, render_mode='', controller=self.controller) # Actually send joint positions to the robot.
        self.env.vis_info = self.controller.update_vis_info(self.env.vis_info)

        # self.feedback = Feedback(self.controller, self.teleop, self.env, robot_config, device_config)
        # self.feedback_thread = Thread(target=self.feedback.send_feedback)
        # self.feedback_thread.start()

        self.shutdown = False
        signal.signal(signal.SIGINT, self.shutdown_handler)
        self.reset = False

        self.last_log_time = None
        #self.render_thread = Thread(target=self.render_thread_func)
        #self.render_thread.start()


    def shutdown_handler(self, sig, frame):
        print("Shutting down the system..")
        self.env.close()
        print("🚫🌏 Env closed")
        self.teleop.close()
        print("🚫🤖 Teleop closed")
        self.controller.close()
        print("🚫🎮 Leader closed")
        sys.exit()

    def log_time(self, msg=''):
        if self.last_log_time is not None and msg != '':
            print(msg, time.time() - self.last_log_time)
        self.last_log_time = time.time()
        return

    def render_thread_func(self):
        while True:
            if self.env.view_im is None or self.reset:
                time.sleep(0.01)
                continue
            cv2.imshow("View", self.env.view_im[:,:,::-1])
            key = cv2.waitKey(1)
            if key == ord("r"):
                self.reset = True
                print("Reset signal detected")

            if key == ord("q"):
                self.shutdown = True
                print("Shutting down signal detected")
                self.shutdown_handler(None, None)
        return
    def run(self):

        init_env_qpos = self.env.reset() # Move the robot to the default positionm
        shutdown = self.controller.launch_init(init_env_qpos) # Wait in the initialize function until the controller is ready (for visionpro and gello)
        if shutdown: return
        while not self.controller.is_ready:
            if self.shutdown: return
            time.sleep(0.01)

        if TIME_DEBUG: self.log_time('')

        initial_command = self.controller.get_status()
        initial_qpos = self.teleop.step(initial_command, initial=True) # process initial command
        self.env.initialize(initial_qpos) # slowly move to initial qpos, using moveit
        if TIME_DEBUG: self.log_time('Initialization Time')

        iter = 0
        status_str = "Running... "
        bar_list = ["|", "|", "|", "|", "/", "/", "/", "/", "-", "-", "-", "-", "\\", "\\", "\\", "\\"]
        while True:
            print(status_str + bar_list[iter%16], end="\r")
            start_time = time.time()
            if TIME_DEBUG:
                print("===========================")
                self.log_time('Start Loop')

            # 1. Get command from controller
            command = self.controller.get_status()

            if TIME_DEBUG: self.log_time('Controller Get Status')

            # If reset signal is detected, reset the environment
            if self.controller.require_end or self.reset:
                self.reset = False
                init_env_qpos = self.env.reset()
                self.teleop.reset()
                shutdown = self.controller.launch_init(init_env_qpos)  # Wait in the initialize function until the controller is ready (for visionpro and gello)
                if shutdown: return
                while not self.controller.is_ready:
                    if self.shutdown: return
                    time.sleep(0.01)
                self.controller.close_init()
                command = self.controller.get_status()
                initial_qpos = self.teleop.step(command, initial=True)
                self.env.initialize(initial_qpos)
                if TIME_DEBUG: self.log_time('Reset Time')
                self.controller.require_end = False
                continue

            # 2. Get joint positions from teleop
            qposes = self.teleop.step(command)

            if TIME_DEBUG: self.log_time('Teleop Step Time ')

            # 3. Send joint positions to the robot
            self.env.step(qposes)

            if TIME_DEBUG: self.log_time('Env Step Time')

            # 4. Sleep for the rest of the time
            loop_time = time.time() - start_time
            left_time = max(self.TELEOP_DT - loop_time, 0.0)
            time.sleep(left_time)
            #if left_time <= 0.0:
            #    warnings.warn(f"Warning: Loop time is too long - {loop_time}", stacklevel=2)
            iter += 1

if __name__ == "__main__":
    runner = Runner(follower_config, leader_config, env_config)
    runner.run()
