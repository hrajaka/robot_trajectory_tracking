#!/usr/bin/env python

"""
Starter script for lab1. 
Author: Chris Correa
"""
import numpy as np
import math
import matplotlib.pyplot as plt

from utils.utils import *

try:
    import rospy
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except:
    pass


class MotionPath:
    def __init__(self, limb, kin, total_time, current_pos, target_pos):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`baxter_pykdl.baxter_kinematics` or :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        total_time : float
            number of seconds you wish the trajectory to run for
        """
        self.limb = limb
        self.kin = kin
        self.total_time = total_time
        self.current_pos = current_pos
        self.target_pos = target_pos

        times = np.linspace(0, total_time, 100)
        pos = np.array([0, 0, 0])
        for t in times:
            pos = np.vstack((pos, self.target_position(t)))
        pos = pos[1:-1, :]
        #print(pos)
        '''
        plt.figure()
        plt.grid(True)
        plt.title('Path')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(pos[:, 0], pos[:, 1], color='r', marker='.')
        plt.axis('equal')
        plt.show()
        '''
        

    def target_position(self, time):
        """
        Returns where the arm end effector should be at time t

        Parameters
        ----------
        time : float        

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired x,y,z position in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the arm's desired x,y,z velocity in workspace coordinates
        at time t

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired velocity in workspace coordinates of the end effector
        """
        pass

    def target_acceleration(self, time):
        """
        Returns the arm's desired x,y,z acceleration in workspace coordinates
        at time t

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired acceleration in workspace coordinates of the end effector
        """
        pass

    def plot(self, num=300):
        times = np.linspace(0, self.total_time, num=num)
        target_positions = np.vstack([self.target_position(t) for t in times])
        target_velocities = np.vstack([self.target_velocity(t) for t in times])

        plt.figure()
        plt.subplot(3, 2, 1)
        plt.plot(times, target_positions[:, 0], label='Desired')
        plt.xlabel("Time (t)")
        plt.ylabel("X Position")

        plt.subplot(3, 2, 2)
        plt.plot(times, target_velocities[:, 0], label='Desired')
        plt.xlabel("Time (t)")
        plt.ylabel("X Velocity")

        plt.subplot(3, 2, 3)
        plt.plot(times, target_positions[:, 1], label='Desired')
        plt.xlabel("time (t)")
        plt.ylabel("Y Position")

        plt.subplot(3, 2, 4)
        plt.plot(times, target_velocities[:, 1], label='Desired')
        plt.xlabel("Time (t)")
        plt.ylabel("Y Velocity")

        plt.subplot(3, 2, 5)
        plt.plot(times, target_positions[:, 2], label='Desired')
        plt.xlabel("time (t)")
        plt.ylabel("Z Position")

        plt.subplot(3, 2, 6)
        plt.plot(times, target_velocities[:, 2], label='Desired')
        plt.xlabel("Time (t)")
        plt.ylabel("Z Velocity")

        plt.show()

    def to_robot_trajectory(self, num_waypoints=300, jointspace=True):
        """
        Parameters
        ----------
        num_waypoints : float
            how many points in the :obj:`moveit_msgs.msg.RobotTrajectory`
        jointspace : bool
            What kind of trajectory.  Joint space points are 7x' and describe the
            angle of each arm.  Workspace points are 3x', and describe the x,y,z
            position of the end effector.  
        """
        traj = JointTrajectory()
        traj.joint_names = self.limb.joint_names()
        points = []

        for t in np.linspace(0, self.total_time, num=num_waypoints):
            point = self.trajectory_point(t, jointspace)
            points.append(point)

        # We want to make a final point at the end of the trajectory so that the
        # controller has time to converge to the final point.
        extra_point = self.trajectory_point(self.total_time, jointspace)
        extra_point.time_from_start = rospy.Duration.from_sec(
            self.total_time + 1)
        points.append(extra_point)

        #print('Generated robot trajectory:')
        #print(points)

        traj.points = points
        traj.header.frame_id = 'base'
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = traj
        return robot_traj

    def trajectory_point(self, t, jointspace):
        """
        takes a discrete point in time, and puts the position, velocity, and
        acceleration into a ROS JointTrajectoryPoint() to be put into a
        RobotTrajectory.  

        Parameters
        ----------
        t : float
        jointspace : bool
            What kind of trajectory.  Joint space points are 7x' and describe the
            angle of each arm.  Workspace points are 3x', and describe the x,y,z
            position of the end effector.  

        Returns
        -------
        :obj:`trajectory_msgs.msg.JointTrajectoryPoint`
        """
        point = JointTrajectoryPoint()
        delta_t = .01
        if jointspace:
            x_t, x_t_1, x_t_2 = None, None, None
            ik_attempts = 0
            theta_t_2 = self.get_ik(self.target_position(t - 2 * delta_t))
            theta_t_1 = self.get_ik(self.target_position(t - delta_t))
            theta_t = self.get_ik(self.target_position(t))

            # we said you shouldn't simply take a finite difference when creating
            # the path, why do you think we're doing that here?
            point.positions = theta_t
            point.velocities = (theta_t - theta_t_1) / delta_t
            point.accelerations = (
                theta_t - 2 * theta_t_1 + theta_t_2) / (2 * delta_t)
        else:
            point.positions = self.target_position(t)
            point.velocities = self.target_velocity(t)
            point.accelerations = self.target_acceleration(t)
        point.time_from_start = rospy.Duration.from_sec(t)
        return point

    def get_ik(self, x, max_ik_attempts=10):
        """
        gets ik

        Parameters
        ----------
        x : 3x' :obj:`numpy.ndarray`
            workspace position of the end effector
        max_ik_attempts : int
            number of attempts before short circuiting

        Returns
        -------
        7x' :obj:`numpy.ndarray`
            joint values to achieve the passed in workspace position
        """
        ik_attempts, theta = 0, None
        while theta is None and not rospy.is_shutdown():
            theta = self.kin.inverse_kinematics(
                position=x,
                orientation=[0, 1, 0, 0]
            )
            ik_attempts += 1
            if ik_attempts > max_ik_attempts:
                rospy.signal_shutdown(
                    'MAX IK ATTEMPTS EXCEEDED AT x(t)={}'.format(x)
                )
        return theta


class LinearPath(MotionPath):
    def __init__(self, limb, kin, total_time, current_pos, target_pos):
        """
        Remember to call the constructor of MotionPath

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        MotionPath.__init__(self, limb, kin, total_time,
                            current_pos, target_pos)

        #raise NotImplementedError

    def target_position(self, time):
        """
        Returns where the arm end effector should be at time t

        Parameters
        ----------
        time : float        

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired x,y,z position in workspace coordinates of the end effector
        """
        return (self.target_pos - self.current_pos) / self.total_time * time + self.current_pos
        #raise NotImplementedError

    def target_velocity(self, time):
        """
        Returns the arm's desired x,y,z velocity in workspace coordinates
        at time t.  You should NOT simply take a finite difference of
        self.target_position()

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired velocity in workspace coordinates of the end effector
        """
        return (self.target_pos - self.current_pos) / self.total_time

    def target_acceleration(self, time):
        """
        Returns the arm's desired x,y,z acceleration in workspace coordinates
        at time t.  You should NOT simply take a finite difference of
        self.target_velocity()

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired acceleration in workspace coordinates of the end effector
        """
        return vec(0, 0, 0)


class CircularPath(MotionPath):
    def __init__(self, limb, kin, total_time, current_pos, target_pos):
        """
        Remember to call the constructor of MotionPath

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        MotionPath.__init__(self, limb, kin, total_time,
                            current_pos, target_pos)

        self.r = 0.1 # radius of circle centered around target_pos
        self.t1 = 5 # Time allocated to reach center of circle
        self.t2 = 7 # Time allocated to reach radius away from circle
        if total_time < self.t1 + self.t2 + 1:
            raise ValueError('Not enough time given')


    def target_position(self, time):
        """
        Returns where the arm end effector should be at time t

        Parameters
        ----------
        time : float        

        Returns
        -------
        3x' :obj:`numpy.ndarray`
           desired x,y,z position in workspace coordinates of the end effector
        """
        if time < self.t1: # Linear move to center
            return (self.target_pos - self.current_pos) / self.t1 * time + self.current_pos
        elif time < self.t2: # Linear move r away from center
            return np.array([self.r, 0, 0]) / (self.t2 - self.t1) * (time - self.t1) + self.target_pos
        else: # Single circle around for time total_time - 5 - 2
            theta = 2 * np.pi * (time - self.t2) / (self.total_time - self.t2)
            return self.r * np.array([np.cos(theta), np.sin(theta), 0]) + self.target_pos

    def target_velocity(self, time):
        """
        Returns the arm's desired velocity in workspace coordinates
        at time t.  You should NOT simply take a finite difference of
        self.target_position()

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
           desired x,y,z velocity in workspace coordinates of the end effector
        """
        if time < self.t1: # Linear move to center
            return (self.target_pos - self.current_pos) / self.t1
        elif time < self.t2: # Linear move r away from center
            return (self.target_pos + np.array([self.r, 0, 0])) / (self.t2 - self.t1)
        else: # Single circle around for time total_time - 5 - 2
            theta = 2 * np.pi * (time - self.t2) / (self.total_time - self.t2)
            return 2 * np.pi / (self.total_time - self.t2) * self.r * np.array([-1*np.sin(theta), np.cos(theta), 0])

    def target_acceleration(self, time):
        """
        Returns the arm's desired x,y,z acceleration in workspace coordinates
        at time t.  You should NOT simply take a finite difference of
        self.target_velocity()

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
           desired acceleration in workspace coordinates of the end effector
        """
        if time < self.t1: # Linear move to center
            return np.array([0, 0, 0])
        elif time < self.t2: # Linear move r away from center
            return np.array([0, 0, 0])
        else: # Single circle around for time total_time - 5 - 2
            theta = 2 * np.pi * (time - self.t2) / (self.total_time - self.t2)
            return (2 * np.pi / (self.total_time - self.t2))**2 * self.r * np.array([-1*np.cos(theta), -1*np.cos(theta), 0])


class MultiplePaths(MotionPath):
    """
    Remember to call the constructor of MotionPath

    You can implement multiple paths a couple ways.  The way I chose when I took
    the class was to create several different paths and pass those into the 
    MultiplePaths object, which would determine when to go onto the next path.
    """

    def __init__(self, limb, kin, total_time, current_pos, positions):
        MotionPath.__init__(self, limb, kin, total_time,
                            current_pos, positions[0])
        self.pos = [current_pos]
        self.pos = self.pos+positions
        '''
        times = np.linspace(0, total_time, 100)
        pos = np.array([0, 0, 0])
        for t in times:
            pos = np.vstack((pos, self.target_position(t)))
        pos = pos[1:-1, :]
        print(pos)
        plt.figure()
        plt.grid(True)
        plt.scatter(pos[:, 0], pos[:, 1], color='r', marker='.')
        plt.axis('equal')
        plt.show()
        '''


    def target_position(self, time):
        """
        Returns where the arm end effector should be at time t

        Parameters
        ----------
        time : float        

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired position in workspace coordinates of the end effector
        """
        t_segment = float(self.total_time) / float(len(self.pos))
        segment = min(int(time / t_segment), len(self.pos)-1)
        pos_start = self.pos[segment]
        if segment < len(self.pos) - 1:
            x = segment+1
            pos_end = self.pos[(segment+1)]
        else:
            x = 1
            pos_end = self.pos[1]
        t = time - segment * t_segment
        #print('total time:{}, number of positions:{}, t_segment:{}'.format(self.total_time, len(self.pos), t_segment))
        #print('time:{}, t:{}, segment:{}, target:{}'.format(time, t, segment, pos_end[1]))
        return (pos_end - pos_start) / t_segment * t + pos_start
        
    def target_velocity(self, time):
        """
        Returns the arm's desired velocity in workspace coordinates
        at time t

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired velocity in workspace coordinates of the end effector
        """

        t_segment = self.total_time / len(self.pos)
        segment = min(int(time / t_segment), len(self.pos)-1)
        pos_start = self.pos[segment]
        if segment < len(self.pos) - 1:
            pos_end = self.pos[(segment+1)]
        else:
            pos_end = self.pos[1]
        return (pos_end - pos_start) / t_segment

    def target_acceleration(self, time):
        """
        Returns the arm's desired acceleration in workspace coordinates
        at time t

        Parameters
        ----------
        time : float

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            desired acceleration in workspace coordinates of the end effector
        """
        return vec(0, 0, 0)
