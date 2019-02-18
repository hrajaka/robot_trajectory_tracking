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
        self.total_time = float(total_time)
        self.current_pos = current_pos
        self.target_pos = target_pos

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

    def plot_me(self):

        times = np.linspace(0, self.total_time, 100)
        pos = np.array([0, 0, 0])
        vel = np.array([0, 0, 0])
        acc = np.array([0, 0, 0])
        for t in times:
            pos = np.vstack((pos, self.target_position(t)))
            vel = np.vstack((vel, self.target_velocity(t)))
            acc = np.vstack((acc, self.target_acceleration(t)))

        pos = pos[1:, :]
        vel = vel[1:, :]
        acc = acc[1:, :]

        plt.figure()
        plt.grid(True)
        plt.title('Path ({})'.format(self.total_time))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(pos[:, 0], pos[:, 1], color='r', marker='.')
        plt.axis('equal')
        plt.axvline(color='k')
        plt.axhline(color='k')
        plt.show()

        plt.figure()
        plt.grid(True)
        plt.title('Path')
        plt.xlabel('t')
        plt.plot(times, pos[:, 0], color='r', marker='.', label='x')
        plt.plot(times, pos[:, 1], color='b', marker='.', label='y')
        plt.plot(times, pos[:, 2], color='g', marker='.', label='z')
        plt.legend()
        plt.show()

        plt.figure()
        plt.grid(True)
        plt.title('Path')
        plt.xlabel('t')
        plt.plot(times, vel[:, 0], color='r', marker='.', label='vx')
        plt.plot(times, vel[:, 1], color='b', marker='.', label='vy')
        plt.plot(times, vel[:, 2], color='g', marker='.', label='vz')
        plt.axvline(color='k')
        plt.axhline(color='k')
        plt.legend()
        plt.show()

        plt.figure()
        plt.grid(True)
        plt.xlabel('t')
        plt.plot(times, acc[:, 0], color='r', marker='.', label='ax')
        plt.plot(times, acc[:, 1], color='b', marker='.', label='ay')
        plt.plot(times, acc[:, 2], color='g', marker='.', label='az')        
        plt.axvline(color='k')
        plt.axhline(color='k')
        plt.legend()
        plt.show()
    


class LinearPath(MotionPath):
    def __init__(self, limb, kin, total_time, current_pos, target_pos):
        """
        Remember to call the constructor of MotionPath

        Parameters
        ----------
        """
        MotionPath.__init__(self, limb, kin, total_time, current_pos, target_pos)


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

        cruise_speed = (self.target_pos - self.current_pos) / self.total_time * 10./9

        if time <= (1./10)*self.total_time:
            return self.current_pos + 10*cruise_speed*0.5*time**2/self.total_time
        elif time <= (9./10)*self.total_time:
            return self.current_pos + 10*cruise_speed*0.5*(self.total_time/10)**2/self.total_time + cruise_speed*(time-self.total_time/10)
        else:
            return self.current_pos + 10*cruise_speed*0.5*(self.total_time/10)**2/self.total_time + cruise_speed*(8*self.total_time/10) + 10*cruise_speed*(time-0.5*time*time/self.total_time - 9 * self.total_time/10 + 0.5*(9*self.total_time/10)**2/self.total_time) 



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
        cruise_speed = (self.target_pos - self.current_pos) / self.total_time * 10./9

        if time <= (1./10)*self.total_time:
            return 10*time/self.total_time*cruise_speed
        elif time <= (9./10)*self.total_time:
            return cruise_speed
        else:
            return (1 - 10*(time - (9./10)*self.total_time)/self.total_time )*cruise_speed


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
        cruise_speed = (self.target_pos - self.current_pos) / self.total_time * 10./9

        if time <= (1./10)*self.total_time:
            return cruise_speed / self.total_time
        elif time <= (9./10)*self.total_time:
            return np.array([0,0,0])
        else:
            return -1*cruise_speed / self.total_time


class CircularPath(MotionPath):
    def __init__(self, limb, kin, total_time, current_pos, target_pos):
        """
        Remember to call the constructor of MotionPath

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """


        self.r = 0.1 # radius of circle centered around target_pos
        self.t1 = 5 # Time allocated to reach center of circle
        self.t2 = self.t1 + 5 # Time allocated to reach radius away from circle
        if total_time < self.t1 + self.t2 + 5:
            raise ValueError('Not enough time given')

        self.L1 = LinearPath(limb, kin, self.t1, current_pos, target_pos)
        end2 = target_pos + np.array([self.r, 0, 0])
        self.L2 = LinearPath(limb, kin, self.t2 - self.t1, target_pos, end2)
        MotionPath.__init__(self, limb, kin, total_time, current_pos, target_pos)


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
            # return (self.target_pos - self.current_pos) / self.t1 * time + self.current_pos
            return self.L1.target_position(time)
            '''
            target = self.target_pos
            current = self.current_pos
            cruise_speed = (current - target) / self.t1 * 10./9

            t = time - 0 
            if t <= (1./10)*self.t1:                
                return self.current_pos + 10*cruise_speed*0.5*t**2/self.t1


            elif t <= (9./10)*self.t1:
                return self.current_pos + 10*cruise_speed*0.5*(self.t1/10)**2/self.t1 + cruise_speed*(t-self.t1/10)
            else:
                return self.current_pos + 10*cruise_speed*0.5*(self.t1/10)**2/self.t1 + cruise_speed*(8*self.t1/10) + 10*cruise_speed*(t-0.5*t*t/self.t1 - 9 * self.t1/10 + 0.5*(9*self.t1/10)**2/self.t1)
            '''

        elif time < self.t2: # Linear move r away from center
            return self.L2.target_position(time - self.t1)
            # return np.array([self.r, 0, 0]) / (self.t2 - self.t1) * (time - self.t1) + self.target_pos
            '''
            target = self.target_pos + np.array([self.r, 0, 0])
            current = self.target_pos
            cruise_speed = (current - target) / self.t1 * 10./9

            t = time - self.t1 
            if t <= (1./10)*self.t2:                
                return self.current_pos + 10*cruise_speed*0.5*t**2/self.t2


            elif t <= (9./10)*self.t2:
                return self.current_pos + 10*cruise_speed*0.5*(self.t2/10)**2/self.t2 + cruise_speed*(t-self.t2/10)
            else:
                return self.current_pos + 10*cruise_speed*0.5*(self.t2/10)**2/self.t2 + cruise_speed*(8*self.t2/10) + 10*cruise_speed*(t-0.5*t*t/self.t2 - 9 * self.t2/10 + 0.5*(9*self.t2/10)**2/self.t2)
            '''
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
            return self.L1.target_velocity(time)
            '''
            # return (self.target_pos - self.current_pos) / self.t1

            target = self.target_pos
            current = self.current_pos
            cruise_speed = (current - target) / self.t1 * 10./9

            t = time - 0 
            if t <= (1./10)*self.t1:                
                return 10*t/self.t1*cruise_speed
            elif t <= (9./10)*self.t1:
                return cruise_speed
            else:
                return (1 - 10*(t - (9./10)*self.t1)/self.t1 )*cruise_speed
            '''



        elif time < self.t2: # Linear move r away from center
            return self.L2.target_velocity(time - self.t1)
            '''
            # return (self.target_pos + np.array([self.r, 0, 0])) / (self.t2 - self.t1)

            target = self.target_pos + np.array([self.r, 0, 0])
            current = self.target_pos
            cruise_speed = (current - target) / self.t1 * 10./9

            t = time - self.t1 
            if t <= (1./10)*self.t2:                
                return 10*t/self.t2*cruise_speed
            elif t <= (9./10)*self.t2:
                return cruise_speed
            else:
                return (1 - 10*(t - (9./10)*self.t2)/self.t2 )*cruise_speed
            '''



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
            return self.L1.target_acceleration(time)
            '''
            # return np.array([0, 0, 0])

            target = self.target_pos
            current = self.current_pos
            cruise_speed = (current - target) / self.t1 * 10./9

            t = time - 0 
            if t <= (1./10)*self.t1:
                return cruise_speed / self.t1
            elif t <= (9./10)*self.t1:
                return np.array([0,0,0])
            else:
                return -1*cruise_speed / self.t1
            '''



        elif time < self.t2: # Linear move r away from center
            return self.L2.target_acceleration(time - self.t1)
            '''
            # return np.array([0, 0, 0])

            target = self.target_pos + np.array([self.r, 0, 0])
            current = self.target_pos
            cruise_speed = (current - target) / self.t2 * 10./9

            t = time - self.t1 

            if t <= (1./10)*self.t2:
                return cruise_speed / self.t2
            elif t <= (9./10)*self.t2:
                return np.array([0,0,0])
            else:
                return -1*cruise_speed / self.t2
            '''

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

    def __init__(self, limb, kin, linear_paths):

        self.linear_paths = linear_paths
        self.start_times = [0]
        for l in self.linear_paths[:-1]:
            self.start_times.append(self.start_times[-1]+l.total_time)

        self.total_time = self.start_times[-1] + self.linear_paths[-1].total_time


        MotionPath.__init__(self, limb, kin, self.total_time, self.linear_paths[0].current_pos, self.linear_paths[-1].target_pos)


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

        i = 0 #i is the 'section' we're in
        while i < len(self.start_times)-1 and time > self.start_times[i+1]  :
            i = i+1

        return self.linear_paths[i].target_position(time-self.start_times[i])


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

        i = 0 #i is the 'section' we're in
        while i < len(self.start_times)-1 and time > self.start_times[i+1]  :
            i = i+1


        return self.linear_paths[i].target_velocity(time-self.start_times[i])


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

        i = 0 #i is the 'section' we're in
        while i < len(self.start_times)-1 and time > self.start_times[i+1]  :
            i = i+1


        return self.linear_paths[i].target_acceleration(time-self.start_times[i])

