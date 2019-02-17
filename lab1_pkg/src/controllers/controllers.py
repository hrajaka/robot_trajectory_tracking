#!/usr/bin/env python

"""
Starter script for lab1.
Author: Chris Correa, Valmik Prabhu
"""

# Python imports
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt

# Lab imports
from utils.utils import *

# ROS imports
try:
    import tf
    import rospy
    import baxter_interface
    import intera_interface
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.msg import RobotTrajectory
except:
    pass

from baxter_pykdl.baxter_pykdl import baxter_kinematics

NUM_JOINTS = 7

class Controller:

    def __init__(self, limb, kin):
        """
        Constructor for the superclass. All subclasses should call the superconstructor

        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`baxter_pykdl.baxter_kinematics` or :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        """

        # Run the shutdown function when the ros node is shutdown
        rospy.on_shutdown(self.shutdown)
        self._limb = limb
        self._kin = kin
        self.controller_name = None

    def step_control(self, target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws):
        """
        makes a call to the robot to move according to it's current position and the desired position
        according to the input path and the current time. Each Controller below extends this
        class, and implements this accordingly.

        Parameters
        ----------
        target_position : 7x' or 6x' :obj:`numpy.ndarray`
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray`
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray`
            desired accelerations
        """
        pass

    def interpolate_path(self, path, t, current_index = 0):
        """
        interpolates over a :obj:`moveit_msgs.msg.RobotTrajectory` to produce desired
        positions, velocities, and accelerations at a specified time

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        t : float
            the time from start
        current_index : int
            waypoint index from which to start search

        Returns
        -------
        target_position : 7x' or 6x' :obj:`numpy.ndarray`
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray`
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray`
            desired accelerations
        current_index : int
            waypoint index at which search was terminated
        """

        # a very small number (should be much smaller than rate)
        epsilon = 0.0001

        max_index = len(path.joint_trajectory.points)-1

        # If the time at current index is greater than the current time,
        # start looking from the beginning
        if (path.joint_trajectory.points[current_index].time_from_start.to_sec() > t):
            current_index = 0

        # Iterate forwards so that you're using the latest time
        while (
            not rospy.is_shutdown() and \
            current_index < max_index and \
            path.joint_trajectory.points[current_index+1].time_from_start.to_sec() < t+epsilon
        ):
            current_index = current_index+1

        # Perform the interpolation
        if current_index < max_index:
            time_low = path.joint_trajectory.points[current_index].time_from_start.to_sec()
            time_high = path.joint_trajectory.points[current_index+1].time_from_start.to_sec()

            target_position_low = np.array(
                path.joint_trajectory.points[current_index].positions
            )
            target_velocity_low = np.array(
                path.joint_trajectory.points[current_index].velocities
            )
            target_acceleration_low = np.array(
                path.joint_trajectory.points[current_index].accelerations
            )

            target_position_high = np.array(
                path.joint_trajectory.points[current_index+1].positions
            )
            target_velocity_high = np.array(
                path.joint_trajectory.points[current_index+1].velocities
            )
            target_acceleration_high = np.array(
                path.joint_trajectory.points[current_index+1].accelerations
            )

            target_position = target_position_low + \
                (t - time_low)/(time_high - time_low)*(target_position_high - target_position_low)
            target_velocity = target_velocity_low + \
                (t - time_low)/(time_high - time_low)*(target_velocity_high - target_velocity_low)
            target_acceleration = target_acceleration_low + \
                (t - time_low)/(time_high - time_low)*(target_acceleration_high - target_acceleration_low)

        # If you're at the last waypoint, no interpolation is needed
        else:
            target_position = np.array(path.joint_trajectory.points[current_index].positions)
            target_velocity = np.array(path.joint_trajectory.points[current_index].velocities)
            target_acceleration = np.array(path.joint_trajectory.points[current_index].velocities)

        return (target_position, target_velocity, target_acceleration, current_index)


    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety
        """
        rospy.loginfo("Stopping Controller")

        # Set velocities to zero
        self.stop_moving()
        rospy.sleep(0.1)

    def stop_moving(self):
        """
        Set robot joint velocities to zero
        """
        zero_vel_dict = joint_array_to_dict(np.zeros(NUM_JOINTS), self._limb)
        self._limb.set_joint_velocities(zero_vel_dict)



    def plot_results(
        self,
        times,
        actual_positions,
        actual_velocities,
        target_positions,
        target_velocities,
        errors,
        d_errors
    ):
        """
        Plots results.
        If the path is in joint space, it will plot both workspace and jointspace plots.
        Otherwise it'll plot only workspace

        Inputs:
        times : nx' :obj:`numpy.ndarray`
        actual_positions : nx7 or nx6 :obj:`numpy.ndarray`
            actual joint positions for each time in times
        actual_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            actual joint velocities for each time in times
        target_positions: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace positions for each time in times
        target_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace velocities for each time in times
        """
        # directory='/home/cc/ee106b/sp19/class/ee106b-aai/Documents/'
        directory = '/home/cc/ee106b/sp19/class/ee106b-aai/ros_workspaces/lab1_ws/src/robot_trajectory_tracking'
        #print('ACTUAL_POSITIONS: ', actual_positions[0])
        #print('ERRORS: ', errors[0])

        # Make everything an ndarray
        times = np.array(times)
        actual_positions = np.array(actual_positions)
        actual_velocities = np.array(actual_velocities)
        target_positions = np.array(target_positions)
        target_velocities = np.array(target_velocities)
        errors = np.array(errors)
        d_errors = np.array(d_errors)

        # Find the actual workspace positions and velocities
        actual_workspace_positions = np.zeros((len(times), 3))
        actual_workspace_velocities = np.zeros((len(times), 3))

        for i in range(len(times)):
            positions_dict = joint_array_to_dict(actual_positions[i], self._limb)
            actual_workspace_positions[i] = \
                self._kin.forward_position_kinematics(joint_values=positions_dict)[:3]
            actual_workspace_velocities[i] = \
                self._kin.jacobian(joint_values=positions_dict)[:3].dot(actual_velocities[i])

        # check if joint space
        if target_positions.shape[1] > 3:
            # it's joint space

            target_workspace_positions = np.zeros((len(times), 3))
            target_workspace_velocities = np.zeros((len(times), 3))

            for i in range(len(times)):
                positions_dict = joint_array_to_dict(target_positions[i], self._limb)
                target_workspace_positions[i] = \
                    self._kin.forward_position_kinematics(joint_values=positions_dict)[:3]
                target_workspace_velocities[i] = \
                    self._kin.jacobian(joint_values=positions_dict)[:3].dot(target_velocities[i])

            # Plot joint space
            plt.figure()
            # print len(times), actual_positions.shape()
            joint_num = len(self._limb.joint_names())
            for joint in range(joint_num):
                '''
                plt.subplot(joint_num,2,2*joint+1)
                plt.plot(times, actual_positions[:,joint], label='Actual')
                plt.plot(times, target_positions[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel(str(joint) + " pos")
                '''

                plt.subplot(joint_num,1,joint+1)
                plt.plot(times, actual_velocities[:,joint], label='Actual')
                plt.plot(times, target_velocities[:,joint], label='Desired')
                #plt.plot(times, errors[:,joint], label='e')
                #plt.plot(times, d_errors[:,joint], label='de')
                #plt.xlabel("Time (t)")
                plt.ylabel(str(joint) + " vel")

            print "Close the plot window to continue"
            #plt.tight_layout()
            plt.savefig(directory+self.controller_name+'_jsplot.png')
            # plt.show()


            # error plots
            plt.figure()
            plt.grid(True)
            plt.axvline(color='k')
            plt.axhline(color='k')
            plt.xlabel("Time (t)")
            plt.ylabel('Velocity Errors')
            for joint in range(joint_num):
                plt.plot(times, errors[:,joint], label='joint '+str(joint))
                plt.xlabel("Time (t)")
                plt.ylabel('Velocity Errors')
            plt.legend()
            plt.tight_layout()
            plt.savefig(directory+self.controller_name+'_err_jsplot.png')
            # plt.show()
            print "Close the plot window to continue"


        else:
            # it's workspace
            target_workspace_positions = target_positions
            target_workspace_velocities = target_velocities

        plt.figure()
        workspace_joints = ('X', 'Y', 'Z')
        joint_num = len(workspace_joints)
        for joint in range(joint_num):
            plt.subplot(joint_num,2,2*joint+1)
            plt.plot(times, actual_workspace_positions[:,joint], label='Actual')
            plt.plot(times, target_workspace_positions[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Position Error")

            plt.subplot(joint_num,2,2*joint+2)
            plt.plot(times, actual_velocities[:,joint], label='Actual')
            plt.plot(times, target_velocities[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Velocity Error")

        print "Close the plot window to continue"
        plt.tight_layout()
        plt.savefig(directory+self.controller_name+'_wsplot.png')
        plt.show()




    def execute_path(self, path, rate=200, timeout=None, log=False):
        """
        takes in a path and moves the baxter in order to follow the path.

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For plotting
        if log:
            times = list()
            actual_positions = list()
            actual_velocities = list()
            target_positions = list()
            target_velocities = list()
            errors_js = list()
            d_errors_js = list()
            errors_ws = list()
            d_errors_ws = list()

        # For interpolation
        max_index = len(path.joint_trajectory.points)-1
        current_index = 0

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)


        prev_t = (rospy.Time.now() - start_t).to_sec()
        prev_error_js = vec(0,0,0,0,0,0,0) # Initially set previous error to zero
        prev_error_ws = vec(0,0,0,0,0,0) # Initially set previous error to zero
        r.sleep()
        while not rospy.is_shutdown():
            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            # If the controller has timed out, stop moving and return false
            if timeout is not None and t >= timeout:
                # Set velocities to zero
                self.stop_moving()
                return False


            current_position_js = get_joint_positions(self._limb)
            current_velocity_js = get_joint_velocities(self._limb)
            jacobian = self._kin.jacobian()
            current_velocity_ws = np.matmul(jacobian, current_velocity_js)

            # Get the desired position, velocity, and effort
            (
                target_position,
                target_velocity,
                target_acceleration,
                current_index
            ) = self.interpolate_path(path, t, current_index)

            #print('t, prev_t', t, prev_t)


            if len(target_velocity) == 3:
                # then it's in workspace

                error_js = 0 # we do not care about it
                d_error_js = 0 # we do not care about it
                target_velocity = np.array([target_velocity[0], target_velocity[1], target_velocity[2], 0, 0, 0]).reshape((6,1))
                current_velocity_ws = current_velocity_ws.reshape(6,1)
                
                prev_error_ws = prev_error_ws.reshape(6,1)

                error_ws = target_velocity - current_velocity_ws
                d_error_ws = (error_ws - prev_error_ws) / (t - prev_t)

            
            else:
                # then it's in jointspace

                error_js = target_velocity - current_velocity_js
                d_error_js = (error_js - prev_error_js) / (t - prev_t)

                target_velocity_ws = np.matmul(jacobian, target_velocity)
                error_ws = target_velocity_ws - current_velocity_ws
                d_error_ws = (error_ws - prev_error_ws) / (t - prev_t)


            # For plotting
            if log:
                times.append(t)
                actual_positions.append(current_position_js)
                actual_velocities.append(current_velocity_js)
                target_positions.append(target_position)
                target_velocities.append(target_velocity)
                errors_js.append(error_js)
                d_errors_js.append(d_error_js)

            # Run controller
            self.step_control(target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws)

            # Sleep for a bit (to let robot move)
            r.sleep()

            if current_index >= max_index:
                self.stop_moving()
                break

            prev_t = t
            prev_error_js = error_js
            prev_error_ws = error_ws

        if log:

            #try:
            self.plot_results(
                times,
                actual_positions,
                actual_velocities,
                target_positions,
                target_velocities,
                errors_js,
                d_errors_js
            )
            #except:
            #    pass
        return True

    def follow_ar_tag(self, tag, rate=200, timeout=None, log=False):
        """
        takes in an AR tag number and follows it with the baxter's arm.  You
        should look at execute_path() for inspiration on how to write this.

        Parameters
        ----------
        tag : int
            which AR tag to use
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """
        raise NotImplementedError

class FeedforwardJointVelocityController(Controller):
    def __init__(self, limb, kin):
        Controller.__init__(self, limb, kin)
        self.controller_name = 'ff_js_vel'

    def step_control(self, target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws):
        """
        Parameters
        ----------
        target_position: 7x' ndarray of desired positions
        target_velocity: 7x' ndarray of desired velocities
        target_acceleration: 7x' ndarray of desired accelerations
        """
        self._limb.set_joint_velocities(joint_array_to_dict(target_velocity, self._limb))

class PDWorkspaceVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  The difference between this controller and the
    PDJointVelocityController is that this controller compares the baxter's current WORKSPACE position and
    velocity desired WORKSPACE position and velocity to come up with a WORKSPACE velocity command to be sent
    to the baxter.  Then this controller should convert that WORKSPACE velocity command into a joint velocity
    command and sends that to the baxter.  Notice the shape of Kp and Kv
    """
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 6x' :obj:`numpy.ndarray`
        Kv : 6x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.controller_name = 'pd_ws_vel'

    def step_control(self, target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws):
        """
        makes a call to the robot to move according to it's current position and the desired position
        according to the input path and the current time. Each Controller below extends this
        class, and implements this accordingly. This method should call
        self._kin.forward_psition_kinematics() and self._kin.forward_velocity_kinematics() to get
        the current workspace position and velocity and self._limb.set_joint_velocities() to set
        the joint velocity to something.  you may have to look at
        http://docs.ros.org/diamondback/api/kdl/html/python/geometric_primitives.html to convert the
        output of forward_velocity_kinematics() to a numpy array.  You may find joint_array_to_dict()
        in utils.py useful

        MAKE SURE TO CONVERT QUATERNIONS TO EULER IN forward_position_kinematics().
        you can use tf.transformations.euler_from_quaternion()

        your target orientation should be (0,0,0) in euler angles and (0,1,0,0) as a quaternion.

        Parameters
        ----------
        target_position: 6x' ndarray of desired positions
        target_velocity: 6x' ndarray of desired velocities
        target_acceleration: 6x' ndarray of desired accelerations
        """

        v_ws = target_velocity + (np.matmul(self.Kp, error_ws) + np.matmul(self.Kv, d_error_ws))


        print('v_ws', v_ws)

        J_pseudoinv =  self._kin.jacobian_pseudo_inverse()

        v_js = np.matmul(J_pseudoinv, v_ws)

        self._limb.set_joint_velocities(joint_array_to_dict(v_js, self._limb))

        # raise NotImplementedError

class PDJointVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  The difference between this controller and the
    PDJointVelocityController is that this controller turns the desired workspace position and velocity
    into desired JOINT position and velocity.  Then it compares the difference between the baxter's
    current JOINT position and velocity and desired JOINT position and velocity to come up with a
    joint velocity command and sends that to the baxter.  notice the shape of Kp and Kv
    """
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.controller_name = 'pd_js_vel'

    def step_control(self, target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws):
        """
        makes a call to the robot to move according to it's current position and the desired position
        according to the input path and the current time. Each Controller below extends this
        class, and implements this accordingly. This method should call
        self._limb.joint_angle and self._limb.joint_velocity to get the current joint position and velocity
        and self._limb.set_joint_velocities() to set the joint velocity to something.  You may find
        joint_array_to_dict() in utils.py useful

        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """

        print('target_velocity.shape', target_velocity.shape)


        v = target_velocity + (np.matmul(self.Kp, error_js) + np.matmul(self.Kv, d_error_js))

        print('v.shape', v.shape)

        self._limb.set_joint_velocities(joint_array_to_dict(v, self._limb))
        #raise NotImplementedError

class PDJointTorqueController(Controller):
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.controller_name = 'pd_js_torque'

    def step_control(self, target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws):
        """
        makes a call to the robot to move according to it's current position and the desired position
        according to the input path and the current time. Each Controller below extends this
        class, and implements this accordingly. This method should call
        self._limb.joint_angle and self._limb.joint_velocity to get the current joint position and velocity
        and self._limb.set_joint_velocities() to set the joint velocity to something.  You may find
        joint_array_to_dict() in utils.py useful

        Look in section 4.5 of MLS.

        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        raise NotImplementedError









######################
# GRAD STUDENTS ONLY #
######################

class WorkspaceImpedanceController(Controller):
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 6x' :obj:`numpy.ndarray`
        Kv : 6x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.controller_name = 'pd_ws_imp'

    def step_control(self, target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws):
        """

        Parameters
        ----------
        target_position: 7x' ndarray of desired positions
        target_velocity: 7x' ndarray of desired velocities
        target_acceleration: 7x' ndarray of desired accelerations
        """
        raise NotImplementedError

class JointspaceImpedanceController(Controller):
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.controller_name = 'pd_js_imp'

    def step_control(self, target_position, target_velocity, target_acceleration, error_js, d_error_js, error_ws, d_error_ws, current_position_js, current_velocity_js, current_velocity_ws):
        """
        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        raise NotImplementedError
