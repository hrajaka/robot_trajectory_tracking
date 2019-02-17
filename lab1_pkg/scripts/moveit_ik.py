#!/usr/bin/env python
import rospy
import moveit_commander
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.srv import GetPositionIKRequest
from moveit_msgs.srv import GetPositionIKResponse
import geometry_msgs.msg
from utils.utils import *

"""
    Important info: To use this module, you need to have ROS installed, and
    the moveit config, such as yumi_moveit_config downloaded.  Before using 
    this module, you must start the move_group using the following commands
    (in their own terminals):
    roscore
    roslaunch yumi_moveit_config real_demo.launch
"""
class MoveitIK(object):
    def __init__(self, group, ik_timeout=1.0, ik_attempts=0,
                 avoid_collisions=False):
        """
        A class to do IK calls thru the MoveIt!'s /compute_ik service.

        Parameters
        ----------
        group : string
            MoveIt! group name
        ik_timeout : float
            default timeout for IK
        ik_attempts : int
            default number of attempts
        avoid_collisions : bool
            if to ask for IKs that take into account collisions
        """
        rospy.loginfo("Initalizing GetIK...")
        self.group_name = group
        self.ik_timeout = ik_timeout
        self.ik_attempts = ik_attempts
        self.avoid_collisions = avoid_collisions
        rospy.loginfo("Computing IKs for group: " + self.group_name)
        rospy.loginfo("With IK timeout: " + str(self.ik_timeout))
        rospy.loginfo("And IK attempts: " + str(self.ik_attempts))
        rospy.loginfo("Setting avoid collisions to: " +
                      str(self.avoid_collisions))
        self.ik_srv = rospy.ServiceProxy('/compute_ik',
                                         GetPositionIK)
        rospy.loginfo("Waiting for /compute_ik service...")
        self.ik_srv.wait_for_service()
        rospy.loginfo("Connected!")

    def get_ik(self, 
        pose_stamped,
        group=None,
        ik_timeout=None,
        ik_attempts=None,
        avoid_collisions=None
    ):
        """
        Do an IK call to pose_stamped pose. geometry_msgs/PoseStamped 
        pose_stamped: The 3D pose (with header.frame_id) to which 
        compute the IK.
        
        Parameters
        ----------
        pose_stamped : :obj:`geometry_msgs.msg.PoseStamped`
            goal pose
        group : string
            The MoveIt! group.
        ik_timeout : float
            The timeout for the IK call.
        ik_attemps : int
            The maximum # of attemps for the IK.
        avoid_collisions : bool
            If to compute collision aware IK.

        Returns
        -------
        moveit_msgs.srv.GetPositionIKResponse
            Response from /compute_ik service
        """
        if group is None:
            group = self.group_name
        if ik_timeout is None:
            ik_timeout = self.ik_timeout
        if ik_attempts is None:
            ik_attempts = self.ik_attempts
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions
        req = GetPositionIKRequest()
        req.ik_request.group_name = group
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.timeout = rospy.Duration(ik_timeout)
        req.ik_request.attempts = ik_attempts
        req.ik_request.avoid_collisions = avoid_collisions

        try:
            resp = self.ik_srv.call(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            resp = GetPositionIKResponse()
            resp.error_code = 99999  # Failure
            return resp

if __name__ == "__main__":
    group = "right_arm"
    ik_srv = MoveitIK(group)

    pos = [.5, 0, .5]
    quat = [0,0,1,0]
    pose = create_pose_stamped_from_pos_quat(pos, quat)
    res = ik_srv.get_ik(pose, avoid_collisions=True)
    print res

    print "ending\n\n\n\n"
