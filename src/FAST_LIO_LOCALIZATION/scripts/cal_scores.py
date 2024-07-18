#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import _thread as thread
import time

import open3d as o3d
import rospy
import ros_numpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import numpy as np
import tf
import tf.transformations
from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PoseWithCovarianceStamped
from threading import Lock

global_map = None
initialized = False
T_map_to_odom = np.eye(4)
cur_odom = None
cur_scan = None
max_score = 0
cloud_topic = None
pub_final_name = None
done_pub = None

g_pose = PoseWithCovarianceStamped()
mtx = Lock()

g_name = String()
mtx_name = Lock()



def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
    )


def msg_to_array(pc_msg):
    pc_array = ros_numpy.numpify(pc_msg)
    pc = np.zeros([len(pc_array), 3])
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    return pc


def registration_at_scale(pc_scan, pc_map, initial, scale):
    result_icp = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale), voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
        1.0 * scale, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )

    return result_icp.transformation, result_icp.fitness


def inverse_se3(trans):
    trans_inverse = np.eye(4)
    # R
    trans_inverse[:3, :3] = trans[:3, :3].T
    # t
    trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inverse


def publish_point_cloud(publisher, header, pc):
    data = np.zeros(len(pc), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    data['x'] = pc[:, 0]
    data['y'] = pc[:, 1]
    data['z'] = pc[:, 2]
    if pc.shape[1] == 4:
        data['intensity'] = pc[:, 3]
    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header = header
    publisher.publish(msg)


def crop_global_map_in_FOV(global_map, pose_estimation, cur_odom):
    # 当前scan原点的位姿
    T_odom_to_base_link = pose_to_mat(cur_odom)
    T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
    T_base_link_to_map = inverse_se3(T_map_to_base_link)

    # 把地图转换到lidar系下
    global_map_in_map = np.array(global_map.points)
    global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
    global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T

    # 将视角内的地图点提取出来
    if FOV > 3.14:
        # 环状lidar 仅过滤距离
        indices = np.where(
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    else:
        # 非环状lidar 保前视范围
        # FOV_FAR>x>0 且角度小于FOV
        indices = np.where(
            (global_map_in_base_link[:, 0] > 0) &
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    global_map_in_FOV = o3d.geometry.PointCloud()
    global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))

    # 发布fov内点云
    header = cur_odom.header
    header.frame_id = 'map'
    publish_point_cloud(pub_submap, header, np.array(global_map_in_FOV.points)[::10])

    return global_map_in_FOV




def global_localization2(pose_estimation):
    global global_map, cur_scan, cur_odom
    # 用icp配准
    rospy.loginfo('Global localization by scan-to-map matching......')

    scan_tobe_mapped = copy.copy(cur_scan)

    tic = time.time()

    global_map_in_FOV = crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)

    # 粗配准
    transformation, _ = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)

    # 精配准
    transformation, fitness = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation,
                                                    scale=1)
    toc = time.time()
    rospy.loginfo('Time: {}'.format(toc - tic))
    rospy.loginfo('score: {}'.format(fitness))
    
    return True, fitness



def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except:
        # for opend3d 0.7 or lower
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down


def initialize_global_map(pc_msg):
    global global_map

    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
    rospy.loginfo('Global map received.')


def cb_save_cur_odom(odom_msg):
    global cur_odom
    cur_odom = odom_msg


def cb_save_cur_scan(pc_msg):
    global cur_scan
    # 注意这里fastlio直接将scan转到odom系下了 不是lidar局部系
    pc_msg.header.frame_id = 'camera_init'
    pc_msg.header.stamp = rospy.Time().now()
    pub_pc_in_map.publish(pc_msg)

    # 转换为pcd
    # fastlio给的field有问题 处理一下
    pc_msg.fields = [pc_msg.fields[0], pc_msg.fields[1], pc_msg.fields[2],
                     pc_msg.fields[4], pc_msg.fields[5], pc_msg.fields[6],
                     pc_msg.fields[3], pc_msg.fields[7]]
    pc = msg_to_array(pc_msg)

    cur_scan = o3d.geometry.PointCloud()
    cur_scan.points = o3d.utility.Vector3dVector(pc[:, :3])


res = []
names = []
num = -1

def num_callback(msg):
    global num
    num = msg.data
    rospy.loginfo("Current num : %d",num)

def name_callback(msg):
    global g_name
    with mtx_name:
        g_name = msg

def pose_callback(msg):
    global g_pose
    with mtx:
        g_pose = msg
   
def cloud_callback(msg):
    rospy.loginfo("Received a point cloud")

    cur_pose = PoseWithCovarianceStamped()
    with mtx:
        cur_pose = g_pose

    cur_name = String()
    with mtx_name:
        cur_name = g_name

    rospy.loginfo("Current PoseWithCovarianceStamped: [%.2f, %.2f, %.2f], [%.2f, %.2f, %.2f, %.2f]",
        cur_pose.pose.pose.position.x, cur_pose.pose.pose.position.y, cur_pose.pose.pose.position.z,
        cur_pose.pose.pose.orientation.x, cur_pose.pose.pose.orientation.y, cur_pose.pose.pose.orientation.z, cur_pose.pose.pose.orientation.w)
    
    initialize_global_map(msg)
    initial_pose = pose_to_mat(cur_pose)

    init = False
    while not init:
        if cur_scan:
            init, score = global_localization2(initial_pose)
            res.append(score)
            names.append(cur_name.data)
            rospy.logwarn("cur_name = %s", cur_name.data)

        else:
            rospy.logwarn('First scan not received!!!!!')

    
    done_msg = Bool()
    done_msg.data = True
    done_pub.publish(done_msg)
    rospy.loginfo("done!!!!!!!!!")

    
    if len(res) == num:
        max_value = max(res)
        max_index = res.index(max_value)
        name = names[max_index]
        res_name = String()
        res_name.data = name
        pub_final_name.publish(res_name)
        
        rospy.logwarn("final_map_result = %s",res_name)




if __name__ == '__main__':

    MAP_VOXEL_SIZE = 0.4
    SCAN_VOXEL_SIZE = 0.1

    # Global localization frequency (HZ)
    FREQ_LOCALIZATION = 0.5

    LOCALIZATION_TH = 0.95
    LOCALIZATION_INI = 0.75
    
    # FOV(rad), modify this according to your LiDAR type
    FOV = 1.6

    # The farthest distance(meters) within FOV
    FOV_FAR = 150


    rospy.init_node('sub_cloud')

    cloud_topic = rospy.get_param('cloud_topic', '/cloud_pcd')

    pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
    pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
    pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)
    
    rospy.Subscriber('/cloud_registered', PointCloud2, cb_save_cur_scan, queue_size=1)
    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)

    sub = rospy.Subscriber(cloud_topic, PointCloud2, cloud_callback)
    sub_pose = rospy.Subscriber('pose_topic', PoseWithCovarianceStamped, pose_callback)
    sub_name = rospy.Subscriber("map_name", String, name_callback)
    sub_num = rospy.Subscriber("map_num", Int32, num_callback)

    done_pub = rospy.Publisher('processing_done', Bool, queue_size=1)
    pub_final_name = rospy.Publisher("final_name", String, queue_size=1)

    rospy.spin()