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

from std_msgs.msg import Bool, String, Int32, Float64, Float64MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from threading import Lock
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

import std_msgs.msg
import threading


GREEN = "\033[92m"
RESET = "\033[0m"

global_map = None
initialized = False
T_map_to_odom = np.eye(4)
cur_odom = None
cur_scan = None
max_score = 0


g_name = String()
mtx_name = Lock()
mtx_num = Lock()
mtx_result_name = Lock()
mtx_pose_arr = Lock()

res_scores = []
names = []
all_poses = []
temp_poses = []
g_pose_arr = []
num = -1

cloud_topic = None
pub_final_name = None
pub_cal_done = None
pub_find_map = None
pub_cusmap = None
pub_score = None
result_pose = None
ros_cloud = None
final_result_name = None
select_map_done = False

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


def global_localization(pose_estimation):
    global global_map, cur_scan, cur_odom, T_map_to_odom, initialized, LOCALIZATION_TH, max_score
    # 用icp配准
    # print(global_map, cur_scan, T_map_to_odom)
    rospy.loginfo('Global localization by scan-to-map matching......')

    # TODO 这里注意线程安全
    scan_tobe_mapped = copy.copy(cur_scan)

    tic = time.time()

    global_map_in_FOV = crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)

    # 粗配准
    transformation, _ = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)

    # 精配准
    transformation, fitness = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation,
                                                    scale=1)
    toc = time.time()

    current_score = Float64()
    current_score.data = fitness
    pub_score.publish(current_score)

    # 当全局定位成功时才更新map2odom
    if initialized == False:
        if fitness < LOCALIZATION_TH:
            LOCALIZATION_TH -= 0.05

    elif initialized == True:

        LOCALIZATION_TH = max(LOCALIZATION_TH , max_score)
        if LOCALIZATION_TH > 0.9:
            LOCALIZATION_TH = 0.9
    
    rospy.loginfo('localization_thresold: {}'.format(LOCALIZATION_TH))
    rospy.loginfo('Time: {:.2f}'.format(toc - tic))
    rospy.loginfo('score: {:.2f}'.format(fitness))


    if fitness > LOCALIZATION_TH:
        # T_map_to_odom = np.matmul(transformation, pose_estimation)

        rospy.loginfo(f"{GREEN}{'Already match!'}{RESET}")
        rospy.loginfo('')

        T_map_to_odom = transformation
        max_score = max(fitness, LOCALIZATION_TH)

        # 发布map_to_odom
        map_to_odom = Odometry()
        xyz = tf.transformations.translation_from_matrix(T_map_to_odom)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_odom)
        map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        map_to_odom.header.stamp = cur_odom.header.stamp
        map_to_odom.header.frame_id = 'map'
        pub_map_to_odom.publish(map_to_odom)
        return True
    else:
        rospy.logwarn('Not match!!!!')
        rospy.loginfo('')
        # rospy.logwarn('{}'.format(transformation))
        # rospy.logwarn('fitness score:{}'.format(fitness))
        return False



def global_localization2(pose_estimation):
    global global_map, cur_scan, cur_odom
    # 用icp配准
    rospy.loginfo('Selecting maps, matching......')

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


def initialize_global_map2(pc_msg):
    global global_map, ros_cloud
    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
    ros_cloud = None
    rospy.loginfo('Global map received, clear ros_cloud !')


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


def thread_localization():
    global T_map_to_odom, select_map_done
    while True and select_map_done:
        # 每隔一段时间进行全局定位
        rospy.sleep(1 / FREQ_LOCALIZATION)
        # TODO 由于这里Fast lio发布的scan是已经转换到odom系下了 所以每次全局定位的初始解就是上一次的map2odom 不需要再拿odom了
        global_localization(T_map_to_odom)


def publish_map_name():
    rospy.wait_for_message("/find_map_done", Bool)
    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():
        res_name = String()
        with mtx_result_name:
            res_name.data = final_result_name
        pub_final_name.publish(res_name)
        # rospy.loginfo('final_name =  {}'.format(res_name.data))
        rate.sleep()


def service_callback(req):
    rospy.loginfo(f"{GREEN}{'Request received: data=%s'}{RESET}", req.data)
    success = True 
    message = "Received true" if req.data else "Received false"
    return SetBoolResponse(success=success, message=message)

def num_callback(msg):
    global num
    with mtx_num:
        num = msg.data
        rospy.loginfo("Numbers of maps: %d",num)

def name_callback(msg):
    global g_name
    with mtx_name:
        g_name = msg

def pose_arr_callback(msg):
    global g_pose_arr
    with mtx_pose_arr:
        g_pose_arr = msg.data

   
def cloud_callback(msg):
    global result_pose, ros_cloud, final_result_name, res_scores, names, all_poses,select_map_done,g_pose_arr, temp_poses
    
    rospy.loginfo("")
    rospy.logwarn("Received point clouds!")
    select_map_done = False

    cur_name = String()
    with mtx_name:
        cur_name = g_name

    cur_pose_arr=[]
    with mtx_pose_arr:
        cur_pose_arr = g_pose_arr
 
    initialize_global_map(msg)

    temp_res=[]
    temp_initial_pose=[]
    for i in range(len(cur_pose_arr)//7):

        pose_with_cov_stamped = PoseWithCovarianceStamped()
        pose_with_cov_stamped.header.stamp = rospy.Time.now()
        pose_with_cov_stamped.header.frame_id = "map"
        pose_with_cov_stamped.pose.pose.position.x = cur_pose_arr[i*7+0]
        pose_with_cov_stamped.pose.pose.position.y = cur_pose_arr[i*7+1]
        pose_with_cov_stamped.pose.pose.position.z = cur_pose_arr[i*7+2]
        pose_with_cov_stamped.pose.pose.orientation.x = cur_pose_arr[i*7+3]
        pose_with_cov_stamped.pose.pose.orientation.y = cur_pose_arr[i*7+4]
        pose_with_cov_stamped.pose.pose.orientation.z = cur_pose_arr[i*7+5]
        pose_with_cov_stamped.pose.pose.orientation.w = cur_pose_arr[i*7+6]
        pose_with_cov_stamped.pose.covariance = [0] * 36

        initial_pose = pose_to_mat(pose_with_cov_stamped)
        temp_initial_pose.append(pose_with_cov_stamped)

        init = False
        while not init:
            if cur_scan:
                init, score = global_localization2(initial_pose)

                # ignore pose that has been used
                # for j in temp_poses:
                #     if  j.pose == pose_with_cov_stamped.pose:
                #         score = 0

                temp_res.append(score)
                rospy.loginfo("name = {}, index = {}, score = {:.2f}".format(cur_name.data, i, score))

            else:
                # rospy.logwarn('First scan not received!!!!!')
                pass

    max_index_temp = temp_res.index(max(temp_res))
    res_scores.append( temp_res[max_index_temp] )
    names.append(cur_name.data)
    all_poses.append( temp_initial_pose[max_index_temp] )


    cur_num = None
    with mtx_num:
        cur_num = num
    
    cal_msg = Bool()
    cal_msg.data = True
    pub_cal_done.publish(cal_msg)
    rospy.loginfo(f"{GREEN}{'Calculated current map score done!'}{RESET}")

    if len(res_scores) == cur_num:
        max_value = max(res_scores)
        max_index = res_scores.index(max_value)
        name = names[max_index]
        result_pose = all_poses[max_index]
        temp_poses.append(result_pose)

        with mtx_result_name:
            final_result_name = name

        # res_name = String()
        # res_name.data = name
        # pub_final_name.publish(res_name)

        map_path = rospy.get_param('map_path', '')
        file_path = map_path + "/" + name + ".pcd"

        rospy.logwarn(file_path)

        o3d_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(o3d_cloud.points)

        o3d_cloud = None

        data = np.zeros(len(points), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('intensity', np.float32),
        ])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        if points.shape[1] == 4:
            data['intensity'] = points[:, 3]
        
        points = None

        ros_cloud = ros_numpy.msgify(PointCloud2, data)

        data = None
        
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        
        ros_cloud.header = header
        pub_cusmap.publish(ros_cloud)

        res_scores = []
        names = []
        all_poses = []
        init = False
        select_map_done = True

        all_done_msg = Bool()
        all_done_msg.data = True
        pub_find_map.publish(all_done_msg)
        rospy.loginfo(f"{GREEN}{'Find map done!'}{RESET}")





if __name__ == '__main__':
    MAP_VOXEL_SIZE = 0.4
    SCAN_VOXEL_SIZE = 0.1

    # Global localization frequency (HZ)
    FREQ_LOCALIZATION = 0.5

    # The threshold of global localization,
    # only those scan2map-matching with higher fitness than LOCALIZATION_TH will be taken
    LOCALIZATION_TH = 0.7    

    # FOV(rad), modify this according to your LiDAR type
    # FOV = 1.6
    FOV = 2.09

    # The farthest distance(meters) within FOV
    FOV_FAR = 150

    rospy.init_node('fast_lio_localization')
    rospy.loginfo('Localization Node Inited...')

    sender_thread = threading.Thread(target=publish_map_name)
    sender_thread.start()


    # publisher
    pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
    pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
    pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)

    rospy.Subscriber('/cloud_registered', PointCloud2, cb_save_cur_scan, queue_size=1)
    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)


    pub_cal_done = rospy.Publisher('/processing_done', Bool, queue_size=1)
    pub_final_name = rospy.Publisher("/final_name", String, queue_size=1)
    pub_find_map = rospy.Publisher("/find_map_done", Bool, queue_size=1)
    pub_cusmap = rospy.Publisher('/map', PointCloud2, queue_size=1)
    pub_score = rospy.Publisher("/current_score", Float64, queue_size=1)

    cloud_topic = rospy.get_param('cloud_topic', '/cloud_pcd')
    sub_cloud = rospy.Subscriber(cloud_topic, PointCloud2, cloud_callback)
    sub_name = rospy.Subscriber("/map_name", String, name_callback)
    sub_num = rospy.Subscriber("/map_num", Int32, num_callback)
    sub_pose_arr = rospy.Subscriber("/pose_arr_topic", Float64MultiArray, pose_arr_callback)
    service = rospy.Service('/set_bool', SetBool, service_callback)

    while True:

        rospy.logwarn('Waiting for find_map_done......')
        rospy.wait_for_message("/find_map_done", Bool)

        # 初始化全局地图
        rospy.logwarn('Waiting for global map......')

        # initialize_global_map(rospy.wait_for_message('/map', PointCloud2))
        initialize_global_map2(ros_cloud)
        
        # 初始化
        initialized = False
        while not initialized:

            if cur_scan:

                rospy.logwarn('Waiting for initial pose....')
                pose_msg = result_pose
                initial_pose = pose_to_mat(pose_msg)

                initialized = global_localization(initial_pose)
                rospy.sleep(0.05)

            else:
                rospy.logwarn('First scan not received!!!!!')

        rospy.loginfo('')
        rospy.loginfo('Initialized successfully!!!!!!')
        rospy.loginfo('')
        # 开始定期全局定位
        thread.start_new_thread(thread_localization, ())



