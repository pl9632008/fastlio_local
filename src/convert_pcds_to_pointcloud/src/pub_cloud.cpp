#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <filesystem>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_srvs/SetBool.h>
#include <mutex>
#include <iostream>
#include <atomic>

#define GREEN "\033[1;32m"
#define BLUE "\033[1;34m"
#define RESET "\033[0m"

ros::Publisher pub_cloud;
ros::Publisher pub_pose_arr;
ros::Publisher pub_name;
ros::Publisher pub_num;

ros::Subscriber sub_final_name;
ros::Subscriber sub_done;
ros::Subscriber sub_odometry;

ros::ServiceClient client;

std::atomic<bool> processing_done = false;
std::atomic<int> cur_map_cnt = 0;
std::atomic<int> cnt = 0;
std::string cloud_topic;
int test_maps_size;

std::mutex mtx_odometry;
double g_before_x = 0;
double g_before_y = 0;
double g_before_z = 0;
double g_cur_x = 0;
double g_cur_y = 0;
double g_cur_z = 0;
bool init_flag = false;
double distance_thresh = 0;


std::vector<std::string> listFiles(const std::string& directory,const std::string & ext) {
    std::vector<std::string> total_names;
    std::filesystem::path p(directory);
    for(auto & entry : std::filesystem::directory_iterator(p)){
        if(entry.path().extension().string() == ext){
            total_names.push_back(entry.path().string());
        }
    }
    return total_names;
}


void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mtx_odometry);
    g_cur_x = msg->pose.pose.position.x;
    g_cur_y = msg->pose.pose.position.y;
    g_cur_z = msg->pose.pose.position.z;
    
}


void finalNameCallback(const std_msgs::String::ConstPtr& msg){

    std::string final_name = msg->data;

}


void doneCallback(const std_msgs::Bool::ConstPtr& msg) {

    processing_done = msg->data ;
    cur_map_cnt++;
    ROS_INFO(GREEN "processing_done = %s , already done = %d / %d " RESET, processing_done.load() ? "true":"false", cur_map_cnt.load(), test_maps_size);
}


void loadAndPublish(const std::vector<std::string>& maps, ros::NodeHandle& nh) {

    if(init_flag==false){

        std::lock_guard<std::mutex> lock(mtx_odometry);
        g_before_x = g_cur_x;
        g_before_y = g_cur_y;
        g_before_z = g_cur_z;
        init_flag = true;

    }else {

        std::lock_guard<std::mutex> lock(mtx_odometry);
        double diff_x = g_cur_x - g_before_x;
        double diff_y = g_cur_y - g_before_y;
        double diff_z = g_cur_z - g_before_z;
        double distance = std::sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

        g_before_x = g_cur_x;
        g_before_y = g_cur_y;
        g_before_z = g_cur_z;

        if(distance < distance_thresh){
            ROS_INFO("moving  %f m, less than %f m, pass loadAndPublish", distance, distance_thresh);
            return;
        }else{

            ROS_INFO("moving  %f m, greater than %f m, doing loadAndPublish", distance, distance_thresh);
        }
    }
 

    std_srvs::SetBool srv;
    srv.request.data = true;  

    while(!client.call(srv)){
        ROS_WARN("Please roslaunch fast_lio_localization localization_horizon_test.launch !");
        ros::Duration(1).sleep();
    }
    ROS_INFO(GREEN "Response: success=%s, message=%s" RESET, srv.response.success ? "true" : "false", srv.response.message.c_str());


    ros::Rate rate(100);  

    ros::Duration(0.5).sleep();

    std::vector<std::string> test_maps;

    for(auto map_pcd : maps){
        std::string map_name = std::filesystem::path(map_pcd).stem().string();
        std::vector<float> pose;
        auto get_name_succeed = nh.getParam(map_name, pose);
        if(!get_name_succeed){
            ROS_WARN("%s does not have initial pose in config.yaml, passes this map!" , map_name.c_str());
        }else{
            test_maps.push_back(map_pcd);
        }
    }


    test_maps_size = test_maps.size();
    std_msgs::Int32 num_msg;
    num_msg.data = test_maps_size;
    pub_num.publish(num_msg);
    cur_map_cnt = 0;
 
    for (const auto& map_pcd : test_maps) {

        std::string map_name = std::filesystem::path(map_pcd).stem().string();
        std::vector<float> pose;
        nh.getParam(map_name, pose);
         
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile(map_pcd, *cloud) < 0) {
            ROS_ERROR_STREAM("Failed to parse pointcloud from file '" << map_pcd << "'");
            continue;
        }
        
        std_msgs::String name_msg;
        name_msg.data = map_name;
        pub_name.publish(name_msg);

        std_msgs::Float64MultiArray pose_array_msg;
        for(int i = 0; i < pose.size(); i++){
            pose_array_msg.data.push_back(pose[i]);
        }
        pub_pose_arr.publish(pose_array_msg);


        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);

        std::string frame_id;
        nh.param<std::string>("frame_id", frame_id, "");
        cloud_msg.header.frame_id = frame_id;
        cloud_msg.header.stamp = ros::Time::now();  // 设置当前时间戳

        ROS_INFO_STREAM(" * File: " << map_pcd);
        ROS_INFO_STREAM(" * Number of points: " << cloud->width * cloud->height);
        
        pub_cloud.publish(cloud_msg);
        ROS_INFO_STREAM(" * pub_cloud cloud done!");

        while (!processing_done) {
            ros::spinOnce();
            rate.sleep();
        }
        processing_done = false;  // 重置标志，准备发布下一个点云

    }

}


int main(int argc, char* argv[]) {
    

    ros::init(argc, argv, "pub_cloud");
    ros::NodeHandle nh;

    std::string matched_path;
    if (!nh.getParam("matched_path", matched_path)) {
        ROS_ERROR("Failed to get parameter 'matched_path'");
        return 1;
    }

    ros::param::get("distance_thresh",distance_thresh);
    ros::param::get("cloud_topic", cloud_topic);

    pub_cloud = nh.advertise<sensor_msgs::PointCloud2>(cloud_topic, 1);
    pub_pose_arr = nh.advertise<std_msgs::Float64MultiArray>("/pose_arr_topic",1);
    pub_name = nh.advertise<std_msgs::String>("/map_name", 1);
    pub_num = nh.advertise<std_msgs::Int32>("/map_num", 1);

    sub_done = nh.subscribe("/processing_done", 1, doneCallback);
    sub_final_name = nh.subscribe("/final_name", 1, finalNameCallback);
    sub_odometry = nh.subscribe<nav_msgs::Odometry>("/Odometry",10,odometryCallback);

    client = nh.serviceClient<std_srvs::SetBool>("/set_bool");

    auto maps = listFiles(matched_path, ".pcd");

    if (maps.empty()) {
        ROS_WARN("No PCD files found in directory '%s'", matched_path.c_str());
        return 0;
    } else{
        loadAndPublish(maps, nh);
    }

    ros::spin();


}