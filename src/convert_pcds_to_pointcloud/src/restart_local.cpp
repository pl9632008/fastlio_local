#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <mutex>
#include <thread>

ros::Subscriber sub_odometry;
double g_cur_x = 0;
double g_cur_y = 0;
double g_cur_z = 0;
double g_init_x = 0;
double g_init_y = 0;
double g_init_z = 0;
bool g_init_flag = false;
double distance_thresh;
float vel_thresh;
int count_thresh;
int cnt = 0;

void killAndRestart(){
    std::string restart_cmd2 = "xterm -e  roslaunch convert_pcds_to_pointcloud convert_local.launch";
    std::system(restart_cmd2.c_str());
}


void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    if(g_init_flag == false){
        g_init_x = msg->pose.pose.position.x;
        g_init_y = msg->pose.pose.position.y;
        g_init_z = msg->pose.pose.position.z;
        g_init_flag = true;

    }else{
        g_cur_x = msg->pose.pose.position.x;
        g_cur_y = msg->pose.pose.position.y;
        g_cur_z = msg->pose.pose.position.z;
    }


    double diff_x = g_cur_x - g_init_x;
    double diff_y = g_cur_y - g_init_y;
    double diff_z = g_cur_z - g_init_z;
    double distance = std::sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

    double vec_x = msg->twist.twist.linear.x;
    double vec_y = msg->twist.twist.linear.y;
    double vec_z = msg->twist.twist.linear.z;
    double vec_total = std::sqrt( vec_x*vec_x + vec_y*vec_y  + vec_z*vec_z );
    if( vec_total * 3.6 < vel_thresh){
        cnt++;
    }else{
        cnt=0;
    }

    if(cnt > count_thresh ){
        if(distance > distance_thresh){
            g_init_x = 0;
            g_init_y = 0;
            g_init_z = 0;
            cnt = 0;
            std::string kill_cmd = "rosnode kill /global_localization /laserMapping /transform_fusion /pub_cloud /sub_odometry /rviz";
            std::system(kill_cmd.c_str());

            std::thread t(killAndRestart);
            t.detach();
        }
    }
    ROS_INFO("cnt = %d, count_thresh = %d, distance = %f, distance_thresh = %f", cnt, count_thresh, distance, distance_thresh);

}



int main(int argc, char* argv[]){

    ros::init(argc, argv, "restart_local");
    ros::NodeHandle nh;
    ros::param::get("distance_thresh", distance_thresh);
    ros::param::get("count_thresh",count_thresh);
    ros::param::get("vel_thresh",vel_thresh);
    sub_odometry = nh.subscribe<nav_msgs::Odometry>("/Odometry", 1, odometryCallback);
    ros::spin();

}