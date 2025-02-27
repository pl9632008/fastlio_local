#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <iostream>
#include <queue>


#define BLUE "\033[1;34m"
#define RESET "\033[0m"
ros::Publisher pub_vec;



class MovingAverageQueue {

public:
    std::queue<double> q;
    int maxSize;
    double sum;

    MovingAverageQueue(int size) {
        maxSize = size;
        sum = 0.0;
    }

    void push(double val) {
        if (q.size() >= maxSize) {
            sum -= q.front();
            q.pop();
        }
        q.push(val);
        sum += val;
    }

    double getAverage() {
        if (q.empty()) {
            return 0.0;
        }
        return sum / q.size();
    }
};


MovingAverageQueue moveavg_x(5);
MovingAverageQueue moveavg_y(5);
MovingAverageQueue moveavg_z(5);



void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    double vec_x = msg->twist.twist.linear.x;
    double vec_y = msg->twist.twist.linear.y;
    double vec_z = msg->twist.twist.linear.z;

    moveavg_x.push(vec_x);
    moveavg_y.push(vec_y);
    moveavg_z.push(vec_z);

    vec_x = moveavg_x.getAverage();
    vec_y = moveavg_y.getAverage();
    vec_z = moveavg_z.getAverage();


    double vec_total = std::sqrt( vec_x*vec_x + vec_y*vec_y  + vec_z*vec_z );

    ROS_WARN("Current vec_total = %f m/s", vec_total);
    ROS_INFO(BLUE "Current vec_total = %f km/h" RESET, vec_total*3.6);

    std_msgs::Float64 msg_vec;
    msg_vec.data = vec_total*3.6;
    pub_vec.publish(msg_vec);

}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "odometry_subscriber");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/Odometry", 1, odometryCallback);
    pub_vec = nh.advertise<std_msgs::Float64>("/current_velocity", 1);

    ros::spin();
    
    return 0;
}
