cmake_minimum_required(VERSION 3.0.2)
project(road_segmentation)

find_package(
  catkin REQUIRED
  COMPONENTS rospy
             std_msgs
             sensor_msgs
             cv_bridge
             jsk_recognition_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    cv_bridge
    jsk_recognition_msgs
)

include_directories(${catkin_INCLUDE_DIRS})
