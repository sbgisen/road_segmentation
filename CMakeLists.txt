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

# catkin_install_python(
#   PROGRAMS scripts/road_segmentation_node.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(
  DIRECTORY src/road_segmentation/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/road_segmentation
)
