cmake_minimum_required(VERSION 3.0.2)
project(adapt_full)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()
catkin_package()

install(PROGRAMS
  adapt_full/adapt_lidar_processing_ros1.py
  adapt_full/adapt_lidar_camera_fusion_ros1.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config waypoints
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
