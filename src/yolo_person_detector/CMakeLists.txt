cmake_minimum_required(VERSION 3.0.2)
project(yolo_person_detector)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()
catkin_package()

install(PROGRAMS
  yolo_person_detector/rgbd_pedestrian_detector_ros1.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
