cmake_minimum_required(VERSION 3.0.2)
project(dpnet_bridge_overlay)

find_package(catkin REQUIRED COMPONENTS
  roslaunch
)

catkin_package()

if(CATKIN_ENABLE_TESTING)
  roslaunch_add_file_check(launch)
endif()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
