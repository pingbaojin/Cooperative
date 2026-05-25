cmake_minimum_required(VERSION 3.0.2)
project(cooperative_link)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  message_generation
)

add_message_files(
  FILES
  UwbRange.msg
  UwbRangeArray.msg
  CooperativeTargets.msg
  CoordFrameEstimate.msg
  DynamicTarget.msg
  DynamicTargetArray.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/cooperative_link_node
  scripts/nav_file_to_ros_publisher
  scripts/replay_dynamic_assoc
  scripts/eval_cooperative_link
  scripts/plot_coop_targets
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch rviz docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
