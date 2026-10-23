cmake_minimum_required(VERSION 3.0.2)
project(ria_lo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  pcl_conversions
  roscpp
  sensor_msgs
  tf
)

find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)

# Ceres 1.14 exports uppercase variables; catkin expects package-case names.
set(Ceres_INCLUDE_DIRS ${CERES_INCLUDE_DIRS})
set(Ceres_LIBRARIES ${CERES_LIBRARIES})

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    nav_msgs
    pcl_conversions
    roscpp
    sensor_msgs
    tf
  DEPENDS EIGEN3 PCL Ceres
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(
  ria_lo_laser_processing_node
  src/laserProcessingNode.cpp
  src/laserProcessingClass.cpp
  src/lidar.cpp
)
target_link_libraries(
  ria_lo_laser_processing_node
  ${EIGEN3_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_executable(
  ria_lo_odometry_node
  src/odomEstimationNode.cpp
  src/lidarOptimization.cpp
  src/lidar.cpp
  src/odomEstimationClass.cpp
)
target_link_libraries(
  ria_lo_odometry_node
  ${EIGEN3_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

install(
  TARGETS
    ria_lo_laser_processing_node
    ria_lo_odometry_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  FILES config/kitti.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
