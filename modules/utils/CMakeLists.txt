cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(utils)
set(LIB_NAME utils)

add_library(${LIB_NAME} 
    src/KITTI_datahandler.cpp
    src/helper.cpp
    )

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	message(STATUS "OpenCV(${OpenCV_VERSION}) Found! - ${OpenCV_DIR}")
endif()

find_package(PCL 1.12.1.99 EXACT REQUIRED PATHS ../thirdparty/pcl/install/share/pcl-1.12)
if (PCL_FOUND)
    message(STATUS "PCL(${PCL_VERSION}) Found! - ${PCL_DIR}")
    add_definitions(${PCL_DEFINITIONS})
endif()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_compile_options(${LIB_NAME} PUBLIC -pthread)


target_link_directories(${LIB_NAME} PUBLIC
    ${PCL_LIBRARY_DIRS}
)

target_link_libraries(${LIB_NAME} PUBLIC
    stdc++fs
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)
