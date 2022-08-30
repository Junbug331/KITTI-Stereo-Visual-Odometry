cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(KITTI_VO)
set(LIB_NAME KITTI_VO)

add_library(${LIB_NAME} 
    src/KITTI_datahandler.cpp
    src/kitti_vo.cpp
    src/frame.cpp
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

find_package(Pangolin REQUIRED)
if (Pangolin_FOUND)
    message(STATUS "Pangolin(${Pangolin_VERSION}) Found! - ${Pangolin_DIR}")
endif()

find_package(spdlog 1.9.2 EXACT REQUIRED PATHS ../thirdparty/spdlog-1.9.2/install/lib/cmake/spdlog)
if (spdlog_FOUND)
    message(STATUS "spdlog(${spdlog_VERSION}) Found! - ${spdlog_DIR}")
    include_directories(../thirdparty/spdlog-1.9.2/install/include)
endif()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

target_compile_options(${LIB_NAME} PUBLIC -pthread)


target_link_directories(${LIB_NAME} PUBLIC
    ${PCL_LIBRARY_DIRS}
    spdlog::spdlog
)

target_link_libraries(${LIB_NAME} PUBLIC
    stdc++fs
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Pangolin_LIBRARIES}
)
