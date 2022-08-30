cmake_minimum_required(VERSION 3.10)
project(visual-odometry)
set(TARGET ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SRC_FILES 
	src/main.cpp
)

add_subdirectory(modules/KITTI_VO)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(${TARGET} ${SRC_FILES})

include_directories(
	modules/KITTI_VO/include
	)

target_compile_options(${TARGET} PRIVATE -pthread)

target_link_libraries(${TARGET} PUBLIC
	KITTI_VO
)

set (src "${CMAKE_SOURCE_DIR}/KITTI")
set (dst "${CMAKE_CURRENT_BINARY_DIR}/KITTI")
add_custom_command(
	TARGET ${TARGET} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dst}
	DEPENDS ${dst}
	COMMENT "symbolic link res folder from ${src} => ${dst}"
	)

