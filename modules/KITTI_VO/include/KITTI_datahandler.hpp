#pragma once
#ifndef KITTI_DATAHANDLER_HPP_
#define KITTI_DATAHANDLER_HPP_

#include <filesystem>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace fs = std::filesystem;

using pcl_point = pcl::PointXYZRGB;
using pcl_cloud = pcl::PointCloud<pcl_point>; 
using pcl_cloud_ptr = pcl_cloud::Ptr;

class KITTIDataHandler
{
public:
// Constructor
KITTIDataHandler() = default;
~KITTIDataHandler() = default;
explicit KITTIDataHandler(std::string &sequence, bool lidar=true);
KITTIDataHandler(const KITTIDataHandler&) = delete;

static pcl_cloud_ptr loadLidarPoints(const std::string &lidar_file_path);
static void pcl2CVMat(const pcl_cloud_ptr &cloud, cv::Mat &lidar_points);

private:
    void loadImageDataset(const std::string& dataset_path);
    void loadPointCloudFiles(const std::string &lidar_path);
    void loadCalibrationFiles(const std::string& calib_file_path);
    void loadGTPoses(const std::string& poses_path);

public:
// Member Variables
	fs::path seq_dir;
    fs::path calib_dir;
	fs::path poses_path;
	fs::path lidar_path;

	std::vector<std::string> left_image_files;
	std::vector<std::string> right_image_files;
    std::vector<std::string> ptcloud_files;

	size_t num_frames;
    int frame_width;
    int frame_height;

    // Rectified Projection
    cv::Mat P0; // Projection Matrix Cam_0 coordinate(3D) to Cam_0 image plane(2D) 
    cv::Mat P1; // Projection Matrix Cam_1 coordinate(3D) to Cam_0 image plane(2D)
    cv::Mat P2; // Projection Matrix Cam_2 coordinate(3D) to Cam_0 image plane(2D)
    cv::Mat P3; // Projection Matrix Cam_3 coordinate(3D) to Cam_0 image plane(2D)
    cv::Mat Rt; // Extrinsic Matrix from 3D lidar coordinate(3D) to Cam_0 coordinate(3D)
    std::vector<float> times; // time stamp

    std::vector<cv::Mat> gt; // [R|t], transformation matrix from relative pose to global pose

    cv::Mat first_image_left;
    cv::Mat first_image_right;
    cv::Mat second_image_left;
    pcl_cloud_ptr first_pointcloud;
};

#endif