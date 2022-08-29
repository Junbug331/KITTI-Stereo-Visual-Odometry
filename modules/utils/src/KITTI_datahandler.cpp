#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>

#include <opencv2/opencv.hpp>

#include "KITTI_datahandler.hpp"

/// Buffer for reading 4D lidar point
struct Buffer 
{
    float x, y, z, r;
};

// Static function
pcl_cloud_ptr KITTIDataHandler::loadLidarPoints(const std::string &lidar_path)
{
    /*
        KITTI devkit code provided in readme.txt 

        in devkit/readme.txt
    */
    int32_t num = 1e6;
    std::unique_ptr<float[]> data = std::make_unique<float[]>(num);

    float *px = data.get()+0;
    float *py = data.get()+1;
    float *pz = data.get()+2;
    float *pr = data.get()+3;
    FILE *stream;
    stream = fopen(lidar_path.c_str(), "rb");
    num = fread(data.get(), sizeof(float), num, stream)/4;

    pcl_cloud_ptr cloud(new pcl_cloud);
    cloud->width = num;
    cloud->height = 1;
    cloud->points.resize(num);
    for (int32_t i=0; i<num; ++i)
    {
        cloud->points[i].x = *px;
        cloud->points[i].y = *py;
        cloud->points[i].z = *pz;
        cloud->points[i].rgb = *pr;
        px += 4; py += 4; pz += 4; pr += 4; 
    }
    fclose(stream);

    return cloud;
}

void KITTIDataHandler::pcl2CVMat(const pcl_cloud_ptr &cloud, cv::Mat &lidar_points)
{
    size_t size = cloud->points.size();
    lidar_points = cv::Mat(size, 3, CV_32F);
    for (int i=0; i<size; ++i)
    {
        float *p = lidar_points.ptr<float>(i);
        const pcl_point& pcl_p = cloud->points[i];

        p[0] = pcl_p.x;
        p[1] = pcl_p.y;
        p[2] = pcl_p.z;
    }
}

KITTIDataHandler::KITTIDataHandler(std::string &sequence, bool lidar)
{
    seq_dir = fs::path("KITTI/dataset/sequences/"+sequence);
    poses_path = fs::path("KITTI/poses/" + sequence + ".txt");
    calib_dir = fs::path("KITTI/calibration/sequences/"+sequence);
    lidar_path = fs::path("KITTI/lidar/dataset/sequences")/fs::path(sequence +"/velodyne");
    
    // load image paths (left_image_files, right_image_files)
    loadImageDataset(seq_dir);

    // load calibration data and times (P1 ~ P3, Rt, times)
    loadCalibrationFiles(calib_dir);

    // load GT poses and times (gt)
    loadGTPoses(poses_path);

    // Load pointcloud paths(ptcloud_files)
    if (lidar)
        loadPointCloudFiles(lidar_path);

    // Set number of frames
    num_frames = left_image_files.size();


    first_image_left = cv::imread(left_image_files[0], 0);
    first_image_right = cv::imread(right_image_files[0], 0);
    second_image_left = cv::imread(left_image_files[1], 0);
    if (lidar)
        first_pointcloud = loadLidarPoints(ptcloud_files[0]);

    // Set frame width and height
    frame_width = first_image_left.cols;
    frame_height = first_image_left.rows;
}

void KITTIDataHandler::loadImageDataset(const std::string &dataset_path)
{
    fs::path image_0 = seq_dir/fs::path("image_0");
    fs::path image_1 = seq_dir/fs::path("image_1");

    if (!fs::is_directory(image_0))
        throw std::runtime_error(image_0.string() + " is not a directory\n");
    if (!fs::is_directory(image_1))
        throw std::runtime_error(image_1.string() + " is not a directory\n");

    left_image_files.reserve(5000);
    right_image_files.reserve(5000);

    cv::glob(image_0.string(), left_image_files);
    cv::glob(image_1.string(), right_image_files);

    std::sort(left_image_files.begin(), left_image_files.end());
    std::sort(right_image_files.begin(), right_image_files.end());
}


void KITTIDataHandler::loadPointCloudFiles(const std::string &lidar_path)
{
    if (!fs::is_directory(lidar_path))
        throw std::runtime_error(lidar_path + " is not a directory\n");

    ptcloud_files.reserve(5000);

    cv::glob(lidar_path, ptcloud_files);
    std::sort(ptcloud_files.begin(), ptcloud_files.end());
}



void KITTIDataHandler::loadCalibrationFiles(const std::string &calib_file_path)
{
    // Load calibration matrices
    cv::Mat matrices[5];
    for (int i=0; i<5; ++i)
        matrices[i] = cv::Mat::zeros(3, 4, CV_32F);
    std::ifstream in((calib_file_path/fs::path("calib.txt")));
    std::string line;

    for (int i=0; i<5; ++i)
    {
        getline(in, line);

        std::istringstream iss(line);
        int j=0;

        std::string tmp; iss >> tmp;
        while( iss >> matrices[i].at<float>(j/4, j%4)) j++;
    }

    P0 = matrices[0];
    P1 = matrices[1];
    P2 = matrices[2];
    P3 = matrices[3];
    Rt = matrices[4];

    in.close();

    // Load time stamp

    times.reserve(5000);
    // Load timestamp
    in = std::ifstream((calib_file_path/fs::path("times.txt")));
    while(getline(in, line))
    {
        times.push_back(stof(line));
    }

    in.close();
}

void KITTIDataHandler::loadGTPoses(const std::string &poses_path)
{
    gt.reserve(5000);

    std::ifstream in(poses_path);
    if (!in.is_open())
    {
        throw std::runtime_error("Can't open " + poses_path);
    }

    std::string line;

    int i=0, j; 
    while(getline(in, line))
    {
        std::istringstream iss(line);
        gt.emplace_back(cv::Mat(3, 4, CV_32F));
        j = 0;
        while (iss >> gt.back().at<float>(j/4, j%4)) j++;
    }
}