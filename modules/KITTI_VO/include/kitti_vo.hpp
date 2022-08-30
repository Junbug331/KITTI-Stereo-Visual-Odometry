#pragma once
#ifndef KITTI_VO_HPP_
#define KITTI_VO_HPP_

#include <tuple>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include "KITTI_datahandler.hpp"

namespace KITTI_VO
{
    constexpr float EPSILON = std::numeric_limits<float>::epsilon();
    enum DETECTOR {orb, sift, CUDA_orb};
    enum MATHCHER {BF, FLANN};
    enum STEREO_MATHCER {bm, sgbm};
    enum METHOD {mono, stereo, lidar};

    // Calculate disparty map for each pixel(x_L - x_R)
    cv::Mat computeLeftDisparityMap(const cv::Mat &img_left, const cv::Mat &img_right, STEREO_MATHCER matcher=STEREO_MATHCER::sgbm);

    void decomposeProjectionMatrix(const cv::Mat &P, cv::Mat &K, cv::Mat &R, cv::Mat &t);

    cv::Mat calculateDepthMap(const cv::Mat &disp_left, const cv::Mat &K, const cv::Mat &t_left, const cv::Mat &t_right, bool rectified=true);

    std::tuple<cv::Mat, cv::Mat> stereo2depth(const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &P0, const cv::Mat &P1,
                         STEREO_MATHCER matcher = STEREO_MATHCER::bm, bool rectified = true);

    void extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts, cv::OutputArray &desc,
                         cv::InputArray mask = cv::noArray(), DETECTOR detector = DETECTOR::sift);

    void matchFeatures(cv::OutputArray desc1, cv::OutputArray desc2, std::vector<cv::DMatch> &matches,
                       DETECTOR detector = DETECTOR::sift,
                       MATHCHER matcher = MATHCHER::BF,
                       bool sort = true,
                       int k = 2);

    void visualizeMatches(const cv::Mat &img1, const std::vector<cv::KeyPoint> &kpts1,
                          const cv::Mat &img2, const std::vector<cv::KeyPoint> &kpts2,
                          const std::vector<cv::DMatch> &matches);

    cv::Mat pointCloud2Image(pcl_cloud_ptr cloud, size_t img_height, size_t img_width, const cv::Mat &Rt, const cv::Mat &P0);

    void estimateMotion(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::KeyPoint> &kp2,
                        const cv::Mat &K,
                        cv::Mat &R, cv::Mat &t, std::vector<cv::Point2f> &img1_points, std::vector<cv::Point2f> &img2_points,
                        cv::InputArray depth1 = cv::noArray(), float max_depth = 3000.f);

    void cvMat2pcl(const cv::Mat &lidar_points, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    cv::Mat inverseTransformation(cv::InputArray Rt);
    std::tuple<cv::Mat, cv::Mat> inverseTransformation(cv::InputArray R, cv::InputArray t);

    float calculateDepthError(const cv::Mat &depth, const cv::Mat &lidar, bool verbose=false);

    double calculatePoseError(const std::vector<cv::Mat> &estimate, const std::vector<cv::Mat> &gt);

    void visualOdometry(KITTIDataHandler &handler,
                        METHOD method,
                        STEREO_MATHCER stereo_matcher,
                        DETECTOR detector,
                        MATHCHER matcher,
                        std::vector<cv::Mat> &trajectory);

    /// Visualizers
    pcl::visualization::PCLVisualizer::Ptr customColourVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
    pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
    void saveTrajectory(const std::vector<cv::Mat>& trajectory, const std::string &filename);
    void drawTrajectory(const std::vector<cv::Mat>& trajectory);

}


#endif