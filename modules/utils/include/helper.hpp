#pragma once
#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <opencv2/opencv.hpp>
#include <tuple>
#include <numeric>

namespace KITTI_VO
{

    constexpr float EPSILON = std::numeric_limits<float>::epsilon();

    enum STEREO_MATHCER {bm, sgbm};

    // Calculate disparty map for each pixel(x_L - x_R)
    cv::Mat computeLeftDisparityMap(const cv::Mat &img_left, const cv::Mat &img_right, STEREO_MATHCER matcher=STEREO_MATHCER::sgbm);

    void decomposeProjectionMatrix(const cv::Mat &P, cv::Mat &K, cv::Mat &R, cv::Mat &t);

    cv::Mat calculateDepthMap(const cv::Mat &disp_left, const cv::Mat &K, const cv::Mat &t_left, const cv::Mat &t_right, bool rectified=true);

    std::tuple<cv::Mat, cv::Mat> stereo2depth(const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &P0, const cv::Mat &P1,
                         STEREO_MATHCER matcher = STEREO_MATHCER::sgbm, bool rectified = true);
}



#endif