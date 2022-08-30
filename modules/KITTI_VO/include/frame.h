//
// Created by junbug331 on 22. 8. 30.
//

#ifndef VISUAL_ODOMETRY_FRAME_H
#define VISUAL_ODOMETRY_FRAME_H

#include <opencv2/core/cuda.hpp>
#include "KITTI_datahandler.hpp"

struct Frame
{
    int num;
    cv::Mat img_l;
    cv::Mat img_r;
    cv::Mat depth;
    cv::Mat mask;
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    cv::cuda::GpuMat d_desc;
    std::vector<cv::Point2f> points;
};


#endif //VISUAL_ODOMETRY_FRAME_H
