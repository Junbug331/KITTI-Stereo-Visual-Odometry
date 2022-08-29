#include "helper.hpp"
#include <opencv2/calib3d.hpp>

cv::Mat KITTI_VO::computeLeftDisparityMap(const cv::Mat &img_left, const cv::Mat &img_right, STEREO_MATHCER matcher)
{
    int sad_window = 6;
    int num_disparities = sad_window*16;
    int block_size = 11;
    cv::Mat disp_left;

    if (matcher == STEREO_MATHCER::bm)
    {
        auto matcher = cv::StereoBM::create(num_disparities, block_size);
            matcher->compute(img_left, img_right, disp_left);
    }
    else if (matcher == STEREO_MATHCER::sgbm)
    {
        auto matcher = cv::StereoSGBM::create(
            0,
            num_disparities,
            block_size,
            8*3*(sad_window*sad_window),
            32*3*(sad_window*sad_window),
            0, 0, 0, 0, 0,
            cv::StereoSGBM::MODE_SGBM_3WAY
        );
        matcher->compute(img_left, img_right, disp_left);
    }
    disp_left.convertTo(disp_left, CV_32F);

    return disp_left/16; // IMPORTANT divide by 16
}

void KITTI_VO::decomposeProjectionMatrix(const cv::Mat &P, cv::Mat &K, cv::Mat &R, cv::Mat &t)
{
    cv::decomposeProjectionMatrix(P, K, R, t);
    t /= t.at<float>(3);
}


cv::Mat KITTI_VO::calculateDepthMap(const cv::Mat &disp_left, const cv::Mat &K, const cv::Mat &t_left, const cv::Mat &t_right, bool rectified)
{
    /*
     *  Z = (f*b)/disparity
     */
    float f = K.at<float>(0, 0); // focal length

    float b; // baseline
    if (rectified)
        b = t_right.at<float>(0) - t_left.at<float>(0);
    else
        b = t_left.at<float>(0) - t_right.at<float>(0);

    cv::Mat depth_map;
    disp_left.copyTo(depth_map);

    int cols = disp_left.cols;
    size_t size = disp_left.rows*cols;
    cv::parallel_for_(cv::Range(0, size), [&](const cv::Range &range){
        for (int i=range.start; i<range.end; ++i)
        {
            int r = i/cols;
            int c = i - (r*cols);

            auto &p = depth_map.at<float>(r, c);
            if (abs(p-0.f) <= EPSILON) p = 0.1f;
            if (abs(p+1.f) <= EPSILON) p = 0.1f;
        }
    });

    float multiplier = f*b;
    depth_map = multiplier / depth_map;

    return depth_map;
}

std::tuple<cv::Mat, cv::Mat> KITTI_VO::stereo2depth(const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &P0, const cv::Mat &P1,
                               STEREO_MATHCER matcher, bool rectified)
{
    // Compute disparity map
    cv::Mat disp = computeLeftDisparityMap(img_left, img_right, matcher);

    // Decompose projection matrices
    cv::Mat K_l, R_l, t_l, K_r, R_r, t_r;
    decomposeProjectionMatrix(P0, K_l, R_l, t_l);
    decomposeProjectionMatrix(P1, K_r, R_r, t_r);

    // Calculate depth map for the left camera and return
    cv::Mat depth = calculateDepthMap(disp, K_l, t_l, t_r);

    // Calculate Mask
    float maxVal = depth.at<float>(0, 0);
    int x_loc; // first non_max index on x-axis
    float *p = depth.ptr<float>(4);
    for (int i=0; i<depth.cols; i++)
    {
        if (p[i] < maxVal)
        {
            x_loc = i;
            break;
        }
    }
    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8U);
    std::cout << maxVal << ", " << x_loc << std::endl;
    mask(cv::Rect( x_loc, 0, depth.cols-x_loc, depth.rows)) = 255;

    return {depth, mask};
}
