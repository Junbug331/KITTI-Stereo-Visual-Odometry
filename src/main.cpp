#include <iostream>
#include <vector>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include "kitti_vo.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    fs::path ROOT(ROOT_DIR);
	string seq = "00";
	KITTIDataHandler handler(seq);
    vector<cv::Mat> trajectory;
    vector<cv::DMatch> debug_matches;
    vector<cv::KeyPoint> debug_kp1, debug_kp2;

    KITTI_VO::visualOdometry(handler,
                             KITTI_VO::METHOD::stereo,
                             KITTI_VO::STEREO_MATHCER::bm,
                             KITTI_VO::DETECTOR::sift,
                             KITTI_VO::MATHCHER::BF,
                             trajectory);
    cout << trajectory[0] << endl;
    fs::path filename = ROOT/fs::path("poses_"+seq+".txt");
    KITTI_VO::saveTrajectory(trajectory, filename.string());
    //KITTI_VO::drawTrajectory(trajectory);
    cout << KITTI_VO::calculatePoseError(trajectory, handler.gt) << endl;

//    cv::Mat P0, P1, img1_l, img1_r, img2, desc1, desc2, K, R, t;
//    std::vector<KeyPoint> kp1, kp2;
//    std::vector<DMatch> matches;
//    std::vector<Point2f> pt1, pt2;
//    pcl_cloud_ptr cloud = KITTIDataHandler::loadLidarPoints(handler.ptcloud_files[2]);
//    img1_l = imread(handler.left_image_files[2], 0);
//    img1_r = imread(handler.right_image_files[2], 0);
//    img2 = imread(handler.left_image_files[3], 0);
//    P0 = handler.P0;
//    P1 = handler.P1;


//    auto[depth, mask] = KITTI_VO::stereo2depth(img1_l, img1_r, P0, P1);
//
//    cv::Mat lidar = KITTI_VO::pointCloud2Image(cloud, handler.frame_height, handler.frame_width, handler.Rt, P0);
//    KITTI_VO::decomposeProjectionMatrix(P0, K, R, t);
//    KITTI_VO::extractFeatures(img1_l, kp1, desc1, mask, KITTI_VO::DETECTOR::sift);
//    KITTI_VO::extractFeatures(img2, kp2, desc2, mask, KITTI_VO::DETECTOR::sift);
//    KITTI_VO::matchFeatures(desc1, desc2, matches, KITTI_VO::DETECTOR::sift, KITTI_VO::MATHCHER::BF);
//    cv::Mat R_, t_;
//    KITTI_VO::estimateMotion(matches, kp1, kp2, K, R_, t_, pt1, pt2, depth);
//
//
//    cout << "main\n";
//    cout << R_ << endl << endl << t_ << endl;

    cout << "exiting successfully" << endl;
	return 0;
}
