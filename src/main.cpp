#include <iostream>
#include <vector>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>

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

    cout << "exiting successfully" << endl;
	return 0;
}
