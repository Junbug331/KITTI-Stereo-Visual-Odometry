#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include "KITTI_datahandler.hpp"
#include "helper.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	fs::path ROOT(ROOT_DIR);

	string seq = "00";
	KITTIDataHandler handler(seq);
	cout << handler.ptcloud_files[0] << endl;
	pcl_cloud_ptr cloud = KITTIDataHandler::loadLidarPoints(handler.ptcloud_files[0]);

	cv::Mat lidar_points;
	KITTIDataHandler::pcl2CVMat(cloud, lidar_points);
	lidar_points = lidar_points.t();

    auto[depth, mask] = KITTI_VO::stereo2depth(handler.first_image_left,
                                               handler.first_image_right,
                                               handler.P0,
                                               handler.P1);
    cv::Mat depth_display;
    depth.convertTo(depth_display, CV_8U);
    imshow("left", handler.first_image_left);
    imshow("mask", mask);
    imshow("depth_map", depth_display);
    waitKey();
    destroyAllWindows();


	return 0;
}
