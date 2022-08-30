#include <algorithm>
#include <cassert>

#include <opencv2/calib3d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <pangolin/pangolin.h>

#include <spdlog/spdlog.h>

#include "frame.h"
#include "kitti_vo.hpp"

using std::cout;
using std::endl;
using namespace spdlog;

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
            0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY
        );
        matcher->compute(img_left, img_right, disp_left);
    }

    disp_left.convertTo(disp_left, CV_32F);
    disp_left /= 16.f;

    return disp_left; // IMPORTANT divide by 16
}

void KITTI_VO::decomposeProjectionMatrix(const cv::Mat &P, cv::Mat &K, cv::Mat &R, cv::Mat &t)
{
    cv::decomposeProjectionMatrix(P, K, R, t);
    t /= t.at<float>(3);
}

cv::Mat KITTI_VO::calculateDepthMap(const cv::Mat &disp_left, const cv::Mat &K, const cv::Mat &t_left, const cv::Mat &t_right, bool rectified)
{
    /*
     *  Z(depth) = (f*b)/disparity
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
    //double minVal, maxVal;
    //cv::minMaxLoc(depth, &minVal, &maxVal);

    // first non_max index on x-axis
    int x_loc;
    float *p = depth.ptr<float>(depth.rows/2);
    for (int i=0; i<depth.cols; i++)
    {
        if (p[i] < maxVal)
        {
            x_loc = i;
            break;
        }
    }

    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8U);
    mask(cv::Rect( x_loc, 0, depth.cols-x_loc, depth.rows)) = 255;

    return {depth, mask};
}

void KITTI_VO::extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> &kpts,
                               cv::OutputArray &desc,
                               cv::InputArray mask,
                               DETECTOR detector)
{
    if (detector == DETECTOR::orb)
    {
        auto orb = cv::ORB::create();
        orb->detectAndCompute(img, mask, kpts, desc);
    }
    else if (detector == DETECTOR::sift)
    {
        auto sift = cv::SIFT::create();
        sift->detectAndCompute(img, mask, kpts, desc);
    }
    else if (detector == DETECTOR::CUDA_orb)
    {
        cv::cuda::GpuMat d_img, d_mask, d_kpts, d_desc;
        d_img.upload(img.clone());
        d_mask.upload(mask);

        auto cuda_orb = cv::cuda::ORB::create();
        cv::cuda::Stream st;
        cuda_orb->detectAndComputeAsync(d_img, d_mask, d_kpts, desc.getGpuMatRef(), false, st);
        st.waitForCompletion();

        cuda_orb->convert(d_kpts, kpts);
    }
}


void KITTI_VO::matchFeatures(cv::OutputArray desc1, cv::OutputArray desc2, std::vector<cv::DMatch> &matches,
                             DETECTOR detector,
                             MATHCHER matcher_,
                             bool sort, int k)
{
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    if (matcher_ == MATHCHER::FLANN)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        matcher->knnMatch(desc1, desc2, knn_matches, k);
    }
    else if (matcher_ == MATHCHER::BF)
    {
        if (detector != DETECTOR::CUDA_orb)
        {
            if (detector == DETECTOR::sift)
                matcher = cv::BFMatcher::create(cv::NORM_L2);
            else if (detector == DETECTOR::orb)
                matcher = cv::BFMatcher::create(cv::NORM_HAMMING2);

            matcher->knnMatch(desc1, desc2, knn_matches, k);
        }
        else
        {
            cv::cuda::GpuMat d_matches;
            auto cuda_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
            cv::cuda::Stream st;
            cuda_matcher->knnMatchAsync(desc1.getGpuMatRef(), desc2.getGpuMatRef(), d_matches, k, cv::noArray(), st);
            st.waitForCompletion();
            cuda_matcher->knnMatchConvert(d_matches, knn_matches);
        }
    }

    // Filter good matches
    matches.reserve(knn_matches.size());
    for (const auto& m : knn_matches)
    {
        if (m.size() > 1 && m[0].distance < m[1].distance * 0.45)
           matches.push_back(m[0]);
    }
}

void KITTI_VO::visualizeMatches(const cv::Mat &img1, const std::vector<cv::KeyPoint> &kpts1,
                                const cv::Mat &img2, const std::vector<cv::KeyPoint> &kpts2,
                                const std::vector<cv::DMatch>& matches)
{
    cv::Mat img;
    cv::drawMatches(img1, kpts1, img2, kpts2, matches, img);

    cv::imshow("Matches", img);
    cv::waitKey();
    cv::destroyAllWindows();
}

cv::Mat KITTI_VO::pointCloud2Image(pcl_cloud_ptr cloud, size_t img_height, size_t img_width, const cv::Mat &Rt, const cv::Mat &P0)
{
    /// X-axis points forward. Hence, ignore x value less than or equal to 0.
    pcl_cloud_ptr filtered_cloud(new pcl_cloud);
    filtered_cloud->points.reserve(cloud->points.size());
    auto &filtered_points = filtered_cloud->points;

    for (int i=0; i<cloud->points.size(); ++i)
    {
        pcl_point& point = cloud->points[i];
        if (point.x > 0.f) filtered_points.push_back(point);
    }

    cv::Mat lidar_points;
    KITTIDataHandler::pcl2CVMat(filtered_cloud, lidar_points);
    cv::Mat ones = cv::Mat::ones(lidar_points.rows, 1, CV_32F);
    cv::hconcat(lidar_points, ones, lidar_points);
    lidar_points = lidar_points.t(); // transpose it for matrix multiplication

    // Transform pointcloud into Cam0 coordinate frame
    cv::Mat cam0_xyz= Rt * lidar_points;

    // Extract the Z row which is the depth from camera
    cv::Mat cam0_z = cam0_xyz.row(2).clone();

    // Normalize the coordinate
    for (int i=0; i<3; i++)
        cam0_xyz.row(i) /= cam0_z;

    // Add row of ones to make a homogeneous
    ones = cv::Mat(1, cam0_xyz.cols, CV_32F);
    cv::vconcat(cam0_xyz, ones, cam0_xyz);

    // Project onto Cam0's image plane
    cv::Mat cam0_img = P0 * cam0_xyz;

    cv::Mat render = cv::Mat::zeros(img_height, img_width, CV_32F);

    for (int i=0; i<cam0_img.cols; i++)
    {
        auto col = cam0_img.col(i);
        int c = col.at<float>(0);
        int r = col.at<float>(1);
        if (c >= 0 && c < img_width && r >= 0 && r < img_height)
            render.at<float>(r, c) = cam0_z.at<float>(i);
    }

    cv::parallel_for_(cv::Range(0, img_height*img_width), [&](const cv::Range& range){
        for (int i=range.start; i<range.end; ++i)
        {
            int r = i/img_width;
            int c = i - r*img_width;
            auto &p = render.at<float>(r, c);
            if (p == 0.f)
                p = 3861.45f;
        }
    });

    return render;
}


void KITTI_VO::estimateMotion(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::KeyPoint> &kp2,
                    const cv::Mat &K,
                    cv::Mat &R, cv::Mat &t, std::vector<cv::Point2f> &img1_points, std::vector<cv::Point2f> &img2_points,
                    cv::InputArray depth1, float max_depth)
{
    // Bug fixed
    if (!img1_points.empty()) img1_points.clear();
    if (!img2_points.empty()) img2_points.clear();
    img1_points.reserve(matches.size());
    img2_points.reserve(matches.size());
    for (const auto &m : matches)
    {
        img1_points.push_back(kp1[m.queryIdx].pt);
        img2_points.push_back(kp2[m.trainIdx].pt);
    }

    if (!depth1.empty())
    {
        cv::Mat depth = depth1.getMat();
        float cx = K.at<float>(0, 2);
        float cy = K.at<float>(1, 2);
        float fx = K.at<float>(0, 0);
        float fy = K.at<float>(1, 1);

        cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

        std::vector<cv::Point3f> obj_points;
        std::vector<cv::Point2f> img_points;
        obj_points.reserve(img1_points.size());
        img_points.reserve(img1_points.size());

        for (int i=0; i<img1_points.size(); ++i)
        {
            auto[r, c] = std::tuple(img1_points[i].y, img1_points[i].x);

            if (r < 0 || r >= depth.rows || c < 0 || c >= depth.cols) continue;

            float z = depth.at<float>(r, c);

            if (z >= max_depth || z <= FLT_EPSILON) continue;

            // Normalized image plane (3D camera coordinate when z = 0)
            float x = z*(c-cx)/fx;
            float y = z*(r-cy)/fy;
            obj_points.push_back({x, y, z});
            img_points.push_back(img2_points[i]); // Cam1 pixel coordinate
        }

        // Use solvePnP RANSAC to get Rt(Cam0 coordinate to Cam1 coordinate)
        cv::solvePnPRansac(obj_points, img_points, K, distCoeffs, R, t);
        cv::Rodrigues(R, R);
    }
    else
    {
        cv::Mat E = cv::findEssentialMat(img1_points, img2_points, K);
        cv::recoverPose(E, img1_points, img2_points, R, t);
        cv::Rodrigues(R, R);
    }
}

void KITTI_VO::cvMat2pcl(const cv::Mat &lidar_points, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    cloud->width = lidar_points.rows;
    cloud->height = 1;
    cloud->points.reserve(cloud->width * cloud->height);
    auto &points = cloud->points;
    for (int i=0; i<lidar_points.rows; ++i)
    {
        const float *p = lidar_points.ptr<float>(i);
        pcl::PointXYZ point;
        point.x = p[0];
        point.y = p[1];
        point.z = p[2];
        points.push_back(point);
    }
}

std::tuple<cv::Mat, cv::Mat> KITTI_VO::inverseTransformation(cv::InputArray R, cv::InputArray t)
{
    cv::Mat R_inv = R.getMat().t();
    cv::Mat t_inv = -(R_inv * t.getMat());

    return {R_inv, t_inv};
}

cv::Mat KITTI_VO::inverseTransformation(cv::InputArray Rt)
{
    cv::Mat Rt_mat = Rt.getMat();
    cv::Mat R_inv = Rt_mat(cv::Rect(0, 0, 3, 3));
    cv::Mat t_inv = Rt_mat(cv::Rect(3, 0, 1, 3));

    R_inv = R_inv.t();
    t_inv = -(R_inv * t_inv);

    return Rt_mat;
}

float KITTI_VO::calculateDepthError(const cv::Mat &depth, const cv::Mat &lidar, bool verbose)
{
    float mean = 0.f;
    int cnt = 0;
    for (int r=0; r<depth.rows; r++)
    {
        for (int c=0; c<depth.cols; c++)
        {
            float z_depth = depth.at<float>(r, c);
            float z_lidar = lidar.at<float>(r, c);
            if (z_depth >= 0 && z_depth < 3000 && z_lidar >= 0 && z_lidar <= 3000)
            {
                if (verbose)
                {
                    std::cout << "at [" << r << ", " << c << "]\t\tdepth map: " << z_depth << "\tlidar: " << z_lidar << endl;
                }
                mean += abs(z_depth - z_lidar);
                cnt++;
            }
        }
    }

    return mean/static_cast<float>(cnt);
}


double KITTI_VO::calculatePoseError(const std::vector<cv::Mat> &estimate, const std::vector<cv::Mat> &gt)
{
    assert(estimate.size() == gt.size());
    double err = 0.0;
    cv::Mat diff;
    for (int i=0; i<gt.size(); i++)
    {
        cv::absdiff(estimate[i], gt[i], diff);
        err += (cv::sum(diff)[0])/12.;
    }
    return err / static_cast<double>(gt.size());
}


void KITTI_VO::visualOdometry(KITTIDataHandler &handler,
                              METHOD method,
                              STEREO_MATHCER stereo_matcher,
                              DETECTOR detector,
                              MATHCHER matcher,
                              std::vector<cv::Mat>& trajectory)
{
    bool vis = true;

    size_t num_frames = handler.num_frames;
    size_t width = handler.frame_width;
    size_t height = handler.frame_height;
    cv::Mat P0 = handler.P0;
    cv::Mat P1 = handler.P1;
    cv::Mat Rt = handler.Rt;
    std::vector<std::string> &left_images = handler.left_image_files;
    std::vector<std::string> &right_images = handler.right_image_files;

    // Decompose projection matrix to get intrinsic matrix K
    cv::Mat K, R_left, t_left;
    decomposeProjectionMatrix(P0, K, R_left, t_left);

    trajectory.reserve(num_frames);

    // Establish homogeneous transformation matrix. First pose is identity.
    cv::Mat T_tot = cv::Mat::eye(4, 4, CV_64F);
    trajectory.emplace_back(T_tot(cv::Rect(0, 0, 4, 3)).clone());

    // Initialize circular buffer
    std::vector<std::shared_ptr<Frame>> ring;
    ring.reserve(2);

    cv::namedWindow("lidar");
    cv::namedWindow("img_l");
    cv::namedWindow("depth");
    if (!vis) cv::destroyAllWindows();

    for (int i=0; i<num_frames; ++i)
    {
        info("{}_th frame is being processed({}/{})", i, i, num_frames-1);
        std::shared_ptr<Frame> new_frame = std::make_shared<Frame>();
        new_frame->num = i;
        new_frame->img_l = cv::imread(left_images[i], 0);
        new_frame->img_r = cv::imread(right_images[i], 0);

        // get depth image
        cv::Mat depth, mask;
        if (method == METHOD::stereo)
        {
            std::tie(new_frame->depth, new_frame->mask) =
                    stereo2depth(new_frame->img_l, new_frame->img_r, P0, P1, stereo_matcher);
            if (vis)
            {
                cv::Mat display;
                new_frame->depth.convertTo(display, CV_8U);
                cv::imshow("depth", display);
                cv::imshow("img_l", new_frame->img_l);
                cv::waitKey(100);
            }
        }
        else if (method == METHOD::lidar)
        {
            pcl_cloud_ptr cloud = KITTIDataHandler::loadLidarPoints(handler.ptcloud_files[i]);
            new_frame->depth = pointCloud2Image(cloud, height, width, Rt, P0);

            if (vis)
            {
                cv::Mat display;
                new_frame->depth.convertTo(display, CV_8U);
                cv::imshow("lidar", display);
                cv::imshow("img_l", new_frame->img_l);
                cv::waitKey(100);
            }
        }

        // Get keypoints and descriptors for two sequential frames
        if (detector != DETECTOR::CUDA_orb)
            extractFeatures(new_frame->img_l, new_frame->kp, new_frame->desc, new_frame->mask, detector);
        else
            extractFeatures(new_frame->img_l, new_frame->kp, new_frame->d_desc, new_frame->mask, detector);
        ring.push_back(new_frame);

        // Only proceeds when buffer has 2 frames
        if (ring.size() < 2) continue;

        // Get matches
        std::vector<cv::DMatch> matches;
        if (detector != DETECTOR::CUDA_orb)
            matchFeatures(ring[0]->desc, ring[1]->desc, matches, detector, matcher);
        else
            matchFeatures(ring[0]->d_desc, ring[1]->d_desc, matches, detector, matcher);

        // Estimate motion between two frames i'th frame -> (i+1)'th frame,
        cv::Mat R, t;
        estimateMotion(matches, ring[0]->kp, ring[1]->kp, K, R, t, ring[0]->points, ring[1]->points, ring[0]->depth);

        cv::Mat Tmat = cv::Mat::eye({4, 4}, CV_64F);
        R.copyTo(Tmat(cv::Rect(0, 0, 3, 3)));
        t.copyTo(Tmat(cv::Rect(3, 0, 1, 3)));

        // Track trajectory
        T_tot = T_tot * inverseTransformation(Tmat);
        trajectory.emplace_back(std::move(T_tot(cv::Rect(0, 0, 4, 3)).clone()));

        // Circular buffer
        if (ring.size() == 2)
        {
            std::swap(ring[0], ring.back());
            ring.back().reset();
            ring.pop_back();
        }
    }
    if (vis)
        cv::destroyAllWindows();
}

void KITTI_VO::saveTrajectory(const std::vector<cv::Mat>& trajectory, const std::string &filename)
{
     std::ofstream out(filename);
     if (!out.is_open())
     {
         std::cerr << "can't open the file to write" << std::endl;
         return;
     }

     for(const auto &P : trajectory)
     {
         out << P.at<double>(0, 0) << ' ' << P.at<double>(0, 1) << ' ' << P.at<double>(0, 2) << ' ' << P.at<double>(0, 3) << ' '
             << P.at<double>(1, 0) << ' ' << P.at<double>(1, 1) << ' ' << P.at<double>(1, 2) << ' ' << P.at<double>(1, 3) << ' '
             << P.at<double>(2, 0) << ' ' << P.at<double>(2, 1) << ' ' << P.at<double>(2, 2) << ' ' << P.at<double>(2, 3) << '\n';
     }
     out.close();
}

void KITTI_VO::drawTrajectory(const std::vector<cv::Mat>& trajectory)
{
    // Create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 1024);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.f, 1.f, 1.f, 1.f);
        glLineWidth(2);

        cv::Mat x_hat = cv::Mat_<double>({3, 1}, {1, 0, 0});
        cv::Mat y_hat = cv::Mat_<double>({3, 1}, {0, 1, 0});
        cv::Mat z_hat = cv::Mat_<double>({3, 1}, {0, 0, 1});

        for (const auto& P : trajectory)
        {
            cv::Mat R = P(cv::Rect(0, 0, 3, 3));
            cv::Mat t = P.col(3);

            // Projection of basis vectors(X, Y, Z) to world_frame
            cv::Mat Ow = t.clone();
            cv::Mat Xw = R * (0.1 * x_hat);
            cv::Mat Yw = R * (0.1 * y_hat);
            cv::Mat Zw = R * (0.1 * z_hat);

            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2));
            glVertex3d(Xw.at<double>(0), Xw.at<double>(1), Xw.at<double>(2));
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2));
            glVertex3d(Yw.at<double>(0), Yw.at<double>(1), Yw.at<double>(2));
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2));
            glVertex3d(Zw.at<double>(0), Zw.at<double>(1), Zw.at<double>(2));
            glEnd();
        }

        // Draw connection
        for (size_t i = 0; i< trajectory.size()-1; ++i)
        {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            cv::Mat t1, t2;
            t1 = trajectory[i].col(3);
            t2 = trajectory[i+1].col(3);
            glVertex3d(t1.at<double>(0), t1.at<double>(1), t1.at<double>(2));
            glVertex3d(t2.at<double>(0), t2.at<double>(1), t2.at<double>(2));
            glEnd();
        }
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

pcl::visualization::PCLVisualizer::Ptr KITTI_VO::customColourVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    return viewer;
}

pcl::visualization::PCLVisualizer::Ptr KITTI_VO::rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    viewer->setCameraPosition(0, 0, 50, 1, 0, 1);
    return viewer;
}
