#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{


  Mat R1, R2, P1, P2, Q;
  Mat K1, K2, R,T;

  Mat D1, D2;
  Mat img1 = imread("/home/peptoni/Pictures/wenas/left/10.jpg", cv::IMREAD_COLOR);
  Mat img2 = imread("/home/peptoni/Pictures/wenas/right/10.jpg",cv::IMREAD_COLOR);
  cv::FileStorage fs1("/home/peptoni/Desktop/IMEDEA/distance/calib_files/stereo_cam.yml", cv::FileStorage::READ);
  fs1["K1"] >> K1;
  fs1["K2"] >> K2;
  fs1["D1"] >> D1;
  fs1["D2"] >> D2;
  fs1["R"] >> R;
  fs1["T"] >> T;

  fs1["R1"] >> R1;
  fs1["R2"] >> R2;
  fs1["P1"] >> P1;
  fs1["P2"] >> P2;
  fs1["Q"] >> Q;
  cv::Mat lmapx, lmapy, rmapx, rmapy;
  cv::Mat imgU1, imgU2;

  cv::initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32F, rmapx, rmapy);
  cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
  cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
  printf("%d %d\n", imgU1.cols, imgU1.rows);
  imwrite("left.jpg", imgU1);
  imwrite("right.jpg", imgU2);
  cv::imshow("img1", imgU1);
  cv::imshow("img2", imgU2);
  cv::waitKey(0);
  return 0;
}
