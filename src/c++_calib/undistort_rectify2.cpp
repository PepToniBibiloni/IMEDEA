#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/stereo.hpp>
#include "popt_pp.h"
using namespace std;
using namespace cv;

// function calculate the disparity map
void calculateDisparityMap(Mat &left_img, Mat &right_img, Mat &disparity_map, int min_disparity, int num_disp, int block_size,int P1,int P2,int disp12MaxDiff,int pre_filter_cap, int uniqueness_ratio,int speckle_window_size,int speckle_range, int mode)
{
    // create the stereo matcher
    Ptr<StereoSGBM> bm = StereoSGBM::create(
        min_disparity,
        num_disp,
        block_size,
        P1,
        P2,
        disp12MaxDiff,
        pre_filter_cap,
        uniqueness_ratio,
        speckle_window_size,
        speckle_range,
        mode);
      
    /*  max_disparity, block_size);
    bm->setPreFilterCap(pre_filter_cap);
    bm->setSpeckleWindowSize(speckle_window_size);
    bm->setSpeckleRange(speckle_range);
    bm->setUniquenessRatio(uniqueness_ratio);
    bm->setSpeckleWindowSize(speckle_window_size);
    bm->setSpeckleRange(speckle_range);
    bm->setMode(mode);*/

    // calculate the disparity map
    bm->compute(left_img, right_img, disparity_map);

    // filter the disparity map
    Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(bm);
    wls_filter->setLambda(80000);
    wls_filter->setSigmaColor(1.5);
    wls_filter->filter(left_img, right_img, disparity_map, disparity_map);

    // normalize the disparity map
    normalize(disparity_map, disparity_map, 0, 255, NORM_MINMAX, CV_8UC1);
    // scale the disparity map
   // disparity_map.convertTo(disparity_map, CV_8U, -disp_scale*16.0/num_disp);

}



int main(int argc, char const *argv[])
{
  char* leftimg_filename;
  char* rightimg_filename;
  char* calib_file;
  char* leftout_filename;
  char* rightout_filename;

  static struct poptOption options[] = {
    { "leftimg_filename",'l',POPT_ARG_STRING,&leftimg_filename,0,"Left imgage path","STR" },
    { "rightimg_filename",'r',POPT_ARG_STRING,&rightimg_filename,0,"Right image path","STR" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file,0,"Stereo calibration file","STR" },
    { "leftout_filename",'L',POPT_ARG_STRING,&leftout_filename,0,"Left undistorted imgage path","STR" },
    { "rightout_filename",'R',POPT_ARG_STRING,&rightout_filename,0,"Right undistorted image path","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };
  printf("Starting Calibration\n");
  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

  Mat R1, R2, P1, P2, Q;
  Mat K1, K2, R;
  Vec3d T;
  Mat D1, D2;

  cv::FileStorage fs1("stereo_cam.yml", cv::FileStorage::READ);
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
  printf("Calculating undistort maps\n");
  VideoCapture left(leftimg_filename);
  VideoCapture right(rightimg_filename);

  Size frameSize(static_cast<int>(1280), static_cast<int>(720));
  auto writerL = cv::VideoWriter("left.avi", VideoWriter::fourcc('M','J','P','G'), 24,frameSize, true);
  auto writerR = cv::VideoWriter("right.avi", VideoWriter::fourcc('M','J','P','G'), 24,frameSize, true);
  
  if(!left.isOpened() || !right.isOpened())
  {
    printf("Error opening video stream or file\n");
    return -1;
  }
  int count = 0;
  while (1) {
    Mat imgL, imgR;
    left >> imgL;
    right >> imgR;
    if (imgL.empty() || imgR.empty()) {
      printf("End of images\n");
      break;
    }
    cv::initUndistortRectifyMap(K1, D1, R1, P1, imgL.size(), CV_32F, lmapx, lmapy);
    cv::remap(imgL, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, imgR.size(), CV_32F, rmapx, rmapy);
    cv::remap(imgR, imgU2, rmapx, rmapy, cv::INTER_LINEAR); 
    
    // grayscale for disparity calculation
    cv::Mat imgL_gray, imgR_gray;
    cv::cvtColor(imgU1, imgL_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgU2, imgR_gray, cv::COLOR_BGR2GRAY);

    Mat imgDisp;

    calculateDisparityMap(imgL_gray, imgR_gray, imgDisp,
      /*64, //Max disparity
      5, //Block size 
      -1, //Disparity scale
      0,  // Minimum disparity 
      64, // Number of disparities
      3,  // Pre-filter cap
      15, // Speckle window size
      1, // Speckle range
      0, // Uniqueness ratio
      0, //Speckle filter
      cv::StereoSGBM::MODE_SGBM // StereoSGBM::MODE_HH*/
      /*192, //Max disparity
      3, //Block size 
      -1, //Disparity scale
      -1,  // Minimum disparity 
      80, // Number of disparities
      3,  // Pre-filter cap
      50, // Speckle window size
      32, // Speckle range
      10, // Uniqueness ratio
      0, //Speckle filter

      */

      0, //min disparity
      16, //num_disp
      11, //block_size
      0, //P1 
      0, // P2
      0, //disp12MaxDiff
      15, //pre_filter_cap
      6, //uniqueness_ratio
      100, //speckle_window_size
      2, //speckle_range
      cv::StereoSGBM::MODE_HH // StereoSGBM::MODE_HH
    );

    writerD << imgDisp;
    count++;
    writerL << imgU1;
    writerR << imgU2;
    cv::imshow("Disparity", imgDisp);
    cv::imshow("left", imgU1);
    cv::imshow("right", imgU2);
    cv::waitKey(30);
  }
  writerD.release();
  writerL.release();
  writerR.release();
  return 0;
}
