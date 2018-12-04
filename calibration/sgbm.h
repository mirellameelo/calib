#ifndef __SGBM_H__
#define __SGBM_H__

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <emmintrin.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <climits>
//#include <pthread.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define DEBUG false
#define MAX_SHORT std::numeric_limits<unsigned short>::max()
#define PATHS_PER_SCAN 5

struct path {
    short rowDiff;
    short colDiff;
    short index;
};

class Sgbm {
  
private:
  int height;
  int width;
  int downScaleParam;
  int dispRange;
  int CLRDispMax; 
  int minDiff; 
  int filterLimitValue;
  int wradiusY; 
  int wradiusX; 
  int P1; 
  int P2;
  int dheight;
  int dwidth;
  int ddispRange;
  Mat downFirstImage, downSecondImage;
  Mat firstImage_grad_x, secondImage_grad_x;  
  int first_image_temp_grad_x, second_image_temp_grad_x;
  int first_image_temp_grad_y, second_image_temp_grad_y;
  Mat disparityMapDb, disparityMapDm;
  Mat disparityMap;
  int casoBase;
  

  unsigned short **PFI;    // pre-processed first image W x H
  unsigned short **PSE;    // pre-processed second image W x H    
  unsigned short ***C;     // pixel cost array W x H x D
  unsigned short ***S;     // aggregated cost array W x H x D
  unsigned short ***minA;    // single path cost array P W x H
  unsigned short ****A;    // single path cost array P x W x H x D 

  int xGradient(Mat image, int x, int y);
  int yGradient(Mat image, int x, int y);
  unsigned short calculatePixelCostOneWayBT(int row, int leftCol, int rightCol, const cv::Mat &leftImage, const cv::Mat &rightImage);
  unsigned short calculatePixelCostBT(int row, int leftCol, int rightCol, const cv::Mat &leftImage, const cv::Mat &rightImage);
  void calcSadCost(Mat *imageL, Mat *imageR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x);
  void calcSadCostR(Mat *imageL, Mat *imageR, Mat *imageLR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x);
  void calcSadCostv2(Mat *imageL, Mat *imageR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x);
  void initializeFirstScanPaths(std::vector<path> &paths, unsigned short pathCount);
  void initializeSecondScanPaths(std::vector<path> &paths, unsigned short pathCount);
  void setInitialCondition(Mat &firstImage, Mat &secondImage, int dispRange);
  void preProcess(Mat &firstImage, Mat &secondImage, Mat &firstImage_grad_x, Mat &second_image_temp_grad_x);
  unsigned short aggregateCost(int row, int col, int d, path &p, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ***A, int P1, int P2); 
  unsigned short aggregateCost(int row, int col, int d, path &p, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ***A, unsigned short **minA, int P1, int P2);  
  void aggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, int P1, int P2);
  void aggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2);
  
  void computeDisparity(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat *disparityMap);
  void computeDisparity(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat &disparityMap);
  void removeInconsistency(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat *disparityMap, int minimalDifference);
  void removeInconsistency(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat &disparityMap, int minimalDifference);
  //void removeInconsistency(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat *disparityMap, int minimalDifference);
  void computeDisparityDm(unsigned short ***S, cv::Mat *disparityMapInput, int rows, int cols, int disparityRange, cv::Mat *disparityMap);
  void computeDisparityDm(unsigned short ***S, cv::Mat &disparityMapInput, int rows, int cols, int disparityRange, cv::Mat &disparityMap);
  //void computeDisparityDm(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat &disparityMap);
  void leftRightCheck(Mat *inputImage, Mat *inputImage2, Mat *imageResult, int dispMax);
//  void leftRightCheck(Mat &inputImage, Mat &inputImage2, Mat *imageResult, int dispMax);
  //void leftRightCheck(Mat &inputImage, Mat &inputImage2, Mat &imageResult, int dispMax);
  
  void removeInconsistencyv2(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat &disparityMap, int minimalDifference);
  
  void leftRightCheck(Mat &inputImage, Mat &inputImage2, Mat &imageResult, int dispMax, int dispTotal);
  void subPixelEnhancement(unsigned short ***S, Mat *D, int disparityRange, int fator);
  void subPixelEnhancement(unsigned short ***S, Mat &D, int disparityRange, int fator);
  
  void resizeI(Mat *in, Mat *out, int fator);
  void resizeD(Mat *in, Mat *out, int fator);
  void findTexturelessArea(Mat &inputImage, int wradius_y, int wradius_x);
  
  void calcSadCostv4(Mat *imageL, Mat *imageR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x);
  
  void preProcess2(Mat &firstImage, Mat &secondImage, Mat &firstImage_grad_x, Mat &secondImage_grad_x, int wradius_y, int wradius_x);
  
  //void calcCostPlusAggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2);
  
  unsigned short calculatePixelCostOneWayBTv2(int row, int leftCol, int rightCol, const cv::Mat &leftImage, const cv::Mat &rightImage);
  
  //void setInitialConditionv2(Mat &firstImage, Mat &secondImage, Mat &secondImage, Mat &firstImage_grad_x, int dispRange);
  
  void setInitialConditionv2(Mat &firstImage, Mat &secondImage, Mat &firstImage_grad_x, Mat &secondImage_grad_x, int dispRange);
  
  void calcCostPlusAggregateCostsv3(Mat *imageL, Mat *imageR, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2);
  
  //void calcCostPlusAggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2, int wradius_y, int wradius_x);
  
  //void calcCostPlusAggregateCosts(Mat *imageL, Mat *imageR, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2, int wradius_y, int wradius_x);
  
  unsigned short calculatePixelCostBTv2(int row, int leftCol, int rightCol, const cv::Mat &leftImage, const cv::Mat &rightImage);
  
  void calcSadCostv3(Mat *imageL, Mat *imageR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x);
  
  void aggregateCostsv3(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2);
  
  int xGradientv2(Mat image, int x, int y);
  
  void calcCostPlusAggregateCosts(Mat *imageL, Mat *imageR, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2, int wradius_y, int wradius_x, cv::Mat &disparityMap);
   
  void calcCostPlusAggregateCostsv2(Mat *imageL, Mat *imageR, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2, int wradius_y, int wradius_x, cv::Mat &disparityMap);
public:
  Sgbm();
  Sgbm(int height, int width, int downScaleParam, int dispRange, int CLRDispMax, int minDiff, int filterLimitValue, int wradiusY, int wradiusX, int P1, int P2);
  void run(Mat firstImage, Mat secondImage, Mat &disparityMap, int dispRange);
  void run(Mat firstImage, Mat secondImage, Mat &disparityMap, int dispRange, int CLRDispMax, int minDiff, int downScaleParam, int filterLimitValue, int wradiusY, int wradiusX, int P1, int P2);
  void run(Mat firstImage, Mat secondImage, int dispRange);
  void runWithDownScale(Mat &firstImage, Mat &secondImage, int dispRange, int CLRDispMax, int minDiff, int downScaleParam, int filterLimitValue, int wradiusY, int wradiusX, int P1, int P2, Mat &disparityMap);
  void runA(Mat &firstImage, Mat &secondImage, Mat &disparityMap);
  void runAB(Mat &firstImage, Mat &secondImage);
  void runC(Mat firstImage, Mat secondImage, int dispRange);
  
  
  
  
  Mat disparityMapOut;
  
};

#endif
