#ifndef CORE_H
#define CORE_H

#include <iostream>
#include <string>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
//#include <dirent.h>
//#include <pthread.h>

//#include "individual.hpp"
//#include "elm.hpp"

using namespace std;
//using namespace cv;

class Point2D{
    public:
        Point2D();
        int x;
        int y;
        int isValid;
};

#endif

