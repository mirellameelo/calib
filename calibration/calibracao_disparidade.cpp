#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){

	Mat im = imread("c:/full/path/to/lena.jpg");
	if (im.empty()){

		cout << "Cannot load image!" << endl;
		return -1;

	}

	imshow("Image", im);
	waitKey(0);

}