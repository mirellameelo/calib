#include "sgbm.h"

Sgbm::Sgbm(){

}

Sgbm::Sgbm(int height, int width, int downScaleParam, int dispRange, int CLRDispMax, int minDiff, int filterLimitValue, int wradiusY, int wradiusX, int P1, int P2){
  this->downScaleParam = downScaleParam;
  this->dispRange = dispRange;
  this->CLRDispMax = CLRDispMax; 
  this->minDiff = minDiff; 
  this->filterLimitValue = filterLimitValue;
  this->wradiusY = wradiusY; 
  this->wradiusX = wradiusX; 
  this->P1 = P1; 
  this->P2 = P2;
  this->height = height;
  this->width = width;
  this->dheight = this->height/this->downScaleParam;
  this->dwidth = this->width/this->downScaleParam;
  this->ddispRange = this->dispRange/this->downScaleParam;
  this->casoBase = 1;
  
  PFI = new unsigned short*[this->dheight];
  PSE = new unsigned short*[this->dheight];	 
	 
	// allocate cost arrays
  C = new unsigned short**[this->dheight];
  S = new unsigned short**[this->dheight];
  
  
  for (int row = 0; row < this->dheight; ++row) {
      C[row] = new unsigned short*[this->dwidth];
      S[row] = new unsigned short*[this->dwidth]();
      //minA[row] = new unsigned short[this->dwidth]();
      PFI[row] = new unsigned short[this->dwidth]();
      PSE[row] = new unsigned short[this->dwidth]();
      for (int col = 0; col < this->dwidth; ++col) {
          //minA[row][col] = MAX_SHORT;
          C[row][col] = new unsigned short[this->ddispRange];
          S[row][col] = new unsigned short[this->ddispRange](); // initialize to 0
      }
  }
  
  A = new unsigned short ***[PATHS_PER_SCAN];
  minA = new unsigned short**[PATHS_PER_SCAN];
  for(int path = 0; path < PATHS_PER_SCAN; ++path) {
      A[path] = new unsigned short **[this->dheight];
      minA[path] = new unsigned short *[this->dheight];
      for (int row = 0; row < this->dheight; ++row) {
          minA[path][row] = new unsigned short[this->dwidth];
          A[path][row] = new unsigned short*[this->dwidth];
          for (int col = 0; col < this->dwidth; ++col) {
              A[path][row][col] = new unsigned short[this->ddispRange];
              minA[path][row][col] = MAX_SHORT;
              for (unsigned int d = 0; d < this->ddispRange; ++d) {
                  A[path][row][col][d] = 0;
              }
          }
      }
  } 
  
  this->firstImage_grad_x = cv::Mat(cv::Mat::zeros(this->dheight, this->dwidth, CV_16UC1) );
  this->secondImage_grad_x = cv::Mat(cv::Mat::zeros(this->dheight, this->dwidth, CV_16UC1) );
  
  //this->disparityMapDb = cv::Mat(cv::Mat::zeros(this->dheight, this->dwidth, CV_8UC1) ); //Db
  this->disparityMapDb.create( this->dheight, this->dwidth, CV_8UC1);
  this->disparityMapDm.create( this->dheight, this->dwidth, CV_8UC1);
  
  //cout << this->dheight << endl;
  //cout << this->dwidth << endl;
  
  //this->disparityMap = M(this->dheight,this->dwidth, CV_8UC, Scalar(0,0,255));
  
  this->disparityMap.create( this->dheight, this->dwidth, CV_8UC1);
  
  //this->disparityMapDm = cv::Mat(cv::Mat::zeros(this->dheight, this->dwidth, CV_8UC1) );
}

int Sgbm::xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}

int Sgbm::yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y-1, x) +
                 image.at<uchar>(y-1, x+1) -
                  image.at<uchar>(y+1, x-1) -
                   2*image.at<uchar>(y+1, x) -
                    image.at<uchar>(y+1, x+1);
}

unsigned short Sgbm::calculatePixelCostOneWayBT(int row, int leftCol, int rightCol, const cv::Mat &leftImage, const cv::Mat &rightImage) {

    char leftValue, rightValue, beforeRightValue, afterRightValue, rightValueMinus, rightValuePlus, rightValueMin, rightValueMax;

    if(leftCol<0)
    	leftValue = 0;
    else
	    leftValue = leftImage.at<short>(row, leftCol);

	  if(rightCol<0)
		rightValue = 0;
	else
	    rightValue = rightImage.at<short>(row, rightCol);

    if (rightCol > 0) {
        beforeRightValue = rightImage.at<short>(row, rightCol - 1);
    } else {
        beforeRightValue = rightValue;
    }

	//std::cout << rightCol <<" " <<leftCol<< std::endl;   	
    if (rightCol + 1 < rightImage.cols && rightCol>0) {
        afterRightValue = rightImage.at<short>(row, rightCol + 1);
    } else {
        afterRightValue = rightValue;
    }

    rightValueMinus = round((rightValue + beforeRightValue) / 2.f);
    rightValuePlus = round((rightValue + afterRightValue) / 2.f);

    rightValueMin = std::min(rightValue, std::min(rightValueMinus, rightValuePlus));
    rightValueMax = std::max(rightValue, std::max(rightValueMinus, rightValuePlus));

    return std::max(0, std::max((leftValue - rightValueMax), (rightValueMin - leftValue)));
}

unsigned short Sgbm::calculatePixelCostBT(int row, int leftCol, int rightCol, const cv::Mat &leftImage, const cv::Mat &rightImage) {
    return std::min(calculatePixelCostOneWayBT(row, leftCol, rightCol, leftImage, rightImage),
        calculatePixelCostOneWayBT(row, rightCol, leftCol, rightImage, leftImage));
    //return calculatePixelCostOneWayBTv2(row, leftCol, rightCol, leftImage, rightImage);
}

unsigned short Sgbm::calculatePixelCostBTv2(int row, int leftCol, int rightCol, const cv::Mat &leftImage, const cv::Mat &rightImage) {

    char leftValue, rightValue, beforeRightValue, afterRightValue, rightValueMinus, rightValuePlus, rightValueMin, rightValueMax;
    char beforeLeftValue, afterLeftValue, leftValueMinus, leftValuePlus, leftValueMin, leftValueMax;
    char v1, v2;

    if(leftCol<0){
    	leftValue = 0;
    }	
    else{
	    leftValue = leftImage.at<short>(row, leftCol);
	  }

	  if(rightCol<0){
		  rightValue = 0;
		}
	  else{
	    rightValue = rightImage.at<short>(row, rightCol);
	  }

    if (rightCol > 0) {
        beforeRightValue = rightImage.at<short>(row, rightCol - 1);
    } else {
        beforeRightValue = rightValue;
    }
    
    if (leftCol > 0) {
        beforeLeftValue = leftImage.at<short>(row, leftCol - 1);
    } else {
        beforeLeftValue = leftValue;
    }

    if (rightCol + 1 < rightImage.cols && rightCol>0) {
        afterRightValue = rightImage.at<short>(row, rightCol + 1);
    } else {
        afterRightValue = rightValue;
    }
    
    if (leftCol + 1 < leftImage.cols && leftCol>0) {
        afterLeftValue = leftImage.at<short>(row, leftCol + 1);
    } else {
        afterLeftValue = leftValue;
    }
    
    rightValueMinus = round((rightValue + beforeRightValue) / 2.f);
    rightValuePlus = round((rightValue + afterRightValue) / 2.f);
    
    leftValueMinus = round((leftValue + beforeLeftValue) / 2.f);
    leftValuePlus = round((leftValue + afterLeftValue) / 2.f);

    rightValueMin = std::min(rightValue, std::min(rightValueMinus, rightValuePlus));
    rightValueMax = std::max(rightValue, std::max(rightValueMinus, rightValuePlus));
    
    leftValueMin = std::min(leftValue, std::min(leftValueMinus, leftValuePlus));
    leftValueMax = std::max(leftValue, std::max(leftValueMinus, leftValuePlus));
    
    v1 = std::max(0, std::max((leftValue - rightValueMax), (rightValueMin - leftValue)));
    v2 = std::max(0, std::max((rightValue - leftValueMax), (leftValueMin - rightValue)));
    
    return min(v1,v2);
}

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void Sgbm::calcSadCost(Mat *imageL, Mat *imageR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x){
	int d, row, col;
	int sum;
	int dx, dy;

	for (row = 1; row < (imageL->rows - 1); row++){
	
		for (col = 1; col < (imageL->cols - 1); col++){

			for (d = 0; d < dMax; d++){
			
			    sum = 0;
          			 
					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {
						  sum += calculatePixelCostBT(row + dy, col + dx, col + dx - d, *imageL, *imageR);						 						  
						}						 
					}
					
					//repeating data from penultimate line to last one (leading with edge conditions) 			  
				  if(row == 1) finalResult[row-1][col][d] = sum;
				  if(row == ((imageL->rows) - 2)) finalResult[row+1][col][d] = sum;
				  if(col == 1) finalResult[row][col-1][d] = sum;
				  if(col == ((imageL->cols) - 2)) finalResult[row][col+1][d] = sum;
				  
				  finalResult[row][col][d] = sum;			
			}
		}
	}
} 

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void Sgbm::calcSadCostv2(Mat *imageL, Mat *imageR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x){
	int d, row, col;
	int sum;
	int dx, dy;

	for (row = 0; row < (imageL->rows); row++){
	
		for (col = 0; col < (imageL->cols); col++){

			for (d = 0; d < dMax; d++){
			
			    sum = 0;
          			 
					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {
						  sum += calculatePixelCostBTv2(row + dy, col + dx, col + dx - d, *imageL, *imageR);						 						  
						}						 
					}
					
					//repeating data from penultimate line to last one (leading with edge conditions) 			  
				  if(row == 1) finalResult[row-1][col][d] = sum;
				  if(row == ((imageL->rows) - 2)) finalResult[row+1][col][d] = sum;
				  if(col == 1) finalResult[row][col-1][d] = sum;
				  if(col == ((imageL->cols) - 2)) finalResult[row][col+1][d] = sum;
				  
				  finalResult[row][col][d] = sum;			
			}
		}
	}
}

void Sgbm::calcSadCostv3(Mat *imageL, Mat *imageR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x){
	int d, row, col;
	int sum;
	int dx, dy;
	char leftValue, rightValue, beforeRightValue, afterRightValue, rightValueMinus, rightValuePlus, rightValueMin, rightValueMax;
  char beforeLeftValue, afterLeftValue, leftValueMinus, leftValuePlus, leftValueMin, leftValueMax;
  char v1, v2;
  int row2, leftCol, rightCol;// const cv::Mat &leftImage, const cv::Mat &rightImage;

	for (row = 0; row < (imageL->rows); row++){
	
		for (col = 0; col < (imageL->cols); col++){

			for (d = 0; d < (dMax); d++){
			
			    sum = 0;
          			 
					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {
						  //sum += calculatePixelCostBTv2(row + dy, col + dx, col + dx - d, *imageL, *imageR);	
						  row2 = row + dy;
						  leftCol = col + dx;
						  rightCol = col + dx - d;
						  					 						  
						  if(leftCol<0){
              	leftValue = 0;
              }	
              else{
	              leftValue = imageL->at<short>(row2, leftCol);
	            }

	            if(rightCol<0){
		            rightValue = 0;
		          }
	            else{
	              rightValue = imageR->at<short>(row2, rightCol);
	            }

              if (rightCol > 0) {
                  beforeRightValue = imageR->at<short>(row2, rightCol - 1);
              } else {
                  beforeRightValue = rightValue;
              }
              
              if (leftCol > 0) {
                  beforeLeftValue = imageL->at<short>(row2, leftCol - 1);
              } else {
                  beforeLeftValue = leftValue;
              }

              if (rightCol + 1 < imageR->cols && rightCol>0) {
                  afterRightValue = imageR->at<short>(row2, rightCol + 1);
              } else {
                  afterRightValue = rightValue;
              }
              
              if (leftCol + 1 < imageL->cols && leftCol>0) {
                  afterLeftValue = imageL->at<short>(row2, leftCol + 1);
              } else {
                  afterLeftValue = leftValue;
              }
              
              rightValueMinus = round((rightValue + beforeRightValue) / 2.f);
              rightValuePlus = round((rightValue + afterRightValue) / 2.f);
              
              leftValueMinus = round((leftValue + beforeLeftValue) / 2.f);
              leftValuePlus = round((leftValue + afterLeftValue) / 2.f);

              rightValueMin = std::min(rightValue, std::min(rightValueMinus, rightValuePlus));
              rightValueMax = std::max(rightValue, std::max(rightValueMinus, rightValuePlus));
              
              leftValueMin = std::min(leftValue, std::min(leftValueMinus, leftValuePlus));
              leftValueMax = std::max(leftValue, std::max(leftValueMinus, leftValuePlus));
              
              v1 = std::max(0, std::max((leftValue - rightValueMax), (rightValueMin - leftValue)));
              v2 = std::max(0, std::max((rightValue - leftValueMax), (leftValueMin - rightValue)));
              
              sum += min(v1,v2);
              
              
						}						 
					}
					
					//repeating data from penultimate line to last one (leading with edge conditions) 			  
				  if(row == 1) finalResult[row-1][col][d] = sum;
				  if(row == ((imageL->rows) - 2)) finalResult[row+1][col][d] = sum;
				  if(col == 1) finalResult[row][col-1][d] = sum;
				  if(col == ((imageL->cols) - 2)) finalResult[row][col+1][d] = sum;
				  
				  finalResult[row][col][d] = sum;			
			}
		}
	}
}

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void Sgbm::calcSadCostR(Mat *imageL, Mat *imageR, Mat *imageLR, unsigned short ***finalResult, int dMax, int wradius_y, int wradius_x){
	int d, row, col;
	int sum;
	int dx, dy;
	int interval = 30;
	int meio;

	for (row = 1; row < (imageL->rows - 1); row++){
	
		for (col = 1; col < (imageL->cols - 1); col++){
		
		  meio = imageLR->at<uchar>(row, col);
		  //cout << "meio: " << meio << endl;
		  if(meio == 255) meio = 0;

			for (d = meio-interval; d <= meio+interval; d++){
			
			  if(d >= 0 && d < dMax ){
			
			    sum = 0;
          			 
					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {
						  sum += calculatePixelCostBT(row + dy, col + dx, col + dx - d, *imageL, *imageR);						 						  
						}						 
					}					
					//repeating data from penultimate line to last one (leading with edge conditions) 			  
				  if(row == 1) finalResult[row-1][col][d] = sum;
				  if(row == ((imageL->rows) - 2)) finalResult[row+1][col][d] = sum;
				  if(col == 1) finalResult[row][col-1][d] = sum;
				  if(col == ((imageL->cols) - 2)) finalResult[row][col+1][d] = sum;
				  
				  finalResult[row][col][d] = sum;			
			  }
			}
		}
	}
}

// pathCount can be 1, 2, 4, or 8
void Sgbm::initializeFirstScanPaths(std::vector<path> &paths, unsigned short pathCount) {
    for (unsigned short i = 0; i < pathCount; ++i) {
        paths.push_back(path());
    }
 
    if(paths.size() >= 1) {
        paths[0].rowDiff = 0;
        paths[0].colDiff = -1;
        paths[0].index = 1;
    }

    if(paths.size() >= 2) {
        paths[1].rowDiff = -1;
        paths[1].colDiff = 0;
        paths[1].index = 2;
    }

    if(paths.size() >= 4) {
        paths[2].rowDiff = -1;
        paths[2].colDiff = 1;
        paths[2].index = 4;

        paths[3].rowDiff = -1;
        paths[3].colDiff = -1;
        paths[3].index = 7;
    }
    
    if(paths.size() >= 5) {
        paths[4].rowDiff = -1;
        paths[4].colDiff = 2;
        paths[4].index = 9;
    }

    if(paths.size() >= 8) {
        paths[4].rowDiff = -2;
        paths[4].colDiff = 1;
        paths[4].index = 8;

        paths[5].rowDiff = -2;
        paths[5].colDiff = -1;
        paths[5].index = 9;

        paths[6].rowDiff = -1;
        paths[6].colDiff = -2;
        paths[6].index = 13;

        paths[7].rowDiff = -1;
        paths[7].colDiff = 2;
        paths[7].index = 15;
    }
}

// pathCount can be 1, 2, 4, or 8
void Sgbm::initializeSecondScanPaths(std::vector<path> &paths, unsigned short pathCount) {
    for (unsigned short i = 0; i < pathCount; ++i) {
        paths.push_back(path());
    }

    if(paths.size() >= 1) {
        paths[0].rowDiff = 0;
        paths[0].colDiff = 1;
        paths[0].index = 0;
    }

    if(paths.size() >= 2) {
        paths[1].rowDiff = 1;
        paths[1].colDiff = 0;
        paths[1].index = 3;
    }

    if(paths.size() >= 4) {
        paths[2].rowDiff = 1;
        paths[2].colDiff = 1;
        paths[2].index = 5;

        paths[3].rowDiff = 1;
        paths[3].colDiff = -1;
        paths[3].index = 6;
    }

    if(paths.size() >= 8) {
        paths[4].rowDiff = 2;
        paths[4].colDiff = 1;
        paths[4].index = 10;

        paths[5].rowDiff = 2;
        paths[5].colDiff = -1;
        paths[5].index = 11;

        paths[6].rowDiff = 1;
        paths[6].colDiff = -2;
        paths[6].index = 12;

        paths[7].rowDiff = 1;
        paths[7].colDiff = 2;
        paths[7].index = 14;
    }
}

unsigned short Sgbm::aggregateCost(int row, int col, int d, path &p, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ***A, int P1, int P2) {
    unsigned short aggregatedCost = 0;
    
    aggregatedCost += C[row][col][d];

    if(DEBUG) {
        printf("{P%d}[%d][%d](d%d)\n", p.index, row, col, d);
    }   

    if (row + p.rowDiff < 0 || row + p.rowDiff >= rows || col + p.colDiff < 0 || col + p.colDiff >= cols) {
        A[row][col][d] += aggregatedCost;
        if(DEBUG) {
            printf("{P%d}[%d][%d](d%d)-> %d <BORDER>\n", p.index, row, col, d, A[row][col][d]);
        }
        return A[row][col][d];
    }

    unsigned short minPrev, minPrevOther, prev, prevPlus, prevMinus;
    prev = minPrev = minPrevOther = prevPlus = prevMinus = MAX_SHORT;
    
    for (int disp = 0; disp < disparityRange; ++disp) {
        unsigned short tmp = A[row + p.rowDiff][col + p.colDiff][disp];
        
        if(minPrev > tmp) {
            minPrev = tmp;
        }

        if(disp == d) {
            prev = tmp;
        } else if(disp == d + 1) {
            prevPlus = tmp;
        } else if (disp == d - 1) {
            prevMinus = tmp;
        }         
    }
    
    aggregatedCost += std::min(std::min((int)prevPlus + P1, (int)prevMinus + P1), std::min((int)prev, (int)minPrev + P2));
    aggregatedCost -= minPrev;

    A[row][col][d] += aggregatedCost;

    if(DEBUG) {
        printf("{P%d}[%d][%d](d%d)-> %d<CALCULATED>\n", p.index, row, col, d, A[row][col][d]);
    }
  
    return A[row][col][d];
}

unsigned short Sgbm::aggregateCost(int row, int col, int d, path &p, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ***A, unsigned short **minA, int P1, int P2) {
    unsigned short minPrev, minPrevOther, prev, prevPlus, prevMinus;
    prev = minPrev = minPrevOther = prevPlus = prevMinus = MAX_SHORT;

    if (row + p.rowDiff < 0 || row + p.rowDiff >= rows || col + p.colDiff < 0 || col + p.colDiff >= cols) {
        return 0;
    }
    else{   
    
      prev = A[row + p.rowDiff][col + p.colDiff][d];
      if((d + 1) < disparityRange) prevPlus = A[row + p.rowDiff][col + p.colDiff][d+1];
      if((d) >= 1) prevMinus = A[row + p.rowDiff][col + p.colDiff][d-1];
      minPrev = minA[row + p.rowDiff][col + p.colDiff];  
      
      A[row][col][d] += std::min(std::min((int)prevPlus + P1, (int)prevMinus + P1), std::min((int)prev, (int)minPrev + P2)) - minPrev;
      
      return A[row][col][d];
      
    }
  
    
}

void Sgbm::aggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, int P1, int P2) {
    std::vector<path> firstScanPaths;
    std::vector<path> secondScanPaths;

    initializeFirstScanPaths(firstScanPaths, PATHS_PER_SCAN);
    initializeSecondScanPaths(secondScanPaths, PATHS_PER_SCAN);

    int lastProgressPrinted = 0;
    //std::cout << "First scan..." << std::endl;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
          for (int d = 0; d < disparityRange; ++d) {
              
              for (unsigned int path = 0; path < firstScanPaths.size(); ++path) {
                
                S[row][col][d] += aggregateCost(row, col, d, firstScanPaths[path], rows, cols, disparityRange, C, A[path], P1, P2);                     
              }
              //if(S[row][col][d] < minA[row][col]) minA[row][col] = S[row][col][d];                            
           }
        }
    }

    //lastProgressPrinted = 0;
    //std::cout << "Second scan..." << std::endl;
    /*for (int row = rows - 1; row >= 0; --row) {
        for (int col = cols - 1; col >= 0; --col) {
            
                for (int d = 0; d < disparityRange; ++d) {
                for (unsigned int path = 0; path < secondScanPaths.size(); ++path) {
                    S[row][col][d] += aggregateCost(imageL, imageR, row, col, d, secondScanPaths[path], rows, cols, disparityRange, C, A[path]);
                }
                S[row][col][d] = round(S[row][col][d]/(firstScanPaths.size() * 2));
            }
             
        }
        //lastProgressPrinted = printProgress(rows - 1 - row, rows - 1, lastProgressPrinted);
    }*/
}

void Sgbm::aggregateCosts(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2) {
    std::vector<path> firstScanPaths;
    //std::vector<path> secondScanPaths;
    
    unsigned short minPrev, minPrevOther, prev, prevPlus, prevMinus;
    

    initializeFirstScanPaths(firstScanPaths, PATHS_PER_SCAN);
    //initializeSecondScanPaths(secondScanPaths, PATHS_PER_SCAN);

    //int lastProgressPrinted = 0;
    //std::cout << "First scan..." << std::endl;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
          for (int d = 0; d < disparityRange; ++d) {
          
              S[row][col][d] = 0;
              
              for (unsigned int pathu = 0; pathu < firstScanPaths.size(); ++pathu) { 
                
                A[pathu][row][col][d] = C[row][col][d];
                
                path p = firstScanPaths[pathu];
                
                prev = minPrev = minPrevOther = prevPlus = prevMinus = MAX_SHORT;
                
                if (row + p.rowDiff < 0 || row + p.rowDiff >= rows || col + p.colDiff < 0 || col + p.colDiff >= cols) {
                    //return 0;
                }
                else{
                
                   prev = A[pathu][row + p.rowDiff][col + p.colDiff][d];
                   
                   if((d + 1) < disparityRange) prevPlus = A[pathu][row + p.rowDiff][col + p.colDiff][d+1];
                   
                   if((d) >= 1) prevMinus = A[pathu][row + p.rowDiff][col + p.colDiff][d-1];
                   
                   minPrev = minA[pathu][row + p.rowDiff][col + p.colDiff];  
      
                   A[pathu][row][col][d] += std::min(std::min((int)prevPlus + P1, (int)prevMinus + P1), std::min((int)prev, (int)minPrev + P2)) - minPrev;
                   
                   S[row][col][d] += A[pathu][row][col][d];
                    
                                           
                   //S[row][col][d] += aggregateCost(row, col, d, firstScanPaths[path], rows, cols, disparityRange, C, A[path], minA[path], P1, P2); 
                   if(A[pathu][row][col][d] < minA[pathu][row][col]) minA[pathu][row][col] = A[pathu][row][col][d];
                }
              }              
           }
        }
    }

    //lastProgressPrinted = 0;
    //std::cout << "Second scan..." << std::endl;
    /*for (int row = rows - 1; row >= 0; --row) {
        for (int col = cols - 1; col >= 0; --col) {
            
                for (int d = 0; d < disparityRange; ++d) {
                    S[row][col][d] = C[row][col][d];
                for (unsigned int path = 0; path < secondScanPaths.size(); ++path) {
                    A[path][row][col][d] = C[row][col][d];     
                    S[row][col][d] += aggregateCost(row, col, d, secondScanPaths[path], rows, cols, disparityRange, C, A[path], minA[path], P1, P2); 
                    if(A[path][row][col][d] < minA[path][row][col]) minA[path][row][col] = A[path][row][col][d];
                }
                //S[row][col][d] = round(S[row][col][d]/(firstScanPaths.size() * 2));
            }
             
        }
        //lastProgressPrinted = printProgress(rows - 1 - row, rows - 1, lastProgressPrinted);
    }*/
}

void Sgbm::aggregateCostsv3(int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2) {
    std::vector<path> firstScanPaths;
    //std::vector<path> secondScanPaths;

    initializeFirstScanPaths(firstScanPaths, PATHS_PER_SCAN);
    //initializeSecondScanPaths(secondScanPaths, PATHS_PER_SCAN);

    //int lastProgressPrinted = 0;
    //std::cout << "First scan..." << std::endl;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
          for (int d = 0; d < disparityRange; ++d) {
          
              S[row][col][d] = C[row][col][d];
              
              for (unsigned int path = 0; path < firstScanPaths.size(); ++path) {   
                A[path][row][col][d] = C[row][col][d];                             
                S[row][col][d] += aggregateCost(row, col, d, firstScanPaths[path], rows, cols, disparityRange, C, A[path], minA[path], P1, P2); 
                if(A[path][row][col][d] < minA[path][row][col]) minA[path][row][col] = A[path][row][col][d];
              }
              
           }
        }
    }

    //lastProgressPrinted = 0;
    //std::cout << "Second scan..." << std::endl;
    /*for (int row = rows - 1; row >= 0; --row) {
        for (int col = cols - 1; col >= 0; --col) {
            
                for (int d = 0; d < disparityRange; ++d) {
                    S[row][col][d] = C[row][col][d];
                for (unsigned int path = 0; path < secondScanPaths.size(); ++path) {
                    A[path][row][col][d] = C[row][col][d];     
                    S[row][col][d] += aggregateCost(row, col, d, secondScanPaths[path], rows, cols, disparityRange, C, A[path], minA[path], P1, P2); 
                    if(A[path][row][col][d] < minA[path][row][col]) minA[path][row][col] = A[path][row][col][d];
                }
                //S[row][col][d] = round(S[row][col][d]/(firstScanPaths.size() * 2));
            }
             
        }
        //lastProgressPrinted = printProgress(rows - 1 - row, rows - 1, lastProgressPrinted);
    }*/
}

void Sgbm::calcCostPlusAggregateCosts(Mat *imageL, Mat *imageR, int rows, int cols, int disparityRange, unsigned short ***C, unsigned short ****A, unsigned short ***S, unsigned short ***minA, int P1, int P2, int wradius_y, int wradius_x) {
    int sum;
    int dy, dx;

    std::vector<path> firstScanPaths;
    std::vector<path> secondScanPaths;

    initializeFirstScanPaths(firstScanPaths, PATHS_PER_SCAN);
    //initializeSecondScanPaths(secondScanPaths, PATHS_PER_SCAN);

    //int lastProgressPrinted = 0;
    //std::cout << "First scan..." << std::endl;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
          for (int d = 0; d < disparityRange; ++d) {            
            sum = 0;         
             			 
					  for (dy = -wradius_y; dy <= wradius_y; dy++) {
						  for (dx = -wradius_x; dx <= wradius_x; dx++) {
						    sum += calculatePixelCostBT(row + dy, col + dx, col + dx - d, *imageL, *imageR);						 						  
						  }						 
					  }
					
					  //C[row][col][d] = sum;
          
            S[row][col][d] = sum;
            
            for (unsigned int path = 0; path < firstScanPaths.size(); ++path) {   
              A[path][row][col][d] = sum
              ;                             
              S[row][col][d] += aggregateCost(row, col, d, firstScanPaths[path], rows, cols, disparityRange, C, A[path], minA[path], P1, P2); 
              if(A[path][row][col][d] < minA[path][row][col]) minA[path][row][col] = A[path][row][col][d];
            }              
         }
      }
  }

    //lastProgressPrinted = 0;
    //std::cout << "Second scan..." << std::endl;
    /*for (int row = rows - 1; row >= 0; --row) {
        for (int col = cols - 1; col >= 0; --col) {
            
                for (int d = 0; d < disparityRange; ++d) {
                    S[row][col][d] = C[row][col][d];
                for (unsigned int path = 0; path < secondScanPaths.size(); ++path) {
                    A[path][row][col][d] = C[row][col][d];     
                    S[row][col][d] += aggregateCost(row, col, d, secondScanPaths[path], rows, cols, disparityRange, C, A[path], minA[path], P1, P2); 
                    if(A[path][row][col][d] < minA[path][row][col]) minA[path][row][col] = A[path][row][col][d];
                }
                //S[row][col][d] = round(S[row][col][d]/(firstScanPaths.size() * 2));
            }
             
        }
        //lastProgressPrinted = printProgress(rows - 1 - row, rows - 1, lastProgressPrinted);
    }*/
}

void Sgbm::computeDisparity(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat *disparityMap) {
    unsigned short disparity = 0, minCost;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            minCost = MAX_SHORT;
            disparity = 0;
            for (int d = 0; d < disparityRange; d++) {                
                if(minCost > S[row][col][d]) {
                    minCost = S[row][col][d];
                    disparity = d;                    
                }
            }
            disparityMap->at<uchar>(row, col) = disparity;
        }           
    }
}

void Sgbm::computeDisparity(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat &disparityMap) {
    unsigned short disparity = 0, minCost;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            minCost = MAX_SHORT;
            disparity = 0;
            for (int d = 0; d < disparityRange; d++) {                
                if(minCost > S[row][col][d]) {
                    minCost = S[row][col][d];
                    disparity = d;                    
                }
            }
            disparityMap.at<uchar>(row, col) = disparity;
        }           
    }
}

void Sgbm::removeInconsistency(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat *disparityMap, int minimalDifference) {
    unsigned short disparity = 0, minCost;
    unsigned short disparity2 = 0, minCost2;
    int condicao;
    
    //cout << "disparityRange: " << disparityRange << endl;
    //getchar();
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {          
            minCost = MAX_SHORT;
            disparity = 0;
            for (int d = 0; d < disparityRange; d++) {
            //cout << S[row][col][d] << endl;
                if(minCost > S[row][col][d]) {
                    minCost = S[row][col][d];
                    disparity = d;
                    
                }
            }
            condicao = 1;
            minCost2 = MAX_SHORT;
            disparity2 = 0;
            for (int d = 0; d < disparityRange && condicao == 1; d++) {
            
                if((((100 - minimalDifference) * S[row][col][d]) < minCost*100) && (abs(disparity - d) > 1)){
                  disparityMap->at<uchar>(row, col) = 255;
                  condicao = 0;
                }
                
                /*if(minCost2 > S[row][col][d] && d != disparity) {
                    minCost2 = S[row][col][d];
                    disparity2 = d;
                    
                }*/
                
            }
            /*cout << "1: " << disparity << endl;
            cout << "2: " << disparity2;
            getchar();*/
            
            
            /*else{
              disparityMap->at<uchar>(row, col) = disparity;
            }*/
            
         }            

    }
}

void Sgbm::removeInconsistency(unsigned short ***S, int rows, int cols, int disparityRange, cv::Mat &disparityMap, int minimalDifference) {
    unsigned short disparity = 0, minCost;
    unsigned short disparity2 = 0, minCost2;
    int condicao;
    
    //cout << "disparityRange: " << disparityRange << endl;
    //getchar();
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {          
            minCost = MAX_SHORT;
            disparity = 0;
            for (int d = 0; d < disparityRange; d++) {
            //cout << S[row][col][d] << endl;
                if(minCost > S[row][col][d]) {
                    minCost = S[row][col][d];
                    disparity = d;
                    
                }
            }
            condicao = 1;
            minCost2 = MAX_SHORT;
            disparity2 = 0;
            for (int d = 0; d < disparityRange && condicao == 1; d++) {
            
                if((((100 - minimalDifference) * S[row][col][d]) < minCost*100) && (abs(disparity - d) > 1)){
                  disparityMap.at<uchar>(row, col) = 255;
                  condicao = 0;
                }
                
                /*if(minCost2 > S[row][col][d] && d != disparity) {
                    minCost2 = S[row][col][d];
                    disparity2 = d;
                    
                }*/
                
            }
            /*cout << "1: " << disparity << endl;
            cout << "2: " << disparity2;
            getchar();*/
            
            
            /*else{
              disparityMap->at<uchar>(row, col) = disparity;
            }*/
            
         }            

    }
}

void Sgbm::computeDisparityDm(unsigned short ***S, cv::Mat *disparityMapInput, int rows, int cols, int disparityRange, cv::Mat *disparityMap) {
    unsigned short disparity = 0, minCost;
    unsigned short disparity2 = 0, minCost2;
    int colm;
    
    //cout << "disparityRange: " << disparityRange << endl;
    //getchar();
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            
            minCost = MAX_SHORT;
            disparity = 0;
            for (int d = 0; d < disparityRange; d++) {
                //colm = col - d;
                colm = col + d;
                //imageL->at<uchar>(row, col) - imageL->at<uchar>(row + p.rowDiff, col + p.colDiff)      
                if((colm < cols)) { 
                  if(disparityMapInput->at<uchar>(row, colm) != 255){
                    //disparityMap->at<uchar>(row, colm) = disparityMapInput->at<uchar>(row, col);
                    if(minCost > S[row][colm][d]){
                      minCost = S[row][colm][d];
                      disparity = d;
                    }
                    
                    }
                    
                }
                
                //else{
                  
                //}
            //}
            /*minCost2 = MAX_SHORT;
            disparity2 = 0;
            for (int d = 0; d < disparityRange; d++) {
                
                if(minCost2 > S[row][col][d] && d != disparity) {
                    minCost2 = S[row][col][d];
                    disparity2 = d;
                    
                }
            }
            if(abs(minCost2 - minCost) < 2){
              disparityMap->at<uchar>(row, col) = (disparity + disparity2)/2;
            }
            else{*/
            
            //}
            
            //cout << (unsigned short)disparityMap.at<uchar>(row, col)  << endl;
            //getchar();
        }
        disparityMap->at<uchar>(row, col) = disparity;
    //}
}
}
}

void Sgbm::computeDisparityDm(unsigned short ***S, cv::Mat &disparityMapInput, int rows, int cols, int disparityRange, cv::Mat &disparityMap) {
    unsigned short disparity = 0, minCost;
    unsigned short disparity2 = 0, minCost2;
    int colm;
    
    //cout << "disparityRange: " << disparityRange << endl;
    //getchar();
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            
            minCost = MAX_SHORT;
            disparity = 0;
            for (int d = 0; d < disparityRange; d++) {
                //colm = col - d;
                colm = col + d;
                //imageL->at<uchar>(row, col) - imageL->at<uchar>(row + p.rowDiff, col + p.colDiff)      
                if((colm < cols)) { 
                  if(disparityMapInput.at<uchar>(row, colm) != 255){
                    //disparityMap->at<uchar>(row, colm) = disparityMapInput->at<uchar>(row, col);
                    if(minCost > S[row][colm][d]){
                      minCost = S[row][colm][d];
                      disparity = d;
                    }
                    
                    }
                    
                }
                
                //else{
                  
                //}
            //}
            /*minCost2 = MAX_SHORT;
            disparity2 = 0;
            for (int d = 0; d < disparityRange; d++) {
                
                if(minCost2 > S[row][col][d] && d != disparity) {
                    minCost2 = S[row][col][d];
                    disparity2 = d;
                    
                }
            }
            if(abs(minCost2 - minCost) < 2){
              disparityMap->at<uchar>(row, col) = (disparity + disparity2)/2;
            }
            else{*/
            
            //}
            
            //cout << (unsigned short)disparityMap.at<uchar>(row, col)  << endl;
            //getchar();
        }
        disparityMap.at<uchar>(row, col) = disparity;
    //}
}
}
}


void Sgbm::leftRightCheck(Mat *inputImage, Mat *inputImage2, Mat *imageResult, int dispMax){
	int dMin = 0, dMax = inputImage->size[2];
	int d, row, col;
	int bestMatchSoFar;// = dMin;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532, corrScore;

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
  //*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_32FC1, float(0));

	//cout << "ops1" << endl;
	//getchar();

	for (row = 0; row < inputImage->rows; row++){
		for (col = 0; col < inputImage->cols; col++){
			/*if (mode1 == 0){
				prevcorrScore = 65532;
			}
			else{
				prevcorrScore = -64000;
			}*/
			//bestMatchSoFar = dMin;

			//cout << "row: " << row << "col: " << col << " "; 

			//CALCULO DE DISPARIDADE
			//for (d = dMin; d < dMax; d++){

				//cout << inputImage->at<float>(row, col, d) << " ";

			//cout << inputImage->at<float>(row, col) << " ";
			//cout << "inputImage2->at<float>(row + inputImage->at<float>(row, col), col): " << inputImage2->at<float>(row + inputImage->at<float>(row, col), col) << endl;
			//getchar();
			
			/*cout << (int)inputImage->at<uchar>(row, col) << endl;
			cout << (int)inputImage2->at<uchar>(row, col) << endl;  
			getchar();*/

			/*if (((col - inputImage->at<uchar>(row, col)) >= 0) && (abs(inputImage->at<uchar>(row, col) - inputImage2->at<uchar>(row, col - inputImage->at<uchar>(row, col))) > dispMax)){			  
				imageResult->at<uchar>(row, col) = 0;
			}
			else{
				imageResult->at<uchar>(row, col) = inputImage->at<uchar>(row, col);
				//cout << "ops" << endl;
			}*/
			
			if(inputImage->at<uchar>(row, col) == 255){
			  imageResult->at<uchar>(row, col) = inputImage->at<uchar>(row, col);
			}
			else{
			
			  if (((col - inputImage->at<uchar>(row, col)) >= 0) && (abs(inputImage->at<uchar>(row, col) - inputImage2->at<uchar>(row, col - inputImage->at<uchar>(row, col))) > dispMax)){			  
				  imageResult->at<uchar>(row, col) = 255;
			  }
			  else{
			    
				  imageResult->at<uchar>(row, col) = inputImage->at<uchar>(row, col);/*std::min(inputImage->at<uchar>(row, col), inputImage2->at<uchar>(row, col - inputImage->at<uchar>(row, col)))*/;
				  //cout << "ops" << endl;
			  }
			}

				//CUSTO DO MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
				/*if (mode2 == 0){
					corrScore = inputImage->at<float>(row, col, d);
				}
				else{
					corrScore = inputImage->at<float>(row + d, col, d);
				}*/

				//CALCULO DA DISPARIDADE. 
				//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
				/*if (mode1 == 0){
					if (prevcorrScore > corrScore) {
						prevcorrScore = corrScore;
						bestMatchSoFar = d;
					}
				}
				else{
					if (prevcorrScore < corrScore) {
						prevcorrScore = corrScore;
						bestMatchSoFar = d;
					}
				}*/

			//}

			//cout << endl;

			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			//imageResult->at<float>(row, col) = bestMatchSoFar;
		}
		//cout << endl;
	}
	//IMPRIMINDO A MATRIZ DE DISPARIDADE
	/*ofstream myfile;
	myfile.open("cplusplus.txt");
	for (int i = 0; i < (imageResult->rows); i++){
	for (int j = 0; j < (imageResult->cols); j++){
	myfile << (float)imageResult->at<float>(i, j) << " "; //endl;
	}
	myfile << endl;
	}
	myfile.close();*/
	//cout << "finalizou" << endl;
	//getchar();
}

void Sgbm::leftRightCheck(Mat &inputImage, Mat &inputImage2, Mat &imageResult, int dispMax){
	//int dMin = 0, dMax = inputImage.size[2];
	int row, col;
	//int bestMatchSoFar;// = dMin;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	//double prevcorrScore = 65532, corrScore;

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
  //*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_32FC1, float(0));

	//cout << "ops1" << endl;
	//getchar();

	for (row = 0; row < inputImage.rows; row++){
		for (col = 0; col < inputImage.cols; col++){
			/*if (mode1 == 0){
				prevcorrScore = 65532;
			}
			else{
				prevcorrScore = -64000;
			}*/
			//bestMatchSoFar = dMin;

			//cout << "row: " << row << "col: " << col << " "; 

			//CALCULO DE DISPARIDADE
			//for (d = dMin; d < dMax; d++){

				//cout << inputImage->at<float>(row, col, d) << " ";

			//cout << inputImage->at<float>(row, col) << " ";
			//cout << "inputImage2->at<float>(row + inputImage->at<float>(row, col), col): " << inputImage2->at<float>(row + inputImage->at<float>(row, col), col) << endl;
			//getchar();
			
			/*cout << (int)inputImage->at<uchar>(row, col) << endl;
			cout << (int)inputImage2->at<uchar>(row, col) << endl;  
			getchar();*/

			/*if (((col - inputImage->at<uchar>(row, col)) >= 0) && (abs(inputImage->at<uchar>(row, col) - inputImage2->at<uchar>(row, col - inputImage->at<uchar>(row, col))) > dispMax)){			  
				imageResult->at<uchar>(row, col) = 0;
			}
			else{
				imageResult->at<uchar>(row, col) = inputImage->at<uchar>(row, col);
				//cout << "ops" << endl;
			}*/
			
			if(inputImage.at<uchar>(row, col) == 255){
			  imageResult.at<uchar>(row, col) = inputImage.at<uchar>(row, col);
			}
			else{
			
			  if (((col - inputImage.at<uchar>(row, col)) >= 0) && (abs(inputImage.at<uchar>(row, col) - inputImage2.at<uchar>(row, col - inputImage.at<uchar>(row, col))) > dispMax)){			  
				  imageResult.at<uchar>(row, col) = 255;
			  }
			  else{
			    
				  imageResult.at<uchar>(row, col) = inputImage.at<uchar>(row, col);/*std::min(inputImage->at<uchar>(row, col), inputImage2->at<uchar>(row, col - inputImage->at<uchar>(row, col)))*/;
				  //cout << "ops" << endl;
			  }
			}

				//CUSTO DO MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
				/*if (mode2 == 0){
					corrScore = inputImage->at<float>(row, col, d);
				}
				else{
					corrScore = inputImage->at<float>(row + d, col, d);
				}*/

				//CALCULO DA DISPARIDADE. 
				//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
				/*if (mode1 == 0){
					if (prevcorrScore > corrScore) {
						prevcorrScore = corrScore;
						bestMatchSoFar = d;
					}
				}
				else{
					if (prevcorrScore < corrScore) {
						prevcorrScore = corrScore;
						bestMatchSoFar = d;
					}
				}*/

			//}

			//cout << endl;

			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			//imageResult->at<float>(row, col) = bestMatchSoFar;
		}
		//cout << endl;
	}
	//IMPRIMINDO A MATRIZ DE DISPARIDADE
	/*ofstream myfile;
	myfile.open("cplusplus.txt");
	for (int i = 0; i < (imageResult->rows); i++){
	for (int j = 0; j < (imageResult->cols); j++){
	myfile << (float)imageResult->at<float>(i, j) << " "; //endl;
	}
	myfile << endl;
	}
	myfile.close();*/
	//cout << "finalizou" << endl;
	//getchar();
}

void Sgbm::subPixelEnhancement(unsigned short ***S, Mat *D, int disparityRange, int fator){
  int row, col, d;
  int denom2;
  for (row = 0; row < (D->rows); row++){
		for (col = 0; col < (D->cols); col++){
		  if(D->at<uchar>(row, col) != 255){
		    d = D->at<uchar>(row, col);		    
		    if(d > 0 && d < (disparityRange - 1)){		  
          denom2 = max(S[row][col][d-1] - 2.0*S[row][col][d] + S[row][col][d+1] ,1.0);	      
		      D->at<uchar>(row, col) = round(d*fator + ((S[row][col][d-1] - S[row][col][d+1])*fator + denom2)/(2.0*denom2));
		    }
		    else{
		      D->at<uchar>(row, col) = d;
		    }
		  }
		}		
	}
}

void Sgbm::subPixelEnhancement(unsigned short ***S, Mat &D, int disparityRange, int fator){
  int row, col, d;
  int denom2;
  for (row = 0; row < (D.rows); row++){
		for (col = 0; col < (D.cols); col++){
		  if(D.at<uchar>(row, col) != 255){
		    d = D.at<uchar>(row, col);		    
		    if(d > 0 && d < (disparityRange - 1)){		  
          denom2 = max(S[row][col][d-1] - 2.0*S[row][col][d] + S[row][col][d+1] ,1.0);	      
		      D.at<uchar>(row, col) = round(d*fator + ((S[row][col][d-1] - S[row][col][d+1])*fator + denom2)/(2.0*denom2));
		    }
		    else{
		      D.at<uchar>(row, col) = d*fator;
		    }
		  }
		}		
	}
}

void Sgbm::run(Mat firstImage, Mat secondImage, Mat &disparityMap, int dispRange, int CLRDispMax, int minDiff, int downScaleParam, int filterLimitValue, int wradiusY, int wradiusX, int P1, int P2){

  /*unsigned short **PFI;    // pre-processed first image W x H
  unsigned short **PSE;    // pre-processed second image W x H    
  unsigned short ***C;     // pixel cost array W x H x D
  unsigned short ***S;     // aggregated cost array W x H x D
  unsigned short ****A;    // single path cost array P x W x H x D*/
  
  Mat firstImage_grad_x;        
  Mat secondImage_grad_x;
  
  //allocate image space for the preprocessed images
  PFI = new unsigned short*[firstImage.rows];
  PSE = new unsigned short*[firstImage.rows];	 
	 
	// allocate cost arrays
  C = new unsigned short**[firstImage.rows];
  S = new unsigned short**[firstImage.rows];
  
  for (int row = 0; row < firstImage.rows; ++row) {
      C[row] = new unsigned short*[firstImage.cols];
      S[row] = new unsigned short*[firstImage.cols]();
      PFI[row] = new unsigned short[firstImage.cols]();
      PSE[row] = new unsigned short[firstImage.cols]();
      for (int col = 0; col < firstImage.cols; ++col) {
          C[row][col] = new unsigned short[dispRange];
          S[row][col] = new unsigned short[dispRange](); // initialize to 0
      }
  }
  
  A = new unsigned short ***[PATHS_PER_SCAN];
  for(int path = 0; path < PATHS_PER_SCAN; ++path) {
      A[path] = new unsigned short **[firstImage.rows];
      for (int row = 0; row < firstImage.rows; ++row) {
          A[path][row] = new unsigned short*[firstImage.cols];
          for (int col = 0; col < firstImage.cols; ++col) {
              A[path][row][col] = new unsigned short[dispRange];
              for (unsigned int d = 0; d < dispRange; ++d) {
                  A[path][row][col][d] = 0;
              }
          }
      }
  }      
     
  
   
  //cv::Mat localDisparityMap;// = cv::Mat(cv::Size(firstImage.cols, firstImage.rows), CV_8UC1, cv::Scalar::all(0));
  //cv::Mat localDisparityMap1;
    
  firstImage_grad_x = cv::Mat(cv::Mat::zeros(firstImage.rows, firstImage.cols, CV_16UC1) );
  secondImage_grad_x = cv::Mat(cv::Mat::zeros(firstImage.rows, firstImage.cols, CV_16UC1) );
     
  int first_image_temp_grad_x;
  int second_image_temp_grad_x;  
 
  for(int y = 1; y < (firstImage.rows-1); y++){
    for(int x = 1; x < (firstImage.cols-1); x++){
        first_image_temp_grad_x = xGradient(firstImage, x, y);
        second_image_temp_grad_x = xGradient(secondImage, x, y);      
        
        firstImage_grad_x.at<short>(y,x) = xGradient(firstImage, x, y);
        secondImage_grad_x.at<short>(y,x) = xGradient(secondImage, x, y);      
        
        first_image_temp_grad_x = first_image_temp_grad_x > filterLimitValue ? filterLimitValue:first_image_temp_grad_x;
        first_image_temp_grad_x = first_image_temp_grad_x < -filterLimitValue ? -filterLimitValue:first_image_temp_grad_x;
        
        second_image_temp_grad_x = second_image_temp_grad_x > filterLimitValue ? filterLimitValue:second_image_temp_grad_x;
        second_image_temp_grad_x = second_image_temp_grad_x < -filterLimitValue ? -filterLimitValue:second_image_temp_grad_x;
        
        firstImage_grad_x.at<short>(y,x) = first_image_temp_grad_x + filterLimitValue;
        secondImage_grad_x.at<short>(y,x) = second_image_temp_grad_x + filterLimitValue;
    }
  }   
   
  calcSadCost(&firstImage_grad_x, &secondImage_grad_x, C, dispRange, wradiusY, wradiusX);
 
  aggregateCosts(firstImage.rows, firstImage.cols, dispRange, C, A, S, P1, P2);
   
  cv::Mat *disparityMapDb = new cv::Mat(cv::Mat::zeros(firstImage.rows, firstImage.cols, CV_8UC1) ); //Db
  cv::Mat *disparityMapDm = new cv::Mat(cv::Mat::zeros(firstImage.rows, firstImage.cols, CV_8UC1) );
//  *disparityMap = cv::Mat(cv::Mat::zeros(downFirstImage.rows, downFirstImage.cols, CV_8UC1) );
  //cv::Mat *disparityOut = new cv::Mat(cv::Mat::zeros(firstImage.rows, firstImage.cols, CV_8UC1) ); //Db
   
  computeDisparity(S, firstImage.rows, firstImage.cols, dispRange, disparityMapDb); //Db
  
  //imshow("disparityMapDb1", *disparityMapDb);
  //waitKey();
         
  removeInconsistency(S, firstImage.rows, firstImage.cols, dispRange, disparityMapDb, minDiff);
  
  //imshow("disparityMapDb", *disparityMapDb);
  //waitKey();
   
  computeDisparityDm(S, disparityMapDb, firstImage.rows, firstImage.cols, dispRange, disparityMapDm); //Dm
   
  leftRightCheck(disparityMapDb, disparityMapDm, &disparityMap, CLRDispMax);   
  
  //imshow("disparityMap", *disparityMap);
  //waitKey();
   
  subPixelEnhancement(S, &disparityMap, dispRange, downScaleParam);  
  
    
  
  //cout << "opas" << endl;
  //getchar();
  
}

void Sgbm::run(Mat firstImage, Mat secondImage, Mat &disparityMap, int dispRange){  
 
  preProcess(firstImage, secondImage, firstImage_grad_x, secondImage_grad_x);  
   
  calcSadCost(&firstImage_grad_x, &secondImage_grad_x, C, dispRange, wradiusY, wradiusX);
  
  setInitialCondition(firstImage, secondImage, dispRange);
 
  aggregateCosts(firstImage.rows, firstImage.cols, dispRange, C, A, S, minA, P1, P2);
   
  computeDisparity(S, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDb); //Db
         
  removeInconsistency(S, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDb, minDiff);
   
  computeDisparityDm(S, this->disparityMapDb, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDm); //Dm
   
  leftRightCheck(this->disparityMapDb, this->disparityMapDm, disparityMap, CLRDispMax);   
   
  subPixelEnhancement(S, disparityMap, dispRange, downScaleParam);  
  
}

void Sgbm::findTexturelessArea(Mat &inputImage, int wradius_y, int wradius_x){
  Mat textureless; // = cv::Mat(cv::Mat::zeros(firstImage.rows, firstImage.cols, CV_8UC1) );
  textureless.create( inputImage.rows, inputImage.cols, CV_8UC1);
  int row, col, sum1, sum2, result;
  int dy, dx;
  for (row = 0; row < (inputImage.rows); row++){
		for (col = 0; col < (inputImage.cols); col++){
		  sum1 = 0;
		  sum2 = 0;
		  for (dy = -wradius_y; dy <= wradius_y; dy++) {
				for (dx = -wradius_x; dx <= wradius_x; dx++) {
				  sum1 += pow(inputImage.at<uchar>(row + dy,col + dx),2);
				  sum2 += inputImage.at<uchar>(row + dy,col + dx);
				}						 
			}
			
			sum1 = sum1/(wradius_y *  wradius_x);
			
			sum2 = pow(sum2/(wradius_y *  wradius_x),2);
			result = sum2 - sum1;
			if(result >= 300000) textureless.at<uchar>(row,col) = 0;
			else textureless.at<uchar>(row,col) = 255;
			//cout << (int)textureless.at<short>(row,col) << endl;
		}
  }
  imshow("textureless", textureless);
  waitKey(1);  

}

void Sgbm::run(Mat firstImage, Mat secondImage, int dispRange){  

  setInitialCondition(firstImage, secondImage, dispRange);
  
  //findTexturelessArea(firstImage,1, 1);
 
  preProcess(firstImage, secondImage, firstImage_grad_x, secondImage_grad_x); 
  
  //preProcess2(firstImage, secondImage, firstImage_grad_x, secondImage_grad_x, 1, 1);
  
  calcSadCostv3(&firstImage_grad_x, &secondImage_grad_x, C, dispRange, wradiusY, wradiusX);
 
  aggregateCostsv3(firstImage.rows, firstImage.cols, dispRange, C, A, S, minA, P1, P2);
  
  //calcCostPlusAggregateCosts(&firstImage_grad_x, &secondImage_grad_x, firstImage.rows, firstImage.cols, dispRange, C, A, S, minA, P1, P2, wradiusY, wradiusX);
   
  computeDisparity(S, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDb); //Db
         
  removeInconsistency(S, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDb, minDiff);
   
  computeDisparityDm(S, this->disparityMapDb, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDm); //Dm
   
  leftRightCheck(this->disparityMapDb, this->disparityMapDm, this->disparityMap, CLRDispMax);   
   
  subPixelEnhancement(S, this->disparityMap, dispRange, downScaleParam);  
  
}

void Sgbm::runC(Mat firstImage, Mat secondImage, int dispRange){  

  setInitialCondition(firstImage, secondImage, dispRange);
  
  //findTexturelessArea(firstImage,1, 1);
 
  preProcess(firstImage, secondImage, firstImage_grad_x, secondImage_grad_x); 
  
  //preProcess2(firstImage, secondImage, firstImage_grad_x, secondImage_grad_x, 1, 1);
  
  calcSadCost(&firstImage_grad_x, &secondImage_grad_x, C, dispRange, wradiusY, wradiusX);
 
  aggregateCosts(firstImage.rows, firstImage.cols, dispRange, C, A, S, minA, P1, P2);
   
  computeDisparity(S, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDb); //Db
         
  removeInconsistency(S, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDb, minDiff);
   
  computeDisparityDm(S, this->disparityMapDb, firstImage.rows, firstImage.cols, dispRange, this->disparityMapDm); //Dm
   
  leftRightCheck(this->disparityMapDb, this->disparityMapDm, this->disparityMap, CLRDispMax);   
   
  subPixelEnhancement(S, this->disparityMap, dispRange, downScaleParam);  
  
}

void Sgbm::setInitialCondition(Mat &firstImage, Mat &secondImage, int dispRange){
  for (int row = 0; row < firstImage.rows; ++row) {
    for (int col = 0; col < firstImage.cols; ++col) {
      //this->disparityMap.at<uchar>(row, col) = 0;
      for (unsigned int d = 0; d < dispRange; ++d) {
        S[row][col][d] = 0;               
        for(int path = 0; path < PATHS_PER_SCAN; ++path) {
          A[path][row][col][d] = 0;
          minA[path][row][col] = MAX_SHORT;
        }
      }
    }
  }
}

void Sgbm::preProcess(Mat &firstImage, Mat &secondImage, Mat &firstImage_grad_x, Mat &secondImage_grad_x){

  for(int y = 1; y < (firstImage.rows-1); y++){
    for(int x = 1; x < (firstImage.cols-1); x++){
        first_image_temp_grad_x = xGradient(firstImage, x, y);
        second_image_temp_grad_x = xGradient(secondImage, x, y);  
        
        //first_image_temp_grad_y = yGradient(firstImage, x, y);
        //second_image_temp_grad_y = yGradient(secondImage, x, y); 
        
               
        
        first_image_temp_grad_x = first_image_temp_grad_x > filterLimitValue ? filterLimitValue:first_image_temp_grad_x;
        first_image_temp_grad_x = first_image_temp_grad_x < -filterLimitValue ? -filterLimitValue:first_image_temp_grad_x;
        
        second_image_temp_grad_x = second_image_temp_grad_x > filterLimitValue ? filterLimitValue:second_image_temp_grad_x;
        second_image_temp_grad_x = second_image_temp_grad_x < -filterLimitValue ? -filterLimitValue:second_image_temp_grad_x;
        
        firstImage_grad_x.at<short>(y,x) = first_image_temp_grad_x + filterLimitValue;
        secondImage_grad_x.at<short>(y,x) = second_image_temp_grad_x + filterLimitValue;
        
        /*firstImage_grad_x.at<short>(y,x) = abs(first_image_temp_grad_x) + abs(first_image_temp_grad_y);
        secondImage_grad_x.at<short>(y,x) = abs(second_image_temp_grad_x) + abs(second_image_temp_grad_y);
        
        firstImage_grad_x.at<short>(y,x) = firstImage_grad_x.at<short>(y,x) > filterLimitValue ? filterLimitValue:firstImage_grad_x.at<short>(y,x);
        
        secondImage_grad_x.at<short>(y,x) = secondImage_grad_x.at<short>(y,x) > filterLimitValue ? filterLimitValue:secondImage_grad_x.at<short>(y,x);*/
        
    }
  }
}

void Sgbm::preProcess2(Mat &firstImage, Mat &secondImage, Mat &firstImage_grad_x, Mat &secondImage_grad_x, int wradius_y, int wradius_x){

  int row, col, sum1, sum2, result1,result2;
  int dy, dx;
  int sum3, sum4;

  for(row = 1; row < (firstImage.rows-1); row++){
    for(col = 1; col < (firstImage.cols-1); col++){
    
      //Mat textureless; // = cv::Mat(cv::Mat::zeros(firstImage.rows, firstImage.cols, CV_8UC1) );
      //textureless.create( inputImage.rows, inputImage.cols, CV_8UC1);
      
      //for (row = 0; row < (inputImage.rows); row++){
		    //for (col = 0; col < (inputImage.cols); col++){
		      sum1 = 0;
		      sum2 = 0;
		      sum3 = 0;
		      sum4 = 0;
		      for (dy = -wradius_y; dy <= wradius_y; dy++) {
				    for (dx = -wradius_x; dx <= wradius_x; dx++) {
				      sum1 += pow(firstImage.at<uchar>(row + dy,col + dx),2);
				      sum2 += firstImage.at<uchar>(row + dy,col + dx);
				      sum3 += pow(secondImage.at<uchar>(row + dy,col + dx),2);
				      sum4 += secondImage.at<uchar>(row + dy,col + dx);
				    }						 
			    }
			
			    sum1 = sum1/(wradius_y *  wradius_x);			
			    sum2 = pow(sum2/(wradius_y *  wradius_x),2);
			    sum3 = sum3/(wradius_y *  wradius_x);			
			    sum4 = pow(sum4/(wradius_y *  wradius_x),2);
			    
			    result1 = sum2 - sum1;
			    result2 = sum4 - sum3;
			    
			    firstImage_grad_x.at<short>(row,col) = result1;
          secondImage_grad_x.at<short>(row,col) = result2;
			    
			    //if(result1 >= 300000) textureless.at<uchar>(row,col) = 0;
			    //else textureless.at<uchar>(row,col) = 255;
			    //cout << (int)textureless.at<short>(row,col) << endl;
		    }
      }
      //imshow("textureless", textureless);
      //waitKey(1);
    
    
    
    
        /*first_image_temp_grad_x = xGradient(firstImage, x, y);
        second_image_temp_grad_x = xGradient(secondImage, x, y);        
        
        first_image_temp_grad_x = first_image_temp_grad_x > filterLimitValue ? filterLimitValue:first_image_temp_grad_x;
        first_image_temp_grad_x = first_image_temp_grad_x < -filterLimitValue ? -filterLimitValue:first_image_temp_grad_x;
        
        second_image_temp_grad_x = second_image_temp_grad_x > filterLimitValue ? filterLimitValue:second_image_temp_grad_x;
        second_image_temp_grad_x = second_image_temp_grad_x < -filterLimitValue ? -filterLimitValue:second_image_temp_grad_x;
        
        firstImage_grad_x.at<short>(y,x) = first_image_temp_grad_x + filterLimitValue;
        secondImage_grad_x.at<short>(y,x) = second_image_temp_grad_x + filterLimitValue;*/
    //}
  //}
}

//resize to upscale
void Sgbm::resizeI(Mat *in, Mat *out, int fator){
  cv::Mat temp = in->clone(); 
  out->create( in->rows*fator, in->cols*fator, CV_8UC1);
  for(int row = 0; row < out->rows; row++){
      for(int col = 0; col < out->cols; col++){
        if(temp.at<uchar>(row/fator,col/fator) == 255){
          out->at<uchar>(row, col) = 0;
        }
        else{
          out->at<uchar>(row, col) = (temp.at<uchar>(row/fator,col/fator));    
        }
        //out->at<uchar>(row, col) = (temp.at<uchar>(row/fator,col/fator));    
      }
  } 
}

//resize to downscale
void Sgbm::resizeD(Mat *in, Mat *out, int fator){
  cv::Mat temp = in->clone();
  out->create( in->rows/fator, in->cols/fator, CV_8UC1);
  for(int row = 0; row < out->rows; row++){
      out->at<uchar>(row, 0) = temp.at<uchar>(row*fator,0);
      for(int col = 1; col < out->cols; col++){
        if(row > 0)
        out->at<uchar>(row, col) = round((temp.at<uchar>(row*fator,col*fator) + temp.at<uchar>(row*fator,col*fator - 1) + temp.at<uchar>(row*fator,col*fator + 1) + temp.at<uchar>(row*fator+1,col*fator-1) + temp.at<uchar>(row*fator-1,col*fator-1) + temp.at<uchar>(row*fator-1,col*fator+1) + temp.at<uchar>(row*fator+1,col*fator+1))/7.0);
        else
        out->at<uchar>(row, col) = (temp.at<uchar>(row*fator,col*fator) + temp.at<uchar>(row*fator,col*fator - 1) + temp.at<uchar>(row*fator,col*fator + 1) + temp.at<uchar>(row*fator+1,col*fator-1) + temp.at<uchar>(row*fator+1,col*fator+1))/5;     
      }
  } 
}

void Sgbm::runWithDownScale(Mat &firstImage, Mat &secondImage, int dispRange, int CLRDispMax, int minDiff, int downScaleParam, int filterLimitValue, int wradiusY, int wradiusX, int P1, int P2, Mat &disparityMap){

  unsigned int disparityRange = dispRange/downScaleParam;
  Mat downFirstImage, downSecondImage;
   
  resizeD(&firstImage, &downFirstImage, downScaleParam);
  resizeD(&secondImage, &downSecondImage, downScaleParam);

  downFirstImage = downFirstImage;
  downSecondImage = downSecondImage;  
  
  disparityMap.create( downFirstImage.rows, downFirstImage.cols, CV_8UC1);
  
  Sgbm sgbmobj;
    
  sgbmobj.run(downFirstImage, downSecondImage, disparityMap, disparityRange, CLRDispMax, minDiff, downScaleParam, filterLimitValue, wradiusY, wradiusX, P1, P2);  
   
  resizeI(&disparityMap, &disparityMap, downScaleParam);
       
  medianBlur(disparityMap, disparityMap, 3);

}

void Sgbm::runA(Mat &firstImage, Mat &secondImage, Mat &disparityMap){

  //cout << downScaleParam << endl;

  unsigned int disparityRange = dispRange/downScaleParam;
  //Mat downFirstImage, downSecondImage;
   
  resizeD(&firstImage, &downFirstImage, downScaleParam);
  resizeD(&secondImage, &downSecondImage, downScaleParam);
  
  //cout << downScaleParam << endl;

  //downFirstImage = downFirstImage;
  //downSecondImage = downSecondImage;  
  
  disparityMap.create( downFirstImage.rows, downFirstImage.cols, CV_8UC1);
  
  //cout << downScaleParam << endl;
  
  //Sgbm sgbmobj;
    
  this->run(downFirstImage, downSecondImage, disparityMap, disparityRange);  
   
  resizeI(&disparityMap, &disparityMap, downScaleParam);
       
  medianBlur(disparityMap, disparityMap, 3);
  
  

}

void Sgbm::runAB(Mat &firstImage, Mat &secondImage){
   
  resizeD(&firstImage, &downFirstImage, downScaleParam);
  resizeD(&secondImage, &downSecondImage, downScaleParam);
    
  this->run(downFirstImage, downSecondImage, ddispRange);  
   
  resizeI(&this->disparityMap, &this->disparityMapOut, downScaleParam);
       
  medianBlur(this->disparityMapOut, this->disparityMapOut, 3);

}
