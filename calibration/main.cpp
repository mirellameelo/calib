#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <iterator>  // std::begin, std::end
#include <stdio.h>
#include "iostream"
#include <fstream>
#include <vector>
#include <algorithm>    // std::min_element, std::max_element
#include "core.hpp"
#include "StereoEfficientLargeScale.h"
//#include "sgbm.h"
//#include <windows.h>


using namespace cv;
using namespace std;

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)


//enum Direction { west = 0, northwest = 1, north = 2, northeast = 3, east = 4, southeast = 5, south = 6, southwest = 7, enumTypeEnd };

//enum Direction { west = 0, northwest = 5, north = 1, northeast = 6, east = 2, southeast = 7, south = 3, southwest = 8, enumTypeEnd =4 };

//enum Direction { west = 0, northwest = 5, north = 10, northeast = 6, east = 2, southeast = 7, south = 3, southwest = 8, enumTypeEnd = 1 };

enum Direction { west = 0, northwest = 1, north = 2, northeast = 3, east = 5, southeast = 6, south = 7, southwest = 8, enumTypeEnd = 4 };

//enum Direction { west = 0, north = 1, east = 2, south = 3, enumTypeEnd };

void BMOpencv(Mat *imageL, Mat *imageR, Mat *imageResult){
	StereoBM sbm;

	/*sbm.state->SADWindowSize = 15; 
	sbm.state->numberOfDisparities = 112;
	sbm.state->preFilterSize = 5;
	sbm.state->preFilterCap = 1;
	sbm.state->minDisparity = 0;
	sbm.state->textureThreshold = 50;
	sbm.state->uniquenessRatio = 5;
	sbm.state->speckleWindowSize = 0;
	sbm.state->speckleRange = 4;
	sbm.state->disp12MaxDiff = 64;*/
	
	//StereoBM sbm;
	sbm.state->SADWindowSize = 7;
	sbm.state->numberOfDisparities = 128;  //large baseline
	sbm.state->preFilterSize = 127;
	sbm.state->preFilterCap = 61;
	sbm.state->minDisparity = 0;          // large (30cm) baseline and 36 pixel offset toward
											// centerline on each camera (72 pixel preset disparity)
	sbm.state->textureThreshold = 100; //there appear to be multiple minima with this function
	sbm.state->uniquenessRatio = 5;    //I used 5 with good results with other settings above
	sbm.state->speckleWindowSize = 2;  //I used 2 ""
	sbm.state->speckleRange = 9;       //I used 9 ""

	sbm(*imageL, *imageR, *imageResult);
	
	//normalize(*imageResult, *imageResult, 0.1, 255, CV_MINMAX, CV_8U);
}

void SGBMOpencv(Mat *imageL, Mat *imageR, Mat *imageResult){
	StereoSGBM sgbm;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = 1;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = 128;
	sgbm.P1 = 8 * 3 * sgbm.SADWindowSize * sgbm.SADWindowSize;
	sgbm.P2 = 32 * 3 * sgbm.SADWindowSize * sgbm.SADWindowSize;
	sgbm.uniquenessRatio = 0;
	sgbm.speckleWindowSize = 0;
	sgbm.speckleRange = 0;
	sgbm.disp12MaxDiff = -1;
	sgbm.fullDP = false;

	sgbm(*imageL, *imageR, *imageResult);
}

//CALCULO DE DISPARIDADE ENVOLVENDO:
//1.CUSTO DE MATCHING: METODO SAD  
//2.AGREGACAO DE CUSTO: METODO SAD  
//3.CALCULO DE DISPARIDADE: WINNER-TAKES-ALL
//A RESPOSTA DESTE METODO É UMA MATRIZ BI-DIMENSIONAL (y,x) onde 'y' indica coluna, 'x' indica linha
void LOCALMETHODS(Mat *imageL, Mat *imageR, Mat *imageResult){
	int dMin = 0 , dMax = 15;
	int d, row, col;
	Mat tempImageL, tempImageR;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	double corrScore;
	
	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA ESQUERDA
	imageL->convertTo(tempImageL, CV_32F, 1.0 / 255.0);

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA DIREITA
	imageR->convertTo(tempImageR, CV_32F, 1.0 / 255.0);

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(imageL->rows, imageL->cols, CV_32FC1, float(0));
	
	for (row = win; row < (imageL->rows - win); row++){
		for (col = win; col < (imageL->cols - win - dMax); col++){
			prevcorrScore = 65532;
			bestMatchSoFar = dMin;

			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE
			for (d = dMin; d < dMax; d++){

				//RECORTANDO A PARTIR DA IMAGEM BASE (ESQUERDA) A JANELA 
				//CENTRADA EM (x,y) QUE SERA USADA PARA O CALCULO DA FUNCAO DE CUSTO DO PIXEL (x,y) 
				Rect myLeftROI(col - win, row - win, corrWindowSize, corrWindowSize);
				Mat regionLeft = (tempImageL)(myLeftROI); 
				cv::Scalar leftMeanValue = mean(regionLeft);
				
				//RECORTANDO A PARTIR DA IMAGEM DE CORRESPONDENCIA (DIREITA) A JANELA 
				//CENTRADA NO PIXEL (x+d,y) QUE SERA USADA PARA O CALCULO DA FUNCAO DE CUSTO EM RELACAO A JANELA DA IMAGEM BASE
				Rect myRightROI(col - win + d, row - win, corrWindowSize, corrWindowSize);
				Mat regionRight = (tempImageR)(myRightROI);
				cv::Scalar rigthMeanValue = mean(regionRight);				

				//JANELA DE RESPOSTA DA OPERACAO ENVOLVENDO AS DUAS JANELAS (A DE BASE E A DE CORRESPONDENCIA)
				Mat tempCorrScore = Mat(corrWindowSize, corrWindowSize, CV_32FC1);

				//NUCLEO DO METODO AD. ESTAMOS CALCULANDO A DIFERENCA ABSOLUTA ENTRE A JANELA CENTRADA NO PIXEL (x,y) DA IMAGEM BASE E 
				//A JANELA CENTRADA NO PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA 
				for (dispRow = 0; dispRow < corrWindowSize; dispRow++){
					for (dispCol = 0; dispCol < corrWindowSize; dispCol++){
						tempCorrScore.at<float>(dispRow, dispCol) = abs(regionLeft.at<float>(dispRow, dispCol) - regionRight.at<float>(dispRow, dispCol));
					}
				}
				
				//CUSTO DA MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
				corrScore = sum(cv::sum(tempCorrScore)).val[0];
				
				//CALCULO DA DISPARIDADE. 
				//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
				if (prevcorrScore > corrScore) {
					prevcorrScore = corrScore;
					bestMatchSoFar = d;
				}
				
			}
			
			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			imageResult->at<float>(row, col) = bestMatchSoFar;	
		}
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
}

double rootMeanSquared(Mat *disp1, Mat *disp2){
	double distance = 0;
	int row, col;
	int invalidCounter = 0;
	for (row = 0; row < disp1->rows; row++){
		for (col = 0; col < disp1->cols; col++){
			cout << "reference: " << (int)disp1->at<char>(row, col) << endl;
			cout << "result: " << disp2->at<float>(row, col) << endl;
			getchar();
			if (disp2->at<float>(row, col) == -1){
				invalidCounter++;
			}
			else{
				distance += pow((int)disp1->at<char>(row, col) - disp2->at<float>(row, col), 2.0);
			}
		}
	}
	cout << "invalidCounter: " << invalidCounter << endl;
	distance = distance / (float)(disp1->rows * disp1->cols - invalidCounter);
	distance = sqrt(distance);
	return distance;
}

double pixelMatchingBad(Mat *disp1, Mat *disp2, double sigmad){
	double distance = 0;
	int row, col;
	int invalidCounter = 0;
	for (row = 0; row < disp1->rows; row++){
		for (col = 0; col < disp1->cols; col++){
			if (disp1->at<float>(row, col) == -1){
				invalidCounter++;
			}
			else{
				if (abs((int)disp1->at<char>(row, col) - disp2->at<float>(row, col)) > sigmad){
					distance += 1.0;
				}
			}			
		}
	}
	distance = distance / (float)(disp1->rows * disp1->cols - invalidCounter);
	return distance;
}

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void matchingCostCalculate(Mat *imageL, Mat *imageR, Mat *imageResult, int dMin, int dMax){
	//int dMin = 0;
	int d, row, col;
	Mat tempImageL, tempImageR;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 1; //corrWindowSize = row = col from the corrWindow. Obs: esse valor precisar ser impar
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	int dims[3] = { imageL->rows, imageL->cols, dMax };
	double corrScore;

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA ESQUERDA
	imageL->convertTo(tempImageL, CV_32F, 1.0 / 255.0);

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA DIREITA
	imageR->convertTo(tempImageR, CV_32F, 1.0 / 255.0);

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(3, dims, CV_32F, Scalar(0));

	for (row = win; row < (imageL->rows - win); row++){
		for (col = win; col < (imageL->cols - win - dMax); col++){
			prevcorrScore = 65532;
			bestMatchSoFar = dMin;

			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE
			for (d = dMin; d < dMax; d++){

				//RECORTANDO A PARTIR DA IMAGEM BASE (ESQUERDA) A JANELA 
				//CENTRADA EM (x,y) QUE SERA USADA PARA O CALCULO DA FUNCAO DE CUSTO DO PIXEL (x,y) 
				Rect myLeftROI(col - win, row - win, corrWindowSize, corrWindowSize);
				Mat regionLeft = (tempImageL)(myLeftROI);
				//cv::Scalar leftMeanValue = mean(regionLeft);

				//RECORTANDO A PARTIR DA IMAGEM DE CORRESPONDENCIA (DIREITA) A JANELA 
				//CENTRADA NO PIXEL (x+d,y) QUE SERA USADA PARA O CALCULO DA FUNCAO DE CUSTO EM RELACAO A JANELA DA IMAGEM BASE
				Rect myRightROI(col - win + d, row - win, corrWindowSize, corrWindowSize);
				Mat regionRight = (tempImageR)(myRightROI);
				//cv::Scalar rigthMeanValue = mean(regionRight);

				//JANELA DE RESPOSTA DA OPERACAO ENVOLVENDO AS DUAS JANELAS (A DE BASE E A DE CORRESPONDENCIA)
				Mat tempCorrScore = Mat(corrWindowSize, corrWindowSize, CV_32FC1);

				//NUCLEO DO METODO AD. ESTAMOS CALCULANDO A DIFERENCA ABSOLUTA ENTRE A JANELA CENTRADA NO PIXEL (x,y) DA IMAGEM BASE E 
				//A JANELA CENTRADA NO PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA 
				for (dispRow = 0; dispRow < corrWindowSize; dispRow++){
					for (dispCol = 0; dispCol < corrWindowSize; dispCol++){
						tempCorrScore.at<float>(dispRow, dispCol) = abs(regionLeft.at<float>(dispRow, dispCol) - regionRight.at<float>(dispRow, dispCol));
					}
				}

				//CUSTO DA MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
				corrScore = sum(cv::sum(tempCorrScore)).val[0];

				imageResult->at<float>(row, col, d) = corrScore;

				//CALCULO DA DISPARIDADE. 
				//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
				/*if (prevcorrScore > corrScore) {
					prevcorrScore = corrScore;
					bestMatchSoFar = d;
				}*/

			}

			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			//imageResult->at<float>(row, col) = bestMatchSoFar;
		}
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

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void censusMatchingCostCalculate(Mat *imageL, Mat *imageR, Mat *imageResult, int dMin, int dMax){
	int d, row, col;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 1; //corrWindowSize = row = col from the corrWindow. Obs: esse valor precisar ser impar
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	int dims[3] = { imageL->rows, imageL->cols, dMax };
	double corrScore;
	
	int wradius_y = 4, wradius_x = 4; 

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(3, dims, CV_8S, Scalar(-10));
	
	int sum, product, count3, count4, count5, dy, dx, temp1, temp2, count;
	
	for (row = 0; row < (imageL->rows); row++){
		for (col = 0; col < (imageL->cols); col++){
			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE
			for (d = 0; d < dMax; d++){		   
			
				if((col - d) >= 0){
			  
					sum = 0;
					product = 0;                                        
					count3 = 0;
					count4 = 0;
					count5 = 0;
					imageResult->at<schar>(row, col, d) = -50; 
            
					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) { 
							if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0)){
								if ((imageL->at<uchar>(row + dy, col + dx) >= imageL->at<uchar>(row, col))){
									temp1 = 1;
								}
								else{
									temp1 = 0;
								}

								if ((imageR->at<uchar>(row + dy, col - d + dx) >= imageR->at<uchar>(row, col - d))){
									temp2 = 1;
								}
								else{
									temp2 = 0;
								}

								if (temp1 != temp2){
									count5++;
								}
							}
							/*}
							else{
								temp1 = 0;
							}*/		                            
		                        
							/*if( ((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0) ){                                  
			               		if((imageR->at<uchar>(row+dy, col-d+dx) >= imageR->at<uchar>(row, col-d))){            
									temp2 = 1;
								}
								else{
									temp2 = 0;
								}			                      			                         
							} 
							else{
								temp2 = 0;
							}	*/		                          
                  
							
                  
							//count3++;
							//count4++;
						}             
				  }
				              
				  imageResult->at<schar>(row, col, d) = count5;
			  }
		  }
	  }
	}
}

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void censusMatchingCostCalculate(Mat *imageL, Mat *imageR, Mat *imageResult, Mat *finalResult, int dMin, int dMax, int wradius_y, int wradius_x){
	int d, row, col;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 1; //corrWindowSize = row = col from the corrWindow. Obs: esse valor precisar ser impar
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	int dims[3] = { imageL->rows, imageL->cols, dMax };
	double corrScore;
	int min_d = -10;
	int min_value_d = 100000;

	//int wradius_y = 2, wradius_x = 2;

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
//	*imageResult = Mat(3, dims, CV_8S, Scalar(-10));

	//*finalResult = Mat(imageL->size[0], imageL->size[1], CV_32FC1, float(-1));

	*finalResult = Mat(imageL->size[0], imageL->size[1], CV_8S, Scalar(-1));

	int sum, product, count3, count4, count5, dy, dx, temp1, temp2, count;

	for (row = 0; row < (imageL->rows); row++){
		for (col = 0; col < (imageL->cols); col++){

			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE

			min_d = -10;
			min_value_d = 100000;

			for (d = 0; d < dMax; d++){

				if ((col - d) >= 0){

					//sum = 0;
					//product = 0;
					//count3 = 0;
					//count4 = 0;
					count5 = 0;
					//imageResult->at<schar>(row, col, d) = -50;

					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {
							if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0)){
								if ((imageL->at<uchar>(row + dy, col + dx) >= imageL->at<uchar>(row, col))){
									temp1 = 1;
								}
								else{
									temp1 = 0;
								}

								if ((imageR->at<uchar>(row + dy, col - d + dx) >= imageR->at<uchar>(row, col - d))){
									temp2 = 1;
								}
								else{
									temp2 = 0;
								}

								if (temp1 != temp2){
									count5++;
								}
							}
							/*}
							else{
								temp1 = 0;
							}*/

							/*if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0)){
								if ((imageR->at<uchar>(row + dy, col - d + dx) >= imageR->at<uchar>(row, col - d))){
									temp2 = 1;
								}
								else{
									temp2 = 0;
								}
							}
							else{
								temp2 = 0;
							}*/

							

							//count3++;
							//count4++;
						}
					}

					//imageResult->at<schar>(row, col, d) = count5;
					if (min_value_d > count5){
						min_value_d = count5;
						min_d = d;
					}

				}
			}

			if (min_d >= 0) finalResult->at<schar>(row, col) = min_d + 1; // MAP value (min neg-Log probability)
			else          finalResult->at<schar>(row, col) = 0;    // invalid disparity

		}
	}
}

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void rankMatchingCostCalculate(Mat *imageL, Mat *imageR, Mat *imageResult, Mat *finalResult, int dMin, int dMax, int wradius_y, int wradius_x){
	int d, row, col;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 1; //corrWindowSize = row = col from the corrWindow. Obs: esse valor precisar ser impar
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	int dims[3] = { imageL->rows, imageL->cols, dMax };
	double corrScore;
	int min_d = -10;
	int min_value_d = 100000;

	//int wradius_y = 2, wradius_x = 2;

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
//	*imageResult = Mat(3, dims, CV_8S, Scalar(-10));

	//*finalResult = Mat(imageL->size[0], imageL->size[1], CV_32FC1, float(-1));

	*finalResult = Mat(imageL->size[0], imageL->size[1], CV_8S, Scalar(-1));

	int sum, product, count3, count4, count5, dy, dx, temp1, temp2, count;

	for (row = 0; row < (imageL->rows); row++){
		for (col = 0; col < (imageL->cols); col++){

			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE

			min_d = -10;
			min_value_d = 100000;

			for (d = 0; d < dMax; d++){

				if ((col - d) >= 0){

					//sum = 0;
					//product = 0;
					
					temp1 = 0;
					temp2 = 0;
					//count3 = 0;
					//count4 = 0;
					count5 = 0;
					//imageResult->at<schar>(row, col, d) = -50;

					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {
							if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0)){
								if ((imageL->at<uchar>(row + dy, col + dx) < imageL->at<uchar>(row, col))){
									temp1++;
								}
								/*else{
									temp1 = 0;
								}*/

								if ((imageR->at<uchar>(row + dy, col - d + dx) < imageR->at<uchar>(row, col - d))){
									temp2++;
								}
								/*else{
									temp2 = 0;
								}*/

								/*if (temp1 != temp2){
									count5++;
								}*/
							}
							/*}
							else{

								temp1 = 0;
							}*/

							/*if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0)){

								if ((imageR->at<uchar>(row + dy, col - d + dx) >= imageR->at<uchar>(row, col - d))){

									temp2 = 1;
								}

								else{
									temp2 = 0;

								}
							}
							else{
								temp2 = 0;
							}*/

							

							//count3++;
							//count4++;
						}
					}
					
					count5 = abs(temp1-temp2);

					//imageResult->at<schar>(row, col, d) = count5;
					if (min_value_d > count5){
						min_value_d = count5;//count5;
						min_d = d;
					}

				}
			}

			if (min_d >= 0) finalResult->at<schar>(row, col) = min_d + 1; // MAP value (min neg-Log probability)
			else          finalResult->at<schar>(row, col) = 0;    // invalid disparity

		}
	}
}

//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void censusMatchingCostCalculatewithColor(Mat *imageL, Mat *imageR, Mat *imageResult, Mat *finalResult, int dMin, int dMax, int wradius_y, int wradius_x){

  //cout << "opa1" << endl;	
	//getchar();
	int d, row, col;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 1; //corrWindowSize = row = col from the corrWindow. Obs: esse valor precisar ser impar
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	int dims[3] = { imageL->rows, imageL->cols, dMax };
	double corrScore;
	int min_d = -10;
	int min_value_d = 100000;

	//int wradius_y = 2, wradius_x = 2;

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
//	*imageResult = Mat(3, dims, CV_8S, Scalar(-10));

	//*finalResult = Mat(imageL->size[0], imageL->size[1], CV_32FC1, float(-1));

  //cout << "opa1" << endl;	
	//getchar();

	*finalResult = Mat(imageL->size[0], imageL->size[1], CV_8S, Scalar(-1));
	
	//cout << "opa2" << endl;	
	//getchar();
	
	//cout << (imageL->rows) << endl;

	int sum, product, count3, count4, count5, dy, dx, temp1, temp2, count;

	for (row = 0; row < (imageL->rows); row++){
		for (col = 0; col < (imageL->cols); col++){

			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE
			
			//cout << "opa" << endl;

			min_d = -10;
			min_value_d = 100000;

			for (d = 0; d < dMax; d++){

				if ((col - d) >= 0){

					//sum = 0;
					//product = 0;
					//count3 = 0;
					//count4 = 0;
					count5 = 0;
					//imageResult->at<schar>(row, col, d) = -50;

					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {
							if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0)){
							
							
							//image.at<cv::Vec3b>(y,x)[0]
							
								if ((imageL->at<cv::Vec3b>(row + dy, col + dx)[0] >= imageL->at<cv::Vec3b>(row, col)[0])){
									temp1 = 1;
								}
								else{
									temp1 = 0;
								}

								if ((imageR->at<cv::Vec3b>(row + dy, col - d + dx)[0] >= imageR->at<cv::Vec3b>(row, col - d)[0])){
									temp2 = 1;
								}
								else{
									temp2 = 0;
								}

								if (temp1 != temp2){
									count5++;
								}
							}
							/*}
							else{

								temp1 = 0;
							}*/

							/*if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0)){

								if ((imageR->at<uchar>(row + dy, col - d + dx) >= imageR->at<uchar>(row, col - d))){

									temp2 = 1;
								}

								else{
									temp2 = 0;

								}
							}
							else{
								temp2 = 0;
							}*/

							

							//count3++;
							//count4++;
						}
					}

					//imageResult->at<schar>(row, col, d) = count5;
					if (min_value_d > count5){
						min_value_d = count5;
						min_d = d;
					}

				}
			}

			if (min_d >= 0) finalResult->at<schar>(row, col) = min_d + 1; // MAP value (min neg-Log probability)
			else          finalResult->at<schar>(row, col) = 0;    // invalid disparity

		}
	}
	//getchar();
}



//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void sadMatchingCostCalculate(Mat *imageL, Mat *imageR, Mat *imageResult, Mat *finalResult, int dMin, int dMax, int wradius_y, int wradius_x){
	int d, row, col;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 1; //corrWindowSize = row = col from the corrWindow. Obs: esse valor precisar ser impar
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	int dims[3] = { imageL->rows, imageL->cols, dMax };
	double corrScore;
	int min_d = -10;
	int min_value_d = 100000;

	//int wradius_y = 2, wradius_x = 2;

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
//	*imageResult = Mat(3, dims, CV_8S, Scalar(-10));

	//*finalResult = Mat(imageL->size[0], imageL->size[1], CV_32FC1, float(-1));

	*finalResult = Mat(imageL->size[0], imageL->size[1], CV_8S, Scalar(-1));

	int sum, product, count3, count4, count5, dy, dx, temp1, temp2, count;

	for (row = 0; row < (imageL->rows); row++){
		for (col = 0; col < (imageL->cols); col++){

			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE

			min_d = -10;
			min_value_d = 100000;

			for (d = 0; d < dMax; d++){

				if ((col - d) >= 0){

					sum = 0;
					//product = 0;
					//count3 = 0;
					//count4 = 0;
					count5 = 0;
					count = 0;
					//imageResult->at<schar>(row, col, d) = -50;

					for (dy = -wradius_y; dy <= wradius_y; dy++) {
						for (dx = -wradius_x; dx <= wradius_x; dx++) {	
							/*if ((col - d + dx) > 630){
								cout << "col: " << col << endl;
								cout << "dx: " << dx << endl;
								cout << (col - d + dx) << endl;
								cout << d << endl;
							}*/
							if (((col - d + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0) ){
							  sum += abs( imageL->at<uchar>(row + dy, col + dx) - imageR->at<uchar>(row + dy, col - d + dx) );
							  count++;
							}
							/*if ((col - d + dx) > 630){
								cout << "opa" << endl;
							}*/
						}
					}
					
					//*(disparity_grid+addr3) = sum/count;//abs( (*(I1+addr1)) - (*(I2+addr2)) );
					
					count5 = sum/count;
					

					//imageResult->at<schar>(row, col, d) = count5;
					if (min_value_d > count5){
						min_value_d = count5;
						min_d = d;
					}

				}
			}

			if (min_d >= 0) finalResult->at<schar>(row, col) = min_d; // MAP value (min neg-Log probability)
			else          finalResult->at<schar>(row, col) = -1;    // invalid disparity

		}
	}
}



//1.CALCULO DO CUSTO DE MATCHING: METODO SAD
//A RESPOSTA DESTE METODO É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
void census2DMatchingCostCalculate(Mat *imageL, Mat *imageR, Mat *imageResult, int dMin, int dMax){
	//int dMin = 0;
	int d, row, col;
	//Mat tempImageL, tempImageR;
	int i, j;
	int dispRow, dispCol;
	int bestMatchSoFar;// = dMin;
	int corrWindowSize = 1; //corrWindowSize = row = col from the corrWindow. Obs: esse valor precisar ser impar
	int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532;
	int dyMax = 1;
	int dims[3] = { imageL->rows, imageL->cols, dMax };
	//int dims2[2] = { dyMax,dMax };
	double corrScore;
	int ddy;
	
	int wradius_y = 4, wradius_x = 4; 

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA ESQUERDA
	//imageL->convertTo(tempImageL, CV_32F, 1.0 / 255.0);

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA DIREITA
	//imageR->convertTo(tempImageR, CV_32F, 1.0 / 255.0);
	
	//Mat tempImageResult = Mat(2, dims2, CV_8S, Scalar(-10));

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(3, dims, CV_8S, Scalar(-10));
	
	int sum, product, count3, count4, count5, dy, dx, temp1, temp2, count;
	
	int minValue, indexy;
	//int sum = 0, count = 0;
    
  //int temp1 = 0, temp2 = 0, count3 = 0, count4 = 0, count5 = 0;
	
	//for(int32_t y = 0; y < height; y++){ 
        
  //          for(int32_t x = 0; x < width; x++){ 

	for (row = 0; row < (imageL->rows); row++){
		for (col = 0; col < (imageL->cols); col++){
			//prevcorrScore = 65532;
			//bestMatchSoFar = dMin;

			//CALCULANDO O VALOR DE CUSTO PARA CADA DISPARIDADE E ESCOLHENDO AQUELE QUE TEM MENOS CUSTO
			//IMPLICITAMENTE CALCULAMOS:
			//1.CUSTO DE MATCHING
			//2.AGREGACAO DE CUSTO
			//3.CALCULO DE DISPARIDADE
			for (d = 0; d < dMax; d++){
			
			  imageResult->at<schar>(row, col, d) = -50; 
			  
			  minValue = 1000000;
			  indexy = 0;
			  
			  if((col - d) >= 0){
			  
			  for(ddy = 0; ddy < dyMax; ddy++){			   
			
        //cout << "ddy:" << ddy << endl;
        //cout << "row: " << row << endl; 			
			
			  if((row - ddy) >= 0){
			  
			      
			  
			      sum = 0;
            product = 0;                                        
            count3 = 0;
            count4 = 0;
            count5 = 0;
            
            //imageResult->at<schar>(row, col, d) = -50; 
            
            //tempImageResult.at<schar>(ddy,d) = -50; 
            
            for (dy = -wradius_y; dy <= wradius_y; dy++) {
              for (dx = -wradius_x; dx <= wradius_x; dx++) {    
              
                if( ((col + dx) >= 0) && ((row + dy) >= 0) && ((col + dx) < imageL->cols) && ((row + dy) < imageL->rows) && (dx != 0 || dy != 0) ){
                               
                    if((imageL->at<uchar>(row+dy, col+dx) >= imageL->at<uchar>(row, col))){                       
                        temp1 = 1;              
                      }
                      else{
                        temp1 = 0;
                      }             
                    }
                    else{
                      temp1 = 0;
                    }		                            
		                        
                    if( ((col - d + dx) >= 0) && ((row - ddy + dy) >= 0) && ((col - d + dx) < imageL->cols) && ((row - ddy + dy) < imageL->rows) && (dx != 0 || dy != 0) ){                                  
			               
			               if((imageR->at<uchar>(row-ddy+dy, col-d+dx) >= imageR->at<uchar>(row-ddy, col-d))){            
                     //if(((*(I2+addr2)) >= (*(I2+addr5)))){
                      temp2 = 1;
                     }
                     else{
                      temp2 = 0;
                     }			                      			                         
                  } 
                  else{
                    temp2 = 0;
                  }			                          
                  
                  if(temp1	!= temp2){
                    count5++;
                  }
                  
                  count3++;
                  count4++;
             }
             
             
            
				  }
				  
				  product += count5;
             
          //cout << product << endl;
          //getchar();

          //*(disparity_grid+addr3) = (product);

          //imageResult->at<schar>(row, col, d) = product; 
          
          //tempImageResult.at<schar>(ddy,d) = product; 
          /*if(row == 5)
          cout << "product: " << product << endl;*/
                    
          if(minValue > product){
            minValue = product;
            indexy = ddy;          
            }          
			    }
			  }
			  
			  /*if(row == 5){
			  cout << "final: " << minValue << endl;
			  getchar();}*/
			  
			  imageResult->at<schar>(row, col, d) = minValue; 
			  
			  }
			  
			  
			  
		  }
	  }
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

//3.CALCULO DE DISPARIDADE: WINNER-TAKES-ALL
//A MATRIZ DE ENTRADA É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade
//A RESPOSTA DESTE METODO É UMA MATRIZ BI-DIMENSIONAL (y,x) onde 'y' indica coluna, 'x' indica linha
//mode1 defines if the best disparity is chosen as being that with the largest cost (mode 1) value or the smallest one (mode 0)
//mode2 = 0 returns 'Db' mode2 = 1 returns 'Dm'
void disparityCalculate(Mat *inputImage, Mat *imageResult, int mode1, int mode2){
	int dMin = 0, dMax = inputImage->size[2];
	int d, row, col;
	int bestMatchSoFar;// = dMin;
	int allZero = 1;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532, corrScore;
	
	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_32FC1, float(-1));
	
	for (row = 0; row < inputImage->size[0]; row++){
		for (col = 0; col < inputImage->size[1]; col++){
			if (mode1 == 0){
				prevcorrScore = 65532;
			}
			else{
				prevcorrScore = -64000;
			}
			bestMatchSoFar = dMin;

			//cout << "row: " << row << "col: " << col << " "; 
			allZero = 1;
			
			//CALCULO DE DISPARIDADE
			for (d = dMin; d < dMax; d++){

				//cout << inputImage->at<float>(row, col, d) << " ";

				//CUSTO DO MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
				if (mode2 == 0){
					corrScore = inputImage->at<float>(row, col, d);
					if (corrScore != 0 && allZero == 1) allZero = 0;
				}
				else{
					//if ((row + d) < )
					//cout << "col: " << col << " d: " << d << endl;
					if ((col + d) < inputImage->size[1]){
						
						//getchar();
						corrScore = inputImage->at<float>(row, col + d, d);
					}
					else{
						//cout << "ops" << endl;
						//getchar();
						if (mode1 == 0){
							corrScore = 65532;
						}
						else{
							corrScore = -64000;
						}
					}
				}
				
				//CALCULO DA DISPARIDADE. 
				//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
				if (mode1 == 0){
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
				}

			}

			//cout << endl;

			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			if (allZero == 0) imageResult->at<float>(row, col) = bestMatchSoFar;
			//if (mode2 == 1) cout << bestMatchSoFar << " ";
		}
		//if (mode2 == 1)	cout << endl;
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

void disparityCalculate(Mat *inputImage, Mat *imageResult){
	int dMin = 0, dMax = inputImage->size[2];
	int d, row, col;
	int bestMatchSoFar;// = dMin;
	int allZero = 1;
	
	int min_d   = -10;
  int min_value_d   = 100000;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532, corrScore;
	
	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_8S, Scalar(-1));
	
	//cout << "eaeee" << endl;
	
	//cout << inputImage->size[0] << endl;
	//cout << inputImage->size[1] << endl;
	
	for (row = 0; row < inputImage->size[0]; row++){
		for (col = 0; col < inputImage->size[1]; col++){	
		
			//prevcorrScore = 65532;
			min_d = -10;
			min_value_d = 100000;

			//cout << "row: " << row << "col: " << col << " "; 
			//allZero = 1;
			
			//cout << "row: " << row << endl;
			
			//cout << "col: " << col << endl;
			
			//CALCULO DE DISPARIDADE
			for (d = 0; d < dMax; d++){
			
			  if((col - d) >= 0){
			  
			  //cout << inputImage->at<uchar>(row, col, d) << endl;
			  
			  //cout << "s: " << (int)inputImage->at<uchar>(row, col, d) << endl;
			
			  if(inputImage->at<schar>(row, col, d) >= 0 && min_value_d > inputImage->at<schar>(row, col, d)){
          min_value_d = inputImage->at<schar>(row, col, d);
          min_d = d;
        }
        
        }

			}
			
			//cout << "min_d + 1: " << min_d + 1 << endl;
			
			if (min_d>=0) imageResult->at<schar>(row, col) = min_d; // MAP value (min neg-Log probability)
      else          imageResult->at<schar>(row, col) = -1;    // invalid disparity
			
			//imageResult->at<uchar>(row, col, d) = 

			//cout << endl;

			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			//if (allZero == 0) imageResult->at<float>(row, col) = bestMatchSoFar;
			//if (mode2 == 1) cout << bestMatchSoFar << " ";
		}
		//if (mode2 == 1)	cout << endl;
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

void leftRightCheck(Mat *inputImage, Mat *inputImage2, Mat *imageResult){
	int dMin = 0, dMax = inputImage->size[2];
	int d, row, col;
	int bestMatchSoFar;// = dMin;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532, corrScore;

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_32FC1, float(0));

	//cout << "ops1" << endl;
	//getchar();

	for (row = 0; row < inputImage->size[0]; row++){
		for (col = 0; col < inputImage->size[1]; col++){
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

			//if (abs(inputImage->at<float>(row, col) - inputImage2->at<float>(row, col + inputImage->at<float>(row, col))) <= 1){
				imageResult->at<float>(row, col) = inputImage->at<float>(row, col);
			//}
			/*else{
				imageResult->at<float>(row, col) = 0;
				//cout << "ops" << endl;
			}*/

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

/*Point2D previousPoint(int currentX, int currentY, int limitX, int limitY, int direction){ 
    Point2D point2D;
    switch(direction)
    {
        case 0:
            point2D.x = currentX-1;
            point2D.y = currentY;            
            break;
        case 1:
            point2D.x = currentX-1;
            point2D.y = currentY-1;
            break;
        case 2:
            point2D.x = currentX;
            point2D.y = currentY-1;
            break;
        case 3:
            point2D.x = currentX+1;
            point2D.y = currentY-1;
            break;
        case 4:
            point2D.x = currentX+1;
            point2D.y = currentY;
            break;
        case 5:
            point2D.x = currentX+1;
            point2D.y = currentY+1;
            break;
        case 6:
            point2D.x = currentX;
            point2D.y = currentY+1;
            break;
        case 7:
            point2D.x = currentX-1;
            point2D.y = currentY+1;
            break; 
        default:
            break;   
    
    }
    if(point2D.x < 0 || point2D.y < 0 || point2D.x >= limitX || point2D.y >= limitY){
        point2D.isValid = 0;
    }
    else{
        point2D.isValid = 1;
    }
    return point2D;
}*/

Point2D previousPoint(int currentX, int currentY, int limitX, int limitY, Direction direction){
	Point2D point2D;
	switch (direction)
	{
	case west:
		point2D.x = currentX - 1;
		point2D.y = currentY;
		break;
	case northwest:
		point2D.x = currentX - 1;
		point2D.y = currentY - 1;
		break;
	case north:
		point2D.x = currentX;
		point2D.y = currentY - 1;
		break;
	case northeast:
		point2D.x = currentX + 1;
		point2D.y = currentY - 1;
		break;
	case east:
		point2D.x = currentX + 1;
		point2D.y = currentY;
		break;
	case southeast:
		point2D.x = currentX + 1;
		point2D.y = currentY + 1;
		break;
	case south:
		point2D.x = currentX;
		point2D.y = currentY + 1;
		break;
	case southwest:
		point2D.x = currentX - 1;
		point2D.y = currentY + 1;
		break;
	default:
		assert(!"Invalid Direction enum value");
		break;

	}
	if (point2D.x < 0 || point2D.y < 0 || point2D.x >= limitX || point2D.y >= limitY){
		point2D.isValid = 0;
	}
	else{
		point2D.isValid = 1;
	}
	return point2D;
}

//Direction



//px -> col
//py -> row
//direction -> r
double costPathCalculate(Mat *matchingCost, Mat *pathCost, int disparity, int row, int col, Direction direction, int P1, int P2){
	//int dMin = 0, dMax = matchingCost->size[2];
	//int d, row, col;
	//int bestMatchSoFar;// = dMin;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532, corrScore;
	double result; //result -> Lr(p,d)
	double cpd;
	double minValue = 1000000;
	double minValueForAllDisp = 0; //minkLr(p-r,k)
	vector<double>partialValues, pathCostValuesForAllDisparities;
	Point2D point2D;
	int disp;

	partialValues.resize(4);
	pathCostValuesForAllDisparities.resize(matchingCost->size[2]);
	//int dims[3] = { inputImage->size[0], inputImage->size[1], dMax };

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	//*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_32FC1, float(0));


	//double corrScore;

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA ESQUERDA
	//imageL->convertTo(tempImageL, CV_32F, 1.0 / 255.0);

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA DIREITA
	//imageR->convertTo(tempImageR, CV_32F, 1.0 / 255.0);

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	//*imageResult = Mat(3, dims, CV_32F, Scalar(0)); //Lr(p,d)

	//for (row = 0; row < inputImage->size[0]; row++){
	//	for (col = 0; col < inputImage->size[1]; col++){
	//		prevcorrScore = 65532;
	//		bestMatchSoFar = dMin;

			//CALCULO DE DISPARIDADE
	//		for (d = dMin; d < dMax; d++){

				//CUSTO DO MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
	//cout << "oi1" << endl;
	/*cout << "row, col, disparity" << row << ", " << col << ", " << disparity << endl;
	cout << "matchingCost->size[0]: " << matchingCost->size[0] << endl;
	cout << "matchingCost->size[1]: " << matchingCost->size[1] << endl;
	cout << "matchingCost->size[2]: " << matchingCost->size[2] << endl;*/
	//getchar();*/

	if (pathCost->at<float>(row, col, disparity) > -1) {
		/*cout << "oi2" << endl;
		getchar();*/
		result = pathCost->at<float>(row, col, disparity);
		return result;
	}
	else{
		/*cout << "oi3" << endl;
		getchar();*/
		cpd = matchingCost->at<float>(row, col, disparity); //C(p,d)
		//preciso definir um esquema de orientacao para acessar o pixel anterior
		point2D = previousPoint(col, row, matchingCost->size[1], matchingCost->size[0], direction);

		//if (point2D.isValid == 1){

			/*cout << "oi4" << endl;
			getchar();*/

			/*cout << "col:" << col << "point2D.x: " << point2D.x << endl;
			cout << "row: " << row << "point2D.y: " << point2D.y << endl;*/ 
			/*cout << "cpd: " << cpd << endl;
			getchar();*/

			for (disp = 0; disp < matchingCost->size[2]; disp++){
				//cout << "disp: " << disp << endl;
				pathCostValuesForAllDisparities.at(disp) = costPathCalculate(matchingCost, pathCost, disp, point2D.y, point2D.x, direction, P1, P2);
				//getchar();
			}

			//cout << "oi5" << endl;
			//getchar();

			/*cout << "teste" << endl;
			getchar();*/

			minValue = *std::min_element(pathCostValuesForAllDisparities.begin(), pathCostValuesForAllDisparities.end());
			partialValues.at(3) = minValue;
			minValueForAllDisp = minValue;

			//cout << "oi6" << endl;
			//getchar();

			/*cout << "teste2" << endl;
			getchar();*/

			partialValues.at(0) = costPathCalculate(matchingCost, pathCost, disparity, point2D.y, point2D.x, direction, P1, P2);

			//cout << "oi7" << endl;
			//getchar();

			/*cout << "teste3" << endl;
			getchar();*/

			if (disparity > 0) partialValues.at(1) = costPathCalculate(matchingCost, pathCost, disparity - 1, point2D.y, point2D.x, direction, P1, P2);
			else partialValues.at(1) = -1;

			/*cout << "teste4" << endl;
			getchar();*/

			partialValues.at(1) += P1;

			if ((disparity + 1) < matchingCost->size[2]) partialValues.at(2) = costPathCalculate(matchingCost, pathCost, disparity + 1, point2D.y, point2D.x, direction, P1, P2);
			else partialValues.at(2) = -1;

			/*cout << "teste5" << endl;
			getchar();*/

			partialValues.at(2) += P2;

			/*pathCostValues.at(0) = 10;
			pathCostValues.at(1) = 20;
			pathCostValues.at(2) = -20;
			pathCostValues.at(3) = -2;*/
			minValue = *std::min_element(partialValues.begin(), partialValues.end());
			//cout << minValue << endl;
			//pathCostValues.at(3) = pixelCostGivenDisparity(matchingCost, pathCost, disparity, point2D.y, point2D.x, direction);
			//minValue = std::min_element(std::begin(pathCostValues), std::end(pathCostValues));
			//minValue = std::min_element(pathCostValues.begin(), pathCostValues.end());
			result = cpd + minValue - minValueForAllDisp;
			pathCost->at<float>(row, col, disparity) = result;
			return result;
		}
		/*else{
			return 0;
		}*/
	

		//CALCULO DA DISPARIDADE. 
		//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
		/*if (prevcorrScore > corrScore) {
			prevcorrScore = corrScore;
			bestMatchSoFar = d;
		}*/

		//imageResult->at<float>(row, col, d);
	//}


			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			//imageResult->at<float>(row, col) = bestMatchSoFar;
	//	}
	//}
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

void pixelCostGivenDisparity(Mat *localMatchingCost, Mat *semiGlobalMatchingCost){
	int dMin = 0, dMax = localMatchingCost->size[2];
	int row, col, disp;
	int path;
	int P1 = 1, P2 = 2;
	int dims[3] = { localMatchingCost->size[0], localMatchingCost->size[1], dMax };
	//int pathsNumber = 4;
	Mat pathCost;
	Direction realPath; 
	//int d, row, col;
	//int bestMatchSoFar;// = dMin;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	/*double prevcorrScore = 65532, corrScore;
	double result; //result -> Lr(p,d)
	double cpd;
	double minValue = 1000000;
	double minValueForAllDisp = 0; //minkLr(p-r,k)
	vector<double>pathCostValues;
	Point2D point2D;*/

	//pathCostValues.resize(4);
	//int dims[3] = { inputImage->size[0], inputImage->size[1], dMax };

	//inputImage->size[0], inputImage->size[1]

	
	//double corrScore;

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA ESQUERDA
	//imageL->convertTo(tempImageL, CV_32F, 1.0 / 255.0);

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA DIREITA
	//imageR->convertTo(tempImageR, CV_32F, 1.0 / 255.0);

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	//pathCost = Mat(3, dims, CV_32F, Scalar(-1));
	*semiGlobalMatchingCost = Mat(3, dims, CV_32F, Scalar(0));

	//cout << "opa" << endl;
	//getchar();

	
		//		prevcorrScore = 65532;
		//		bestMatchSoFar = dMin;

	//cout << "pronto" << endl;
	//getchar();

	for (path = 0; path != enumTypeEnd; path++){
		//cout << "ops1" << endl;
		pathCost = Mat(3, dims, CV_32F, Scalar(-1));
		realPath = static_cast<Direction>(path);
		for (col = 0; col < localMatchingCost->size[1]; col++){
			for (disp = 0; disp < localMatchingCost->size[2]; disp++){
				pathCost.at<float>(0, col, disp) = localMatchingCost->at<float>(0, col, disp);
				pathCost.at<float>(localMatchingCost->size[0] - 1, col, disp) = localMatchingCost->at<float>(localMatchingCost->size[0] - 1, col, disp);
			}
		}

		for (row = 0; row < localMatchingCost->size[0]; row++){
			for (disp = 0; disp < localMatchingCost->size[2]; disp++){
				pathCost.at<float>(row, 0, disp) = localMatchingCost->at<float>(row, 0, disp);
				pathCost.at<float>(row, localMatchingCost->size[1] - 1, disp) = localMatchingCost->at<float>(row, localMatchingCost->size[1] - 1, disp);
			}
		}
		//cout << "ops3" << endl;
		//get
		for (row = 0; row < localMatchingCost->size[0]; row++){
			for (col = 0; col < localMatchingCost->size[1]; col++){
				for (disp = 0; disp < localMatchingCost->size[2]; disp++){
					/*cout << "opaaaaaaa" << endl;
					cout << "localMatchingCost->size[0]: " << localMatchingCost->size[0] << endl;
					cout << "localMatchingCost->size[1]: " << localMatchingCost->size[1] << endl;
					cout << "localMatchingCost->size[2]: " << localMatchingCost->size[2] << endl;*/
					//costPathCalculate(Mat *matchingCost, Mat *pathCost, int disparity, int row, int col, int direction, int P1, int P2)
					
					costPathCalculate(localMatchingCost, &pathCost, disp, row, col, realPath, P1, P2);
					semiGlobalMatchingCost->at<float>(row, col, disp) += pathCost.at<float>(row, col, disp);
					
				}
			}
		}
		/*cout << "ops2" << endl;
		getchar();*/
	}

	//pathCost

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	//*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_32FC1, float(0));


	//double corrScore;

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA ESQUERDA
	//imageL->convertTo(tempImageL, CV_32F, 1.0 / 255.0);

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA DIREITA
	//imageR->convertTo(tempImageR, CV_32F, 1.0 / 255.0);

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	//*imageResult = Mat(3, dims, CV_32F, Scalar(0)); //Lr(p,d)

	//for (row = 0; row < inputImage->size[0]; row++){
	//	for (col = 0; col < inputImage->size[1]; col++){
	//		prevcorrScore = 65532;
	//		bestMatchSoFar = dMin;

	//CALCULO DE DISPARIDADE
	//		for (d = dMin; d < dMax; d++){

	//CUSTO DO MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
	/*if (pathCost->at<float>(row, col, disparity) > -1) {
	result = pathCost->at<float>(row, col, disparity);
	return result;
	}
	else{
	cpd = matchingCost->at<float>(row, col, disparity); //C(p,d)
	//preciso definir um esquema de orientacao para acessar o pixel anterior
	point2D = nextPoint(col, row, matchingCost->size[1], matchingCost->size[0], direction);
	pathCostValues.at(0) = pixelCostGivenDisparity(matchingCost, pathCost, disparity, point2D.y, point2D.x, direction);
	pathCostValues.at(1) = pixelCostGivenDisparity(matchingCost, pathCost, disparity, point2D.y, point2D.x, direction - 1);
	pathCostValues.at(2) = pixelCostGivenDisparity(matchingCost, pathCost, disparity, point2D.y, point2D.x, direction + 1);
	minValue = *std::min_element(pathCostValues.begin(), pathCostValues.end());
	cout << minValue << endl;
	//pathCostValues.at(3) = pixelCostGivenDisparity(matchingCost, pathCost, disparity, point2D.y, point2D.x, direction);
	//minValue = std::min_element(std::begin(pathCostValues), std::end(pathCostValues));
	//minValue = std::min_element(pathCostValues.begin(), pathCostValues.end());
	pathCost->at<float>(row, col, disparity) = cpd + minValue - minValueForAllDisp;


	}*/


	//CALCULO DA DISPARIDADE. 
	//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
	/*if (prevcorrScore > corrScore) {
	prevcorrScore = corrScore;
	bestMatchSoFar = d;
	}*/

	//imageResult->at<float>(row, col, d);
	//}


	//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
	//imageResult->at<float>(row, col) = bestMatchSoFar;
	//	}
	//}
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



//3.CALCULO DE DISPARIDADE: WINNER-TAKES-ALL
//A MATRIZ DE ENTRADA É UMA MATRIZ TRI-DIMENSIONAL (y,x,d) onde 'y' indica coluna, 'x' indica linha e 'd' indica disparidade //C(p,d)
//A RESPOSTA DESTE METODO É UMA MATRIZ BI-DIMENSIONAL (y,x) onde 'y' indica coluna, 'x' indica linha. //Lr(p,d)
void pathCostTotalCalculate(Mat *inputImage, Mat *imageResult, int direction){
	int dMin = 0, dMax = inputImage->size[2];
	int d, row, col;
	int bestMatchSoFar;// = dMin;
	//int corrWindowSize = 3; //corrWindowSize = row = col from the corrWindow
	//int win = (corrWindowSize - 1) / 2;
	double prevcorrScore = 65532, corrScore;
	int dims[3] = { inputImage->size[0], inputImage->size[1], dMax };

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	//*imageResult = Mat(inputImage->size[0], inputImage->size[1], CV_32FC1, float(0));

	
	//double corrScore;

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA ESQUERDA
	//imageL->convertTo(tempImageL, CV_32F, 1.0 / 255.0);

	//NORMALIZANDO OS VALORES DE PIXEL DAS IMAGENS DA DIREITA
	//imageR->convertTo(tempImageR, CV_32F, 1.0 / 255.0);

	//INICIALIZANDO A MATRIZ DE DISPARIDADE QUE SERA A RESPOSTA DESTE METODO
	*imageResult = Mat(3, dims, CV_32F, Scalar(0)); //Lr(p,d)

	for (row = 0; row < inputImage->size[0]; row++){
		for (col = 0; col < inputImage->size[1]; col++){
			prevcorrScore = 65532;
			bestMatchSoFar = dMin;

			//CALCULO DE DISPARIDADE
			for (d = dMin; d < dMax; d++){

				//CUSTO DO MATCHING DO PIXEL (x,y) DA IMAGEM BASE E O PIXEL (x+d,y) DA IMAGEM DE CORRESPONDENCIA
				corrScore = inputImage->at<float>(row, col, d); //C(p,d)



				//CALCULO DA DISPARIDADE. 
				//ADOTAMOS O WINNER-TAKES-ALL. O 'd' QUE TIVER MENOR VALOR DE CUSTO SERA ESCOLHIDO PARA O PIXEL (x,y) NA MATRIZ DE DISPARIDADE 
				if (prevcorrScore > corrScore) {
					prevcorrScore = corrScore;
					bestMatchSoFar = d;
				}

				imageResult->at<float>(row, col, d);
			}

			
			//ATRIBUINDO O RESULTADO DA DISPARIDADE COM MENOR CUSTO DE MATCHING
			//imageResult->at<float>(row, col) = bestMatchSoFar;
		}
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

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main(int argc, char* argv[])
{
	int numBoards = 30;
	int board_w = 9;
	int board_h = 6;
	char* method = "SGBM";
	int dataLoadingMode = 2;//0 from camera, 1 from image, 2 from video	
	int calibratedFromFile = 0;
	int loadCalibrationImageFromFile = 0;
	char imageName[100];
	int numberTestImage = 62, countTestImage = 0;
	char baseLeft[100];
	int limiar = 0;
	
	char baseRight[100];
	char tempLeft[100];
	char tempRight[100];
	char strCountTestImage[100];
	int metodoDeExecucao = 1; // 0 - opencv, 1 - meumetodo
	
	cv::Size imageSize;
	imageSize.width = 640;
	imageSize.height = 480;
	
	
	int CLRDispMax = 1, minDiff = 1, downScaleParam = 4, filterLimitValue = 63, wradiusY = 0, wradiusX = 0;
  
  
	//parametros do meu algoritmo
	//int CLRDispMax = 4, minDiff = 10, downScaleParam = 1, filterLimitValue = 63, wradiusY = 0, wradiusX = 0;  
  int P1D = 24 * (2*wradiusY + 1) * (2*wradiusX + 1);
  int P2D = 96 * (2*wradiusY + 1) * (2*wradiusX + 1);
  int dispRange = 80;
  
  VideoWriter videoDisp("outResult.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(imageSize.width, imageSize.height), true);
  
  Mat frame,                     // original frame captured from webcam
    working_frame,               // resized frame where I detect face
    detected_frame,              // here I mark the face(s)
    threshold_grayscale_frame,   // frame -> grayscale -> threshold
    threshold_color_frame,       // frame -> threshold
    merged_frame,                // big frame for combining four frames
    roi;                         // Region Of Interest for merged_frame
    
  merged_frame = Mat(Size(2 * imageSize.width, imageSize.height), CV_8UC3);
  
  //Sgbm sgbmobj(imageSize.height,imageSize.width,downScaleParam,dispRange,CLRDispMax,minDiff,filterLimitValue,wradiusY,wradiusX,P1D,P2D);
  
  //int downScaleParam, int dispRange, int CLRDispMax, int minDiff, int filterLimitValue, int wradiusY, int wradiusX, int P1, int P2
	//char baseLeft[100];
	//int conditionImage = 1;

  #ifdef __linux__ 
      strcpy(imageName, "images/melhores/");
      strcpy(baseLeft, "database/");
      strcpy(baseRight, "database/");
    #elif _WIN32
      strcpy_s(imageName, "images\\melhores\\");
      strcpy_s(baseLeft, "database\\");
      strcpy_s(baseRight, "database\\");	
    #else
  #endif

	char temp[100];
	char temp1[100];
	int counter = 0;
	double rootMeanSquaredValue;
	double pixelMatchingBadValue;
	int intervalo = 50;
	int splitInterval;
	int filterGaussianWindow = 15;
	int pointCondition;
	
	double fps = 1;	

	Mat map1x, map1y, map2x, map2y;
	Mat imgU1, imgU2;
	Mat reference; 
	Mat g1, g2;
	Mat disp, disp8, matchCost, dispResult, Db, Dm, pathCost;
	Mat localDispResult;
	Mat imgU1Lab, imgU2Lab;
	Mat segmentedImage(imageSize.height, imageSize.width, CV_8UC3, cv::Scalar(0, 0, 0));

	Size board_sz = Size(board_w, board_h);
	int board_n = board_w*board_h;

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > imagePoints1, imagePoints2;
	vector<Point2f> corners1, corners2;

	vector<Point3f> obj;
	for (int j = 0; j<board_n; j++)
	{
		obj.push_back(Point3f(j / board_w, j%board_w, 0.0f));
	}

	string videoL = "outL.avi";
	string videoR = "outR.avi";
	VideoCapture videoCapL(videoL);
	VideoCapture videoCapR(videoR);

	Mat img1, img2, gray1, gray2;
	
	
	VideoCapture cap1; 
	VideoCapture cap2; 
	
	//if(dataLoadingMode == 0 && calibratedFromFile == 0 && loadCalibrationImageFromFile == 0){
	cap1 = VideoCapture(1);
	cap1.set(CV_CAP_PROP_FRAME_WIDTH, imageSize.width);
	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, imageSize.height);
	//cap1.set(CV_CAP_PROP_BUFFERSIZE, 3);

	cout << "opa" << endl;
	cout << "opa" << endl;

cout << "opa" << endl;

cout << "opa" << endl;

cout << "opa" << endl;

cout << "opa" << endl;
getchar();

	//Sleep(10);
	//cap1.set(CV_CAP_PROP_FRAME_HEIGHT, imageSize.height);
	//cap1.set(CV_CAP_PROP_SATURATION, 1);
	//cap1.set(CV_CAP_PROP_FPS, fps);
	//cap1.set(CV_CAP_PROP_CONVERT_RGB, 0);
	/*cap1.set(CV_CAP_PROP_BRIGHTNESS, 0);
	cap1.set(CV_CAP_PROP_EXPOSURE, -9);
	cap1.set(CV_CAP_PROP_GAIN, 10);*/
	cap2 = VideoCapture(2);	
	cap2.set(CV_CAP_PROP_FRAME_WIDTH, imageSize.width);
	cap2.set(CV_CAP_PROP_FRAME_HEIGHT, imageSize.height);
	//Sleep(1);
	//cap2.set(CV_CAP_PROP_FPS, fps); //desired  FPS
	//cap2.set(CV_CAP_PROP_CONVERT_RGB, 0);
	//cap2.set(CV_CAP_PROP_BRIGHTNESS, 0);
	//cap2.set(CV_CAP_PROP_EXPOSURE, -9);
	//}



	//cap1 >> img1;
	//cap2 >> img2;	

	StereoEfficientLargeScale elas(0, 128);

	int success = 0, k = 0;
	int conditionWhile = 1;
	bool found1 = false, found2 = false;	
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);  
	Mat D1, D2;
	Mat R, T, E, F;
	Rect roi1, roi2;
	Mat R1, R2, P1, P2, Q;	
	ofstream saida;
	saida.open("saidaminoru.txt");

	if (calibratedFromFile == 0){

		if (loadCalibrationImageFromFile == 0){

			while (success < numBoards)
			{
				cap1 >> img1;
				cap2 >> img2;

				cvtColor(img1, gray1, CV_BGR2GRAY);
				cvtColor(img2, gray2, CV_BGR2GRAY);
				//CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
				found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
				found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

				if (found1)
				{
					cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
					drawChessboardCorners(gray1, board_sz, corners1, found1);
				}

				if (found2)
				{
					cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
					drawChessboardCorners(gray2, board_sz, corners2, found2);
				}

				imshow("image1", gray1);
				imshow("image2", gray2);

				k = waitKey(10);
				if (found1 && found2)
				{
					k = waitKey(0);
				}
				if (k == 27)
				{
					break;
				}
				
				//linux code goes here
				#ifdef __linux__ 
					if (k == 1048608 && found1 != 0 && found2 != 0){
					  pointCondition  = 1;
					}
					else{
					  pointCondition  = 0;
					}
				
				#elif _WIN32
					if (k == ' ' && found1 != 0 && found2 != 0){
					  pointCondition  = 1;
					}
					else{
					  pointCondition  = 0;
					}             
				#else
				#endif	
        			
				if(pointCondition == 1){
					imagePoints1.push_back(corners1);
					imagePoints2.push_back(corners2);
					strcpy(temp, imageName);
					
					strcat(temp, "left");
					sprintf(temp1, "%d", counter);
					strcat(temp, temp1);
					strcat(temp, ".png");
					imwrite(temp, img1);
					strcpy(temp, imageName);
					
					strcat(temp, "right");
					sprintf(temp1, "%d", counter);
					strcat(temp, temp1);
					strcat(temp, ".png");
					imwrite(temp, img2);
					object_points.push_back(obj);
					printf("Corners stored\n");
					success++;
					counter++;

					if (success >= numBoards)
					{
						break;
					}
				}
				}
				destroyAllWindows();
			}
			else{
			 
				for (int imageCounter = 0; imageCounter < numBoards; imageCounter++){

					strcpy(temp, imageName);
					strcat(temp, "left");					
					sprintf(temp1, "%d", imageCounter);
					strcat(temp, temp1);
					strcat(temp, ".png");					
					//cout << temp << endl;
					img1 = imread(temp, CV_LOAD_IMAGE_COLOR);
					//imwrite(temp, img1);
					strcpy(temp, imageName);
					strcat(temp, "right");
					sprintf(temp1, "%d", imageCounter);
					strcat(temp, temp1);
					strcat(temp, ".png");
					img2 = imread(temp, CV_LOAD_IMAGE_COLOR);
					
					cvtColor(img1, gray1, CV_BGR2GRAY);
					cvtColor(img2, gray2, CV_BGR2GRAY);
					
					// cout << "opa" << endl;

					found1 = findChessboardCorners(gray1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
					found2 = findChessboardCorners(gray2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
					
					if (found1)
					{
						cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
						drawChessboardCorners(img1, board_sz, corners1, found1);
					}

					if (found2)
					{
						cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
						drawChessboardCorners(img2, board_sz, corners2, found2);
					}
					
					imshow("image1",img1);
					imshow("image2",img2);
					//waitKey(0);

					if (found1 != 0 && found2 != 0)
					{
						imagePoints1.push_back(corners1);
						imagePoints2.push_back(corners2);
						object_points.push_back(obj);
						success++;	
					}
				}
			}

			cout << "success: " << success << endl;

			printf("Starting Calibration\n");
			
			stereoCalibrate(object_points, imagePoints1, imagePoints2,
				CM1, D1, CM2, D2, img1.size(), R, T, E, F,
				cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
				CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST);

			FileStorage fs1("minoru3d.yml", FileStorage::WRITE);
			
			fs1 << "CM1" << CM1;
			fs1 << "CM2" << CM2;
			fs1 << "D1" << D1;
			fs1 << "D2" << D2;
			fs1 << "R" << R;
			fs1 << "T" << T;
			fs1 << "E" << E;
			fs1 << "F" << F;

			printf("Done Calibration\n");

			stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, img1.size(), &roi1, &roi2);

			fs1 << "R1" << R1;
			fs1 << "R2" << R2;
			fs1 << "P1" << P1;
			fs1 << "P2" << P2;
			fs1 << "Q" << Q;
			fs1 << "roi1" << roi1;
			fs1 << "roi2" << roi2;
			fs1.release();

		}
		else{
			cout << "Loading Calibration Data" << endl;
			
			//FileStorage fs1("minoru3d___.yml", FileStorage::READ);
			FileStorage fs1("cameraartesanal.yml", FileStorage::READ);
			
			fs1["CM1"] >> CM1;
			fs1["CM2"] >> CM2;
			fs1["D1"] >> D1;
			fs1["D2"] >> D2;
			fs1["R"] >> R;
			fs1["T"] >> T;
			fs1["E"] >> E;
			fs1["F"] >> F;	
			fs1["roi1"] >> roi1;
			fs1["roi2"] >> roi2;
			
			printf("Loading Done\n");

			printf("Starting Rectification\n");

			fs1["R1"] >> R1;
			fs1["R2"] >> R2;
			fs1["P1"] >> P1;
			fs1["P2"] >> P2;
			fs1["Q"] >> Q;	

			saida << "CM1" << endl;
			saida << CM1 << endl;
			saida << "CM2" << endl;
			saida << CM2 << endl;
			saida << "D1" << endl;
			saida << D1 << endl;
			saida << "D2" << endl;
			saida << D2 << endl;
			saida << "R" << endl;
			saida << R << endl;
			saida << "T" << endl;
			saida << T << endl;
			saida << "E" << endl;
			saida << E << endl;
			saida << "F" << endl;
			saida << F << endl;
			saida << "roi1" << endl;
			saida << roi1 << endl;
			saida << "roi2" << endl;
			saida << roi2 << endl;
			saida << "R1" << endl;
			saida << R1 << endl;
			saida << "R2" << endl;
			saida << R2 << endl;
			saida << "P1" << endl;
			saida << P1 << endl;
			saida << "P2" << endl;
			saida << P2 << endl;
			saida << "Q" << endl;
			saida << Q << endl;
			
			//stereoRectify( CM1, D1, CM2, D2, imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, imageSize, &roi1, &roi2 );

			printf("Done Rectification\n");

			fs1.release();
		}

		printf("Applying Undistort\n");
		
		cout << P1 << endl;
		getchar();

		initUndistortRectifyMap(CM1, D1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
		
	  cout << P1 << endl;
		getchar();
		
		initUndistortRectifyMap(CM2, D2, R2, P2, imageSize, CV_32FC1, map2x, map2y);
		
		//cout << map1x.size() << endl;
		//getchar();
		
		//map1x.at
		
		//cout << map1x.at<uchar>(c, d)

		printf("Undistort complete\n");

		while (conditionWhile == 1)
		{

			if (dataLoadingMode == 0){


				cap1 >> img1;
				cap2 >> img2;

				//remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
				//remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

				//remap(img1, imgU1, map1x, map1y, INTER_LINEAR);
				//remap(img2, imgU2, map2x, map2y, INTER_LINEAR);

				//imshow("image1", imgU1);
				//imshow("image2", imgU2);

				//imgU1 = imgU1(roi1);
				//imgU2 = imgU2(roi2);

				//blur(imgU1, imgU1, Size(w, w));
				//blur(imgU2, imgU2, Size(w, w));

				//cv::waitKey();

				/*splitInterval = imgU1.rows / intervalo;

				for (int inter = 0; inter < intervalo; inter++){

				//cout << splitInterval * inter << endl;

				cv::Point2i p1(0, splitInterval * inter);
				cv::Point2i p2(imgU1.cols - 1, splitInterval * inter);

				cv::line(imgU1, p1, p2, cv::Scalar(1.0), 1, CV_AA); // 1 pixel thick, CV_AA == Anti-aliased flag
				cv::line(imgU2, p1, p2, cv::Scalar(1.0), 1, CV_AA); // 1 pixel thick, CV_AA == Anti-aliased flag

				}*/

				//cvtColor(imgU1, g1, CV_BGR2GRAY);
				//cvtColor(imgU2, g2, CV_BGR2GRAY);


				//cap1.release();
				//cap2.release();

			}
			else if (dataLoadingMode == 1){

				//cout << "ops" << endl;

				sprintf(strCountTestImage, "%d", countTestImage+1);

				strcpy(tempLeft, baseLeft);
				strcpy(tempRight, baseRight);
				
				strcat(tempLeft, "left");
				strcat(tempRight, "right");

				strcat(tempLeft, strCountTestImage);
				strcat(tempRight, strCountTestImage);

				strcat(tempLeft, ".png");
				strcat(tempRight, ".png");

				cout << tempLeft << endl;

				img1 = imread(tempLeft, CV_LOAD_IMAGE_COLOR);
				img2 = imread(tempRight, CV_LOAD_IMAGE_COLOR);
				
				//codigo pra gerar imagens ordenadas sem pular numero (codigo sem proposito aqui) 
				/*int conditionF = 1;
				int countPrintImage = 0;
				int countPrintImageBase = 50;
				
				while(countPrintImage < numberTestImage){
				
				  sprintf(strCountTestImage, "%d", countTestImage+1);

				  strcpy(tempLeft, baseLeft);
				  strcpy(tempRight, baseRight);
				
				  strcat(tempLeft, "left");
				  strcat(tempRight, "right");

				  strcat(tempLeft, strCountTestImage);
				  strcat(tempRight, strCountTestImage);

				  strcat(tempLeft, ".png");
				  strcat(tempRight, ".png");

				  img1 = imread(tempLeft, CV_LOAD_IMAGE_COLOR);
				  img2 = imread(tempRight, CV_LOAD_IMAGE_COLOR);
				
				  //cout << tempLeft << endl;
				  //cout << tempRight << endl;
				
				  //imshow("teste",img1);
				  //waitKey(0);
				
				  if((img1.empty() || img2.empty())){
				    //cout << "opa" << endl;   
				
				  }
				  else{
				  
				    cout << tempLeft << endl;
				    cout << tempRight << endl;
				
				
				    sprintf(strCountTestImage, "%d", countPrintImageBase+1);
				    
				    strcpy(tempLeft, baseLeft);
				    strcpy(tempRight, baseRight);
				    
				    strcat(tempLeft, "temp/");
				    strcat(tempRight, "temp/");
				    
				    strcat(tempLeft, "left");
				    strcat(tempRight, "right");
				    
				    strcat(tempLeft, strCountTestImage);
				    strcat(tempRight, strCountTestImage);
				    
				    strcat(tempLeft, ".png");
				    strcat(tempRight, ".png");
				    
				    //cout << tempLeft << endl;
				    //cout << tempRight << endl;
				    
				    imwrite(tempLeft, img1);
				    imwrite(tempRight, img2);
				    
				    cout << countPrintImage << endl;
				    
				    countPrintImageBase++;
				    
				    countPrintImage++;
				    
				    
				    
				    //countTestImage = (countTestImage + 1) % numberTestImage;
				  }
				
				  countTestImage++;// = (countTestImage + 1) % numberTestImage;
				  
				  
				} 
				
				exit(0);*/
				
				//conditionImage = 0;

				



				//remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
				//remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

				//initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map1x, map1y);
				//initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map2x, map2y);

				//remap(img1, imgU1, map1x, map1y, INTER_LINEAR);
				//remap(img2, imgU2, map2x, map2y, INTER_LINEAR);

				//imshow("image1", imgU1);
				//imshow("image2", imgU2);

				//imgU1 = imgU1(roi1);
				//imgU2 = imgU2(roi2);

				//img1 = img1 + Scalar(50, 50, 50);
				//img2 = img2 + Scalar(50, 50, 50);
				//reference = imread("disp6.png", CV_LOAD_IMAGE_COLOR);

				//Mat dst1,dst2;

				//cv::adaptiveBilateralFilter(img1, dst1, cv::Size(5, 5), 50);//kernal size(11) should be an odd value
				//cv::adaptiveBilateralFilter(img2, dst2, cv::Size(5, 5), 50);//kernal size(11) should be an odd value

				//cout << "opa" << endl;

				//cvtColor(imgU1, g1, CV_BGR2GRAY);
				//cvtColor(imgU2, g2, CV_BGR2GRAY);

			}
			else if (dataLoadingMode == 2){


				//capture >> frame;
				
				//imshow("w", frame);
				


				videoCapL >> img1;
				videoCapR >> img2;

				if (img1.empty() || img2.empty())
					break;

				

				//remap(img1, imgU1, map1x, map1y, INTER_LINEAR);
				//remap(img2, imgU2, map2x, map2y, INTER_LINEAR);

				//imshow("image1", imgU1);
				//imshow("image2", imgU2);

				//imgU1 = imgU1(roi1);
				//imgU2 = imgU2(roi2);

				//blur(imgU1, imgU1, Size(w, w));
				//blur(imgU2, imgU2, Size(w, w));

				//cv::waitKey();

				/*splitInterval = imgU1.rows / intervalo;

				for (int inter = 0; inter < intervalo; inter++){

				//cout << splitInterval * inter << endl;

				cv::Point2i p1(0, splitInterval * inter);
				cv::Point2i p2(imgU1.cols - 1, splitInterval * inter);

				cv::line(imgU1, p1, p2, cv::Scalar(1.0), 1, CV_AA); // 1 pixel thick, CV_AA == Anti-aliased flag
				cv::line(imgU2, p1, p2, cv::Scalar(1.0), 1, CV_AA); // 1 pixel thick, CV_AA == Anti-aliased flag

				}*/

				//cvtColor(imgU1, g1, CV_BGR2GRAY);
				//cvtColor(imgU2, g2, CV_BGR2GRAY);


				//cap1.release();
				//cap2.release();

			}
			
			remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
			remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
			
			imgU1 = imgU1(roi1);
			imgU2 = imgU2(roi2);

			/*imgU1 = img1;
			imgU2 = img2;*/

			if (dataLoadingMode == 1){

				strcpy(tempLeft, baseLeft);
				strcpy(tempRight, baseRight);
				
				strcat(tempLeft, "retificadas/");
				strcat(tempRight, "retificadas/");
				
				strcat(tempLeft, "left");
				strcat(tempRight, "right");

				strcat(tempLeft, strCountTestImage);
				strcat(tempRight, strCountTestImage);

				strcat(tempLeft, "R.png");
				strcat(tempRight, "R.png");
				
				


				imwrite(tempLeft, imgU1);
				imwrite(tempRight, imgU2);

				

			}
		  
		  //cvtColor( imgU1, imgU1, CV_BGR2YUV );
		  //cvtColor( imgU2, imgU2, CV_BGR2YUV );
			

			//GaussianBlur(imgU1, imgU1, Size(filterGaussianWindow, filterGaussianWindow), 0, 0);
			//GaussianBlur(imgU2, imgU2, Size(filterGaussianWindow, filterGaussianWindow), 0, 0);

			//adaptiveBilateralFilter(imgU1, g1, cv::Size(11, 11), 50);
			//adaptiveBilateralFilter(imgU2, g2, cv::Size(11, 11), 50);
			
			//bilateralFilter(imgU1, imgU1, 15, 10, 100);
			//bilateralFilter(imgU2, imgU2, 15, 10, 100);

			//cvtColor(imgU1, imgU1, CV_BGR2Lab)
			//cvtColor(imgU1, imgU1, CV_BGR2Lab);

			cvtColor(imgU2, g2, CV_BGR2GRAY);
			cvtColor(imgU1, g1, CV_BGR2GRAY);


			//medianBlur(g1, g1, 15);
			//medianBlur(g2, g2, 15);

			/*if (!(strcmp(method, "BM")))
			{*/
				//BMOpencv(&g1, &g2, &dispResult);
				//dispResult.convertTo(dispResult, CV_8U, 1.0 / 8);
			/*}
			else if (!(strcmp(method, "SGBM")))
			{*/

			//resize(g1, g1, Size(320, 240));
			//resize(g2, g2, Size(320, 240));
			
			//Sgbm sgbmobj(firstImage.rows,firstImage.cols,downScaleParam,dispRange,CLRDispMax,minDiff,filterLimitValue,wradiusY,wradiusX,P1,P2);
      //sgbmobj.runAB(g1, g2);

			//
			/*if(metodoDeExecucao == 1) {			
			  sgbmobj.runAB(g2, g1);
			  sgbmobj.disparityMapOut.convertTo(sgbmobj.disparityMapOut, CV_8U, 3.0);
			}
			else{*/
			SGBMOpencv(&g1, &g2, &dispResult);
			//normalize(dispResult, dispResult, 0, 255, CV_MINMAX, CV_8UC1);
			dispResult.convertTo(dispResult, CV_8U, 1.0 / 8);
			//}
			
			if (dataLoadingMode == 1){
			  strcpy(tempLeft, baseLeft);
			  
			  strcat(tempLeft, "opencv/");
			
				//strcpy(tempRight, baseRight);

				strcat(tempLeft, strCountTestImage);
				//strcat(tempRight, strCountTestImage);

				strcat(tempLeft, ".png");
				//strcat(tempRight, "R.png");			
				//cout << tempLeft << endl;
				
				/*if(metodoDeExecucao == 1) {			
			    imwrite(tempLeft, sgbmobj.disparityMapOut);
			  }
			  else{*/
			    imwrite(tempLeft, dispResult);
			  //}
				  
				//
				
				//cout << tempLeft << endl;
				
				countTestImage = (countTestImage + 1) % numberTestImage;
			
			}
			//normalize(dispResult, dispResult, 0, 255, CV_MINMAX, CV_8UC1);
			//resize(dispResult, dispResult, Size(640, 480));
			
			

				
			/*}
			else if (!(strcmp(method, "LOCALMETHODS")))
			{
				LOCALMETHODS(&g1, &g2, &dispResult);
			}
			else if (!(strcmp(method, "CENSUS")))
			{*/

				//GaussianBlur(g1, g1, Size(filterGaussianWindow, filterGaussianWindow), 0, 0);
				//GaussianBlur(g2, g2, Size(filterGaussianWindow, filterGaussianWindow), 0, 0);

				

				//resize(g1, g1, Size(160, 120));
				//resize(g2, g2, Size(160, 120));
								
				//censusMatchingCostCalculate(&g1, &g2, &matchCost, 0, 80);
				//census2DMatchingCostCalculate(&g1, &g2, &matchCost, 0, 80);
				//censusMatchingCostCalculate(&g1, &g2, &matchCost, &dispResult, 0, 120, 14, 14);
				//censusMatchingCostCalculatewithColor(&g1, &g2, &matchCost, &dispResult, 0, 120, 14, 14);
				//(&g1, &g2, &matchCost, &dispResult, 0, 80, 5, 5);
				//dispResult.convertTo(dispResult, CV_8U, 1.0 / 8);
				//disparityCalculate(&matchCost, &dispResult);
				//GaussianBlur(dispResult, dispResult, Size(filterGaussianWindow, filterGaussianWindow), 1.5, 1.5);
				//normalize(dispResult, dispResult, 0, 255, CV_MINMAX, CV_8UC1);
				
				//imwrite("resultrank.png", dispResult);
				//resize(dispResult, dispResult, Size(640, 480));
			/*}
			else if (!(strcmp(method, "SAD")))
			{
				//censusMatchingCostCalculate(&g1, &g2, &matchCost, 0, 80);
				//census2DMatchingCostCalculate(&g1, &g2, &matchCost, 0, 80);
				sadMatchingCostCalculate(&g1, &g2, &matchCost, &dispResult, 0, 80, 3, 3);
				//disparityCalculate(&matchCost, &dispResult);
				//GaussianBlur(dispResult, dispResult, Size(filterGaussianWindow, filterGaussianWindow), 1.5, 1.5);
				normalize(dispResult, dispResult, 0, 255, CV_MINMAX, CV_8UC1);
			}
			else if (!(strcmp(method, "SEMIGLOBALMETHODS")))
			{
				matchingCostCalculate(&g1, &g2, &matchCost, 0, 20);
				disparityCalculate(&matchCost, &dispResult, 0, 0);
			}
			else if (!(strcmp(method, "ELAS"))){
				elas(g1, g2, dispResult, 100);
				dispResult.convertTo(dispResult, CV_8U, 1.0 / 8);
			}*/

			//elas(g1, g2, dispResult, 100);
			//dispResult.convertTo(dispResult, CV_8U, 1.0 / 8);

			/*splitInterval = imgU1.rows / intervalo;

			for (int inter = 0; inter < intervalo; inter++){

			//cout << splitInterval * inter << endl;

			cv::Point2i p1(0, splitInterval * inter);
			cv::Point2i p2(imgU1.cols - 1, splitInterval * inter);

			cv::line(imgU1, p1, p2, cv::Scalar(1.0), 1, CV_AA); // 1 pixel thick, CV_AA == Anti-aliased flag
			cv::line(imgU2, p1, p2, cv::Scalar(1.0), 1, CV_AA); // 1 pixel thick, CV_AA == Anti-aliased flag

			}*/
			
			//Mat dispResultc;
			//vector <int> counter;
			//counter.resize(dispRange*3);
			
			//cv::Rect myROI(0, 0, img1.cols-20, img1.rows);
			
			//cv::Mat croppedImage = sgbmobj.disparityMapOut(myROI);
			
			//resize(croppedImage,croppedImage,Size(img1.cols,img1.rows));
			
			//cv::Mat imgresult(img1.rows * 2, img1.cols * 2, img1.type(), cv::Scalar(0, 0, 0));
			
			if(metodoDeExecucao == 1) {
			  /*int maiorValor = 70;
			  for(int c = 0; c < croppedImage.rows; c++){
			  for(int d = 0; d < croppedImage.cols; d++){
			    if(croppedImage.at<uchar>(c, d) > (maiorValor)){
			      segmentedImage.at<cv::Vec3b>(c,d)[0] = croppedImage.at<uchar>(c, d);
			      segmentedImage.at<cv::Vec3b>(c,d)[1] = croppedImage.at<uchar>(c, d);
			      segmentedImage.at<cv::Vec3b>(c,d)[2] = croppedImage.at<uchar>(c, d);   
			    }
			    else{
			      segmentedImage.at<cv::Vec3b>(c,d)[0] = 0;
			      segmentedImage.at<cv::Vec3b>(c,d)[1] = 0;
			      segmentedImage.at<cv::Vec3b>(c,d)[2] = 0;
			    }
			  }
			  }*/

			  /*if(d < (sgbmobj.disparityMapOut.cols - dispRange)){
			      counter.at(sgbmobj.disparityMapOut.at<uchar>(c, d))++;
			      //maiorValor = dispResult.at<uchar>(c, d);
			    }
			    if(sgbmobj.disparityMapOut.at<uchar>(c, d) > (limiar - 10) && d < (sgbmobj.disparityMapOut.cols - dispRange/3) ){
			      segmentedImage.at<cv::Vec3b>(c,d)[0] = 255;
			      segmentedImage.at<cv::Vec3b>(c,d)[1] = 255;
			      segmentedImage.at<cv::Vec3b>(c,d)[2] = 255;			      
          }
          else{
            segmentedImage.at<cv::Vec3b>(c,d)[0] = 0;
			      segmentedImage.at<cv::Vec3b>(c,d)[1] = 0;
			      segmentedImage.at<cv::Vec3b>(c,d)[2] = 0;
          }
			  }
			  }
			  int codicaoddddisp = 1;
			for(int ddddisp = dispRange - 1; ddddisp >= 0 && codicaoddddisp == 1; ddddisp--){
			  if(counter.at(ddddisp) > 50){
			    codicaoddddisp  =0;
			    limiar = ddddisp;
			    //counter.at(ddddisp)
			    
			  }
      }		*/
			  //limiar = maiorValor;		  
		    //cv::cvtColor(croppedImage, dispResultc, cv::COLOR_GRAY2BGR);
		  }
		  else{
		    /*int maiorValor = 0;
		    for(int c = 0; c < dispResult.rows; c++){
			  for(int d = 0; d < dispResult.cols; d++){
			    //cout << (int)dispResult.at<uchar>(c, d) << endl;
 			    //
			    if(d > 150){
			      counter.at(dispResult.at<uchar>(c, d))++;
			      //maiorValor = dispResult.at<uchar>(c, d);
			    }
			    if(dispResult.at<uchar>(c, d) > (limiar - 10) && d > 150){
			      segmentedImage.at<cv::Vec3b>(c,d)[0] = 255;
			      segmentedImage.at<cv::Vec3b>(c,d)[1] = 255;
			      segmentedImage.at<cv::Vec3b>(c,d)[2] = 255;			      
          }
          else{
            segmentedImage.at<cv::Vec3b>(c,d)[0] = 0;
			      segmentedImage.at<cv::Vec3b>(c,d)[1] = 0;
			      segmentedImage.at<cv::Vec3b>(c,d)[2] = 0;
          }
			  }
			}*/
			/*int codicaoddddisp = 1;
			for(int ddddisp = dispRange - 1; ddddisp >= 0 && codicaoddddisp == 1; ddddisp--){
			  if(counter.at(ddddisp) > 50){
			    codicaoddddisp  =0;
			    limiar = ddddisp;
			    //counter.at(ddddisp)
			    
			  }
      }*/
			  //limiar = maiorValor;
		    //cv::cvtColor(dispResult, dispResultc, cv::COLOR_GRAY2BGR);
		  }			
			//
			
			
			
			
			//Mat imgresult;
			
			//hconcat(g1, dispResultc, imgresult);
			
			
			/*if(metodoDeExecucao == 1) {			
		    cv::cvtColor(sgbmobj.disparityMapOut, dispResultc, cv::COLOR_GRAY2BGR);
		  }
		  else{
		    cv::cvtColor(dispResult, dispResultc, cv::COLOR_GRAY2BGR);
		  }	*/
			
			/*img1.copyTo(imgresult(Rect(0,0,img1.cols,img1.rows)));
			img2.copyTo(imgresult(Rect(dispResultc.cols,0,img2.cols,img2.rows)));
			dispResultc.copyTo(imgresult(Rect(0,img1.rows,dispResultc.cols,dispResultc.rows)));
			segmentedImage.copyTo(imgresult(Rect(img1.cols,img1.rows,img1.cols,img1.rows)));
			
			resize(imgresult, imgresult, Size(640, 480));*/
			
			/*merged_frame = Mat(Size(640 * 2, 480), CV_8UC1);
      roi = Mat(merged_frame, Rect(0, 0, 640, 480));
      dispResult.copyTo(roi);
      roi = Mat(merged_frame, Rect(640, 0, 640, 480));
      imgU1.copyTo(roi);
      imshow("roi",roi);*/
      //roi.copyTo(merged_frame);
      //imshow("merged", roi);
      //roi = Mat(merged_frame, Rect(640, 0, 640 * 2, 480));
      //img1.copyTo(roi);*/

			/*imshow("image1o", img1);
			imshow("image2o", img2);*/
			imshow("image1", imgU1);
			imshow("image2", imgU2);
			//imshow("disp", sgbmobj.disparityMapOut);
			//videoDisp.write(sgbmobj.disparityMapOut);
			//imshow("resultdips",imgresult);
			//imshow("segmentedImage",segmentedImage);
			//videoDisp.write(imgresult);

			imshow("disp", dispResult);

		  //imshow("disp", imgU1);	
			

			if (dataLoadingMode != 1) k = waitKey(1);
			else k = waitKey(0);

			if (k == 27)
			{
				break;
			}
			
		}
		

	//}

	return(0);
}

