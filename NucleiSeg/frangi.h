#include <opencv2/opencv.hpp>

using namespace cv;

///Existing Implementation in C++ by: NTNU - Biomedical Optics 
///Modified for satisfying use-case
/*
*C++/OpenCV implementation of the Frangi multiscale vesselness filter in 2D
*(reference: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever,
*“Multiscale vessel enhancement filtering,”
*in Proc. Med. Image. Comput. Assist. Interv. 1496, pp. 130–137 (1998)).
*This code is based on a MATLAB implementation found at MATLAB Central.
*/
//options for the filter
typedef struct{
	//vessel scales
	int sigma_start;
	int sigma_end;
	int sigma_step;
	
	//BetaOne: suppression of blob-like structures. 
	//BetaTwo: background suppression. (See Frangi1998...)
	float BetaOne;
	float BetaTwo;

	bool BlackWhite; //enhance black structures if true, otherwise enhance white structures
} frangi2d_opts_t;

#define DEFAULT_SIGMA_START 3
#define DEFAULT_SIGMA_END 10
#define DEFAULT_SIGMA_STEP 0.3
#define DEFAULT_BETA_ONE 1.6
#define DEFAULT_BETA_TWO 0.08
#define DEFAULT_BLACKWHITE true
#define M_PI 3.14159265358979323846

/////////////////
//Frangi filter//
/////////////////

//apply full Frangi filter to src. Vesselness is saved in J, scale is saved to scale, vessel angle is saved to directions. 
cv::Mat frangi2d_vote(const cv::Mat &src, frangi2d_opts_t opts);



////////////////////
//Helper functions//
////////////////////

//run 2D Hessian filter with parameter sigma on src, save to Dxx, Dxy and Dyy. 
void frangi2d_hessian(const cv::Mat &src, cv::Mat &Dxx, cv::Mat &Dxy, cv::Mat &Dyy, float sigma);

//set opts to default options (sigma_start = 3, sigma_end = 7, sigma_step = 1, BetaOne = 1.6, BetaTwo 0.08)
void frangi2d_createopts(frangi2d_opts_t *opts);

//estimate eigenvalues from Dxx, Dxy, Dyy. Save results to lambda1, lambda2, Ix, Iy. 
cv::Mat frangi2_eig2image(Mat Dxx, Mat Dxy, Mat Dyy);

