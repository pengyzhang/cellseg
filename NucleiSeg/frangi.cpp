#include "frangi.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void frangi2d_hessian(const Mat &src, Mat &Dxx, Mat &Dxy, Mat &Dyy, float sigma){
	//construct Hessian kernels
	int n_kern_x = 2*round(3*sigma) + 1;
	int n_kern_y = n_kern_x;
	float *kern_xx_f = new float[n_kern_x*n_kern_y]();
	float *kern_xy_f = new float[n_kern_x*n_kern_y]();
	float *kern_yy_f = new float[n_kern_x*n_kern_y]();
	int i=0, j=0;
	for (int x = -round(3*sigma); x <= round(3*sigma); x++){
		j=0;
		for (int y = -round(3*sigma); y <= round(3*sigma); y++){
			kern_xx_f[i*n_kern_y + j] = 1.0f/(2.0f*M_PI*sigma*sigma*sigma*sigma) * (x*x/(sigma*sigma) - 1) * exp(-(x*x + y*y)/(2.0f*sigma*sigma));
			kern_xy_f[i*n_kern_y + j] = 1.0f/(2.0f*M_PI*sigma*sigma*sigma*sigma*sigma*sigma)*(x*y)*exp(-(x*x + y*y)/(2.0f*sigma*sigma));
			j++;
		}
		i++;
	}
	for (int j=0; j < n_kern_y; j++){
		for (int i=0; i < n_kern_x; i++){
			kern_yy_f[j*n_kern_x + i] = kern_xx_f[i*n_kern_x + j];
		}
	}

	//flip kernels since kernels aren't symmetric and opencv's filter2D operation performs a correlation, not a convolution
	Mat kern_xx;
	flip(Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xx_f), kern_xx, -1);
	
	Mat kern_xy;
	flip(Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xy_f), kern_xy, -1);

	Mat kern_yy;
	flip(Mat(n_kern_y, n_kern_x, CV_32FC1, kern_yy_f), kern_yy, -1);

	//specify anchor since we are to perform a convolution, not a correlation
	Point anchor(n_kern_x - n_kern_x/2 - 1, n_kern_y - n_kern_y/2 - 1);

	//run image filter
	filter2D(src, Dxx, -1, kern_xx, anchor);
	filter2D(src, Dxy, -1, kern_xy, anchor);
	filter2D(src, Dyy, -1, kern_yy, anchor);


	delete [] kern_xx_f;
	delete [] kern_xy_f;
	delete [] kern_yy_f;
}
void frangi2d_createopts(frangi2d_opts_t *opts){
	//these parameters depend on the scale of the vessel, depending ultimately on the image size...
	opts->sigma_start = DEFAULT_SIGMA_START;
	opts->sigma_end = DEFAULT_SIGMA_END;
	opts->sigma_step = DEFAULT_SIGMA_STEP;

	opts->BetaOne = DEFAULT_BETA_ONE; //ignore blob-like structures?
	opts->BetaTwo = DEFAULT_BETA_TWO; //appropriate background suppression for this specific image, but can change. 

	opts->BlackWhite = DEFAULT_BLACKWHITE; 
}		
Mat frangi2_eig2image(Mat Dxx, Mat Dxy, Mat Dyy){
	//calculate eigenvectors of J, v1 and v2
	Mat tmp, tmp2;
	tmp2 = Dxx - Dyy;
	tmp2.convertTo(tmp2, CV_32F);
	tmp.convertTo(tmp, CV_32F);
	Mat dxy_local = Dxy.clone();
	Mat dyy_local = Dyy.clone();
	Mat dxx_local = Dxx.clone();
	dxy_local.convertTo(dxy_local, CV_32F);
	dxx_local.convertTo(dxx_local, CV_32F);
	dyy_local.convertTo(dyy_local, CV_32F);
	sqrt(tmp2.mul(tmp2) + 4*dxy_local.mul(dxy_local), tmp);
	//compute eigenvalues
	Mat mu1 = 0.5*(dxx_local + dyy_local + tmp);
	return mu1;
}
Mat frangi2d_vote(const Mat &src, frangi2d_opts_t opts){
	Mat vote=Mat::zeros(src.size(),CV_8UC1);
	for (float sigma = opts.sigma_start; sigma <= opts.sigma_end; sigma += 0.3){
		//create 2D hessians
		Mat Dxx, Dyy, Dxy;
		frangi2d_hessian(src, Dxx, Dxy, Dyy, sigma);
		Dxx = Dxx*sigma*sigma;
		Dyy = Dyy*sigma*sigma;
		Dxy = Dxy*sigma*sigma;
	
		//calculate (abs sorted) eigenvalues and vector		
		Mat l1=frangi2_eig2image(Dxx, Dxy, Dyy);
 		Mat tf = l1 < 0;
		Mat one_add = Mat::ones(tf.size(), CV_8UC1);
		add(vote,one_add, vote, tf);
		//cout << "sigma= " << sigma << endl;
	}

	return vote;
}
