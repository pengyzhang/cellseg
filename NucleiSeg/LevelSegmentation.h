#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <limits>
#include <cmath>
#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_coding/sparse_coding.hpp>

#define M_PI 3.14159265358979323846

using namespace std;
using namespace cv;
using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::sparse_coding;

class LevelSegmentation {

	Mat result_intermediate;
	Mat input;
	Mat cu, cb;
	Mat trainingset;
	Mat allU;
	Mat peaks;
	vector<Mat> u,transform;
	Mat g;
	double mu, timestep;
	int xi, omega, nu, sigma,lambdaU, lambdaB, iter_outer, iter_inner, epsilon, c0;
	
public:

	LevelSegmentation(Mat im);
	void lse(Mat input, Mat trainingMat);
	vector<vector<Point>> clustered_contours;
private:

	void updateLSF(Mat g,vector<Mat> transform);
	void updateF();
	void updateSR();
	Mat readMat(string filename, string variable_name);
	Mat Heaviside(Mat x, int epsilon);
	Mat Dirac(Mat x, double epsilon);
	Mat NeumannBoundCond(Mat f);
	pair<Mat, Mat> gradient(Mat & img, float spaceX, float spaceY);
	Mat div_norm(Mat in);
	Mat distReg_p2(Mat phi);
	Mat divergence(Mat X, Mat Y);
	Mat post_process(Mat u, Mat peakX, Mat peakY);
	static Mat gradientX(Mat & mat, float spacing);
	static Mat gradientY(Mat & mat, float spacing);
	static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat1i &X, cv::Mat1i &Y);
	static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv, cv::Mat1i &X, cv::Mat1i &Y);
};