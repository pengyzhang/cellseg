
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>
#include <limits>
#include <cmath>

using namespace std;
using namespace cv;

class initializationPhase {

	Mat result_intermediate;
	Mat input;

  public:
    
	initializationPhase(Mat im);
	vector<Mat> colordeconv(Mat I,Mat M,Mat stains);
	Mat preprocess_hemat_generate_vote(Mat hemat);
	Mat merge1(Mat input,Mat vote);
	Mat merge2(Mat input,Mat im);
	Mat im_32f_or_64f_to_8u(Mat _fpImage);
	Mat matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch);
	bool are_both_mats_same(Mat a, string filename,string variable_name);
  private:
	
	Mat colordeconv_normalize(Mat data);
	Mat im2vec(Mat I);
	Mat colordeconv_denormalize(Mat data);
	Mat complement_contrast_smoothen(Mat hemat);
	Mat diff_image(Mat smoothened);
	Mat voting_map_const(Mat pp);
	Mat peaks;
	string type2str(int type);
	Mat bwareaopen(Mat img, int size);
	template <class T>
	bool findValue(const cv::Mat &mat, T value);
	Mat squareform(Mat vector_mat);
	template <typename T>
	Mat ismember(Mat_<T> mat1, Mat_<T> mat2);
	Mat ismember_poly(Mat mat1, Mat mat2);
	template <typename T>
	Mat matlab_find(Mat_<T> mat1);
	Mat matlab_find_poly(Mat mat1);
	int matlab_min(Mat accuD);
	Mat matlab_pedge(Mat peaks, Mat edge_canny, Mat D, int minD);
	vector<double>linspace(double a, double b, int n);
	

};