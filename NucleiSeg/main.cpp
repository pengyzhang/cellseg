#include "initializationPhase.h"

#include <string>
#include <cstdio>
#include <iostream>
#include <set>
using namespace std;

int main() {

	string filename;
	for (int i = 1; i <=1; i++)
	{   
		if(i<=9)
		filename = "G:/Ankita_Workspace/GSOC/NucleiSeg/512image/0" + to_string(i) + ".tif";
		else
		filename = "G:/Ankita_Workspace/GSOC/NucleiSeg/512image/" + to_string(i) + ".tif";

		cout << filename << endl;
		Mat input = imread(filename);
		if (!input.data)
		{
			cout << "Image not present" << endl;
			exit(-1);

		}
		initializationPhase ip = initializationPhase(input);
		Mat M = (Mat_<double>(3, 3) << 0.5547,0.3808,0,0.7813,0.8721,0,0.2861,0.3071,0);
		/*Mat M = (Mat_<int>(3, 3) << 1,2, 9, 5, 9, 2, 8, 3, 1);
		cout << " M before = " << endl << M << endl;*/
		vector<Mat> deconv;
   		deconv=ip.colordeconv(input, M, Mat::ones(Size(3,1), CV_8UC1));
		imwrite("Hemat_" + to_string(i) + ".png", deconv[0]);
		imwrite("Eosin_" + to_string(i) + ".png", deconv[1]);
		Mat im = imread("Hemat_" + to_string(i) + ".png");
		Mat result=ip.preprocess_hemat_generate_vote(im);
		cout << ip.are_both_mats_same(result, "./vote.yml", "vote") << endl;
		//imwrite("voting_map_" + to_string(i) + ".png", result);

		Mat merge_stage1 = ip.merge1(im, result);
		Mat merge_stage2 = ip.merge2(merge_stage1, im);
		//cout << "centroids: " << merge_stage2 << endl;.
		ip.lse();
		vector<vector<Point>> contours = ip.clustered_contours;
		drawContours(im, contours, 100, Scalar(0, 0, 255), 3);

		imwrite("contour_map_" + to_string(i) + ".png",im);
		
		
	
	}
	


}