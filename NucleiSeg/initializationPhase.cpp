#define HAVE_STRUCT_TIMESPEC

#include "initializationPhase.h"
#include "frangi.h"
#include "library/cvbloblib/BlobResult.h"



initializationPhase::initializationPhase(Mat im)
{
	input = im;
}

Mat initializationPhase::im_32f_or_64f_to_8u(Mat _fpImage) {

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(_fpImage, &minVal, &maxVal, &minLoc, &maxLoc);
	_fpImage -= minVal;
	Mat _8ucImage;
	_fpImage.convertTo(_8ucImage, CV_8U, 255 / (maxVal - minVal));

	return _8ucImage;
}

vector<Mat> initializationPhase::colordeconv(Mat I, Mat M, Mat stains)
{
	Mat diff_checker; vector<Mat> test;
	for (int i = 0; i < 3; i++)
	{
		if (norm(M.col(i)))
			M.col(i) /= norm(M.col(i));
	}
	if (norm(M.col(2)) == 0)
	{
		double x1 = pow(M.at<double>(0, 0), 2);
		double x2 = pow(M.at<double>(0, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(0, 2) = 0;
		else
		{
			M.at<double>(0, 2) = sqrt(1 - (x1 + x2));
		}
		x1 = pow(M.at<double>(1, 0), 2);
		x2 = pow(M.at<double>(1, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(1, 2) = 0;
		else
		{
			M.at<double>(1, 2) = sqrt(1 - (x1 + x2));
		}
		x1 = pow(M.at<double>(2, 0), 2);
		x2 = pow(M.at<double>(2, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(2, 2) = 0;
		else
		{
			M.at<double>(2, 2) = sqrt(1 - (x1 + x2));
		}
		M.col(2) /= norm(M.col(2));
	}
	cout << "M= " << endl << M << endl;
	Mat Q = (Mat_<double>(3, 3) << 4.8869, -0.7311, -3.9831, -4.3780, 1.8015, 3.5684, -0.0688, -0.4440, 1.3462);
	Q = M.inv(DECOMP_LU);
	cvtColor(I, I, CV_BGR2RGB);
	split(I, test);
	merge(test, I);
	Mat temp1 = im2vec(I), temp1_1, temp1_2;
	temp1.convertTo(temp1_2, CV_32F);
	Mat y_OD = colordeconv_normalize(temp1_2);
	y_OD.convertTo(y_OD, CV_64FC1);

	Q.convertTo(Q, CV_32FC1);
	Q.convertTo(Q, CV_64FC1);

	Mat C = Q*y_OD;
	Mat channel = colordeconv_denormalize(C);
	int m = I.rows; int n = I.cols;
	Mat intensity = Mat::zeros(I.size(), CV_32FC3), temp2;
	vector<Mat> splitCh;
	cv::split(intensity, splitCh);
	for (int i = 0; i < stains.cols; i++)
	{
		temp2 = channel.row(i);
		temp2.convertTo(temp2, CV_8UC1);
		splitCh[i] = matlab_reshape(temp2, m, n, 1);
	}
	merge(splitCh, intensity);

	vector<Mat> colorStainImages;
	Mat stain_OD, stain_RGB, temp3;
	for (int i = 0; i <3; i++)
	{
		stain_OD = M.col(i)*C.row(i);
		stain_RGB = colordeconv_denormalize(stain_OD);
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		minMaxLoc(stain_RGB, &minVal, &maxVal, &minLoc, &maxLoc);
		stain_RGB -= minVal;
		stain_RGB.convertTo(stain_RGB, CV_8U, 255 / (maxVal - minVal));
		minMaxLoc(stain_RGB, &minVal, &maxVal, &minLoc, &maxLoc);
		splitCh[0] = matlab_reshape(stain_RGB.row(0), m, n, 1);
		splitCh[1] = matlab_reshape(stain_RGB.row(1), m, n, 1);
		splitCh[2] = matlab_reshape(stain_RGB.row(2), m, n, 1);
		merge(splitCh, temp3);
		colorStainImages.push_back(temp3);
		temp3.release();
	}
	std::cout << colorStainImages.size() << endl;
	return colorStainImages;
}

Mat initializationPhase::preprocess_hemat_generate_vote(Mat hemat)
{
	Mat CCS = complement_contrast_smoothen(hemat);
	Mat diff = diff_image(CCS);
	CCS.release();
	diff.convertTo(diff, CV_32F);
	Mat vote_map = voting_map_const(diff);
	return vote_map;
}

Mat initializationPhase::squareform(Mat vector_mat)
{
	int a = vector_mat.cols;
	Mat temp, result;
	double d = (1 + sqrt(1 + 8 * a)) / 2;
	if (floor(d) == d)
	{
		d = (int)d;
		for (int i = 0; i < d; i++)
		{
			for (int j = 0; j < d; j++)
			{
				cout << "i: " << i << " j: " << j << endl;
				if (i != 0 && j != 0 && i != j)
					temp.push_back(vector_mat.at<int>(Point(i + j, 0)));
				else if (i != j)
					temp.push_back(vector_mat.at<int>(Point(i + j - 1, 0)));
				else
					temp.push_back(0);
			}
			temp = temp.t();
			result.push_back(temp);
			temp.release();
		}
		cout << "result = " << endl << result << endl;
		return result;
	}
	else {
		cout << "invalid input size" << endl;
		return Mat::zeros(Size(3, 3), CV_8UC1);
	}
}

int initializationPhase::matlab_min(Mat accuD)
{
	double min_no = INFINITY; int min_idx = 0;
	for (int i = 0; i < accuD.rows; i++)
	{
		if (accuD.at<double>(Point(0, i)) < min_no)
		{
			min_no = accuD.at<double>(Point(0, i));
			min_idx = i;

		}
	}
	return min_idx;
}

Mat initializationPhase::merge1(Mat input, Mat vote_map)
{
	Mat bin_image;
	Mat temp_input = input.clone();
	cvtColor(temp_input, temp_input, CV_BGR2GRAY);
	cout << "input  ready" << endl;
	cout << temp_input.type() << endl;
	threshold(temp_input, bin_image, 0, 255, THRESH_BINARY | THRESH_OTSU);
	cout << "threshold done" << endl;
	cout << bin_image.data << endl;
	Mat otsuBW = bwareaopen(bin_image, 150);
	double minD = 15, minA = 10, alpha = 10;
	Mat vote = matlab_reshape(vote_map.t(), vote_map.rows*vote_map.cols, 1, 1);
	Mat mt = vote.t();
	mt.convertTo(mt, CV_8UC1);
	vector<int> array(mt.rows*mt.cols);
	if (mt.isContinuous()) {
		array.assign(mt.datastart, mt.dataend);
	}
	else {
		for (int i = 0; i < mt.rows; ++i) {
			array.insert(array.end(), mt.ptr<int>(i), mt.ptr<int>(i) + mt.cols);
		}
	}
	std::set<int> c(array.begin(), array.end());
	std::vector<int> u;
	u.reserve(array.size());
	std::transform(array.begin(), array.end(), std::back_inserter(u),
		[&](int x)
	{
		return (std::distance(c.begin(), c.find(x)));
	});
	set<int>::iterator citer;
	mt.release();
	for (citer = c.begin(); citer != c.end(); citer++) {
		mt.push_back(*(citer));
	}
	Mat mtrev = mt.clone();
	mtrev.convertTo(mtrev, CV_32F);
	cv::sort(mt, mt, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
	Mat vnorm = (mtrev - mt.at<int>(0, mt.rows - 1)) / (float)(mt.at<int>(0, 0) - mt.at<int>(0, mt.rows - 1));
	Mat removeTF;
	for (int i = 0; i < mtrev.rows; i++)
	{
		Mat a = (vote_map >= mt.at<int>(0, i));
		a /= 255;
		const int connectivity_4 = 4;
		Mat labels, stats, centroids;
		int nLabels = connectedComponentsWithStats(a, labels, stats, centroids, connectivity_4, CV_32S);
		Mat merge_criteria1, areas_b, idxes;
		centroids = max(1, centroids);
		centroids.col(0) = min(vote_map.cols, centroids.col(0));
		centroids.col(1) = min(vote_map.rows, centroids.col(1));
		Mat otsuTF;
		double area_threshold = minA + exp(alpha*vnorm.at<float>(0, i));
		Mat removeL;
		for (int i1 = 0; i1 < centroids.rows; i1++)
		{
			//
			//centroids.convertTo(centroids, CV_8UC1);
			//cout << centroids << endl;
			//cout << "centroid x: " << (int)(centroids.col(0)).at<int>(1) << " centroid y: " << (int)(centroids.col(1)).at<int>(1) << endl;
			//cout << otsuBW.at<int>(Point((int)(centroids.col(0)).at<double>(i1), (int)centroids.col(1).at<double>(i1))) << endl;
			otsuTF.push_back(otsuBW.at<int>(Point((int)(centroids.col(0)).at<double>(i1), (int)centroids.col(1).at<double>(i1))));
			areas_b.push_back(stats.at<int>(i + 1, CC_STAT_AREA) > area_threshold);
			bitwise_and(areas_b, otsuTF, merge_criteria1);
			cout << "merge_criteria1 size " << merge_criteria1 << endl;
			otsuTF.release(); areas_b.release();
			if (merge_criteria1.at<int>(i1, 0) != 0)idxes.push_back(i1 + 1);
		}
		Mat b = ismember_poly(labels, idxes);
		//cout << b << endl;
		labels.release(), stats.release(), centroids.release();
		cout << type2str(b.type()) << endl;
		b.convertTo(b, CV_8UC1);
		cout << type2str(b.type()) << endl;
		nLabels = connectedComponentsWithStats(b, labels, stats, centroids, connectivity_4, CV_32S);
		int cand_length = centroids.rows;
		cout << "new " << cand_length << "center candidates at voting threshold: " << mt.at<int>(i) << endl;

		if (!peaks.rows == 0 && !peaks.cols == 0)
		{
			for (int i1 = 0; i1 < peaks.rows; i1++)
			{
				removeL.push_back(labels.at<int>(Point((peaks.col(0)).at<int>(i1), peaks.col(1).at<int>(i1))));
			}
			removeL = removeL.t();
			Mat iter;
			for (int i2 = 0; i2 < cand_length; i2++)
				iter.push_back(i2 + 1);
			iter = iter.t();
			removeTF = ismember_poly(iter, removeL);
			Mat temp_idx = matlab_find_poly(removeTF);
			Mat new_centroids; int prev = 0, curr = 0;
			for (int iwl = 0; iwl < temp_idx.cols; iwl++)
			{
				curr = temp_idx.at<int>(Point(iwl, 1)) - 1;
				for (int jwl = prev; jwl < curr; jwl++)
				{
					new_centroids.push_back(centroids);
				}
				new_centroids.push_back(centroids(Rect(0, prev, 2, curr - prev)));
				prev = curr + 1;
			}
			centroids = new_centroids.clone();
			new_centroids.release();
			cout << " reduce " << sum(removeTF) << " centers covered by connected component of upper level centers:" << endl;
		}
		else
		{
			cout << "removeTF is null" << endl;
		}

		centroids = max(1, centroids);
		centroids.col(0) = min(vote_map.cols, centroids.col(0));
		centroids.col(1) = min(vote_map.rows, centroids.col(1));
		Mat temp_peaks = peaks;
		temp_peaks.push_back(centroids);
		centroids = temp_peaks;
		int dist_length = 0;
		while (1)
		{
			Mat D;

			for (int i2 = 0; i2 < centroids.rows - 1; i2++)
			{
				for (int j2 = 1; j2 < centroids.rows; j2++)
				{
					D.push_back(cv::norm(Point((centroids.col(0)).at<double>(0, i2), centroids.col(1).at<double>(0, i2)) - Point((centroids.col(0)).at<double>(0, j2), centroids.col(1).at<double>(0, j2))));
				}
			}
			//cout << "D size: " << D.size() << " minD: " << minD.size() << endl;
			Mat comp = Mat::ones(D.size(), D.type());
			comp *= minD;
			Mat evaluate;
		    cv::compare(D, comp, evaluate, cv::CMP_GE);
			//comp.release();

			if (sum(evaluate).val[0] == D.rows)
				break;
			D = squareform(D);
			for (int i3 = 0; i3 < D.rows; i3++)
			{
				D.at<float>(i, i) = HUGE_VALF;
			}
			Mat mergeTF = D <= minD;
			Mat accuD = Mat::zeros(Size(1, D.rows - 1), CV_64F);
			accuD = INFINITY;
			Mat mergeTF_temp;
			for (int i4 = 0; i4 < D.rows - 1; i4++)
			{
				Mat lineMergeTF = Mat::zeros(Size(1, i4 + 1), CV_8UC1);
				mergeTF_temp = mergeTF(Rect(i4 + 1, i4, mergeTF.cols - i4 - 1, 1));
				mergeTF_temp = mergeTF_temp.t();
				lineMergeTF.push_back(mergeTF_temp);
				cout << "D= " << type2str(D.type()) << " centroids= " << type2str(centroids.type()) << endl;
				if (sum(lineMergeTF)[0] != 0)
				{
					Mat temp = matlab_find_poly(lineMergeTF);
					float sumatidex = 0;
					for (int iwl = 0; iwl < temp.cols; iwl++)
					{
						sumatidex += D.at<float>(Point(temp.at<int>(Point(iwl, 1)) - 1, i4));

					}
					accuD.at<double>(Point(0, i4)) = sumatidex;
				}

			}
			int minInd = matlab_min(accuD);

			Mat minLineMergeTF = Mat::zeros(Size(1, minInd), CV_8UC1);
			cout << "minLineMergeTF= " << type2str(minLineMergeTF.type()) << " mergeTF_temp= " << type2str(mergeTF_temp.type()) << " mergeTF= " << type2str(mergeTF.type()) << endl;
			minLineMergeTF.push_back(1);
			mergeTF_temp = mergeTF(Rect(minInd + 1, minInd, mergeTF.cols - minInd - 1, 1));
			for (int imtf = 0; imtf < mergeTF.cols; imtf++)
			{
				minLineMergeTF.push_back(mergeTF_temp.at<int>(Point(imtf, 0)));

			}
			Mat temp = matlab_find_poly(minLineMergeTF);
			float sumatidex = 0, sumatidey = 0;
			cout << Point(0, temp.at<int>(Point(1, 1))) << endl;
			cout << Point(1, temp.at<int>(Point(1, 1))) << endl;
			for (int iwl = 0; iwl < temp.cols; iwl++)
			{
				sumatidex += centroids.at<float>(Point(0, temp.at<int>(Point(iwl, 1))));
				sumatidey += centroids.at<float>(Point(1, temp.at<int>(Point(iwl, 1))));
			}
			sumatidex /= temp.cols; sumatidey /= temp.cols;
			Point cluster_c = Point(sumatidex, sumatidey);
			centroids.at<float>(Point(0, minInd)) = sumatidex;
			centroids.at<float>(Point(1, minInd)) = sumatidey;
			Mat remainingMergeTF = Mat::zeros(Size(1, minInd), CV_8UC1);
			mergeTF_temp = mergeTF(Rect(minInd + 1, minInd, mergeTF.cols - minInd - 1, 1));
			mergeTF_temp = mergeTF_temp.t();
			remainingMergeTF.push_back(mergeTF_temp);
			temp = matlab_find_poly(remainingMergeTF);
			Mat new_centroids; int prev = 0, curr = 0;
			for (int iwl = 0; iwl < temp.cols; iwl++)
			{
				curr = temp.at<int>(Point(iwl, 1)) - 1;
				new_centroids.push_back(centroids(Rect(0, prev, 2, curr - prev)));
				prev = curr + 1;
				
			}
			centroids = new_centroids.clone();
			std::cout << "cluster " << sum(minLineMergeTF) << "adjacent points by distance metric: minD= " << minD << endl;
			Scalar s1 = sum(minLineMergeTF);
			dist_length = dist_length + s1(0) - 1;
		}//while 1

		int orig_length = peaks.rows;
		peaks = centroids;
		cout << "After voting threshold" << mt.at<int>(0, i) << " : "
			<< orig_length << "(original)+" << cand_length << "(new candidate)-" << sum(removeTF) << "(connection)-" << dist_length << "(distance)=" << peaks.rows << endl << endl;
		if ((orig_length + cand_length - sum(removeTF)(0) - dist_length) != peaks.rows)
			cout << "error" << endl;

	}//for metrev
	return peaks;
}//merge1

Mat initializationPhase::merge2(Mat input, Mat im)
{
	Mat edge_canny;
	Mat gray_sc;
	cvtColor(im, gray_sc, CV_BGR2GRAY);
	Canny(gray_sc, edge_canny, 50, 150, 3);
	Mat peaks_stage1 = input;
	cout << "----------------------------------------------------------------" << endl << endl;
	cout << "Stage 2: Merge peaks with distance and canny edge constraints..." << endl;
	int minD = 25;
	int canny_dist_length = 0;
	while (1)
	{
		Mat D;

		for (int i2 = 0; i2 < input.rows - 1; i2++)
		{
			for (int j2 = 1; j2 < input.rows; j2++)
			{
				D.push_back(cv::norm(Point((input.col(0)).at<int>(i2), input.col(1).at<int>(i2)) - Point((input.col(0)).at<int>(j2), input.col(1).at<int>(j2))));
			}
		}
		Mat edgeTF = matlab_pedge(peaks, edge_canny, D, minD);
		Mat evaluate = D > minD;
		bitwise_or(evaluate, edgeTF, evaluate);
		if (sum(evaluate).val[0] == D.rows)
			break;

		D = squareform(D);
		edgeTF = squareform(edgeTF);
		for (int i3 = 0; i3 < D.rows; i3++)
		{
			D.at<float>(i3, i3) = HUGE_VALF;
		}
		Mat mergeTF = D <= minD;
		for (int i3 = 0; i3 < D.rows; i3++)
		{
			edgeTF.at<float>(i3, i3) = false;
		}
		Mat accuD = Mat::zeros(Size(1, D.rows - 1), CV_32F);
		accuD = INFINITY;
		Mat mergeTF_temp, edgeTF_temp;
		for (int i4 = 0; i4 < D.rows - 1; i4++)
		{
			Mat lineMergeTF = Mat::zeros(Size(1, i4 + 1), CV_8UC1);
			mergeTF_temp = mergeTF(Rect(i4 + 1, i4, mergeTF.cols - i4 - 1, 1));
			mergeTF_temp = mergeTF_temp.t();
			lineMergeTF.push_back(mergeTF_temp);
			Mat lineEdgeTF = Mat::zeros(Size(1, i4 + 1), CV_8UC1);
			edgeTF_temp = edgeTF(Rect(i4 + 1, i4, edgeTF.cols - i4 - 1, 1));
			edgeTF_temp = edgeTF_temp.t();
			lineEdgeTF.push_back(edgeTF_temp);
			Mat evaluate;
			bitwise_not(lineEdgeTF, lineEdgeTF);
			bitwise_and(lineMergeTF, lineEdgeTF, evaluate);
			if (sum(evaluate)[0] != 0)
			{
				Mat temp = matlab_find_poly(evaluate);
				float sumatidex = 0;
				for (int iwl = 0; iwl < temp.cols; iwl++)
				{
					sumatidex += D.at<float>(Point(temp.at<int>(Point(iwl, 1)) - 1, i4));

				}
				accuD.at<double>(Point(0, i4)) = sumatidex;
			}
		}
		int minInd = matlab_min(accuD);

		Mat minLineMergeTF = Mat::zeros(Size(1, minInd), CV_8UC1);
		minLineMergeTF.push_back(1);
		Mat temp_eval;
		bitwise_not(edgeTF, edgeTF);
		bitwise_and(mergeTF, edgeTF, temp_eval);
		mergeTF_temp = temp_eval(Rect(minInd + 1, minInd, mergeTF.cols - minInd - 1, 1));
		mergeTF_temp = mergeTF_temp.t();
		minLineMergeTF.push_back(mergeTF_temp);
		Mat temp = matlab_find_poly(minLineMergeTF);
		float sumatidex = 0, sumatidey = 0;
		for (int iwl = 0; iwl < temp.cols; iwl++)
		{
			sumatidex += input.at<float>(Point(0, temp.at<int>(Point(iwl, 1))));
			sumatidey += input.at<float>(Point(1, temp.at<int>(Point(iwl, 1))));
		}
		sumatidex /= temp.cols; sumatidey /= temp.cols;
		Point cluster_peak = Point(sumatidex, sumatidey);
		peaks_stage1.at<float>(Point(0, minInd)) = sumatidex;
		peaks_stage1.at<float>(Point(1, minInd)) = sumatidey;
		Mat remainingMergeTF = Mat::zeros(Size(1, minInd), CV_8UC1);
		mergeTF_temp = temp_eval(Rect(minInd + 1, minInd, temp_eval.cols - minInd - 1, 1));
		mergeTF_temp = mergeTF_temp.t();
		remainingMergeTF.push_back(mergeTF_temp);
		temp = matlab_find_poly(remainingMergeTF);
		Mat new_centroids; int prev = 0, curr = 0;
		for (int iwl = 0; iwl < temp.cols; iwl++)
		{
			curr = temp.at<int>(Point(iwl, 1)) - 1;
			new_centroids.push_back(peaks_stage1(Rect(0, prev, 2, curr - prev)));
			prev = curr + 1;
			// centroids.at<float>(Point(0,curr));
			// centroids.at<float>(Point(1, temp.at<int>(Point(iwl, 1))));
		}
		peaks_stage1 = new_centroids.clone();
		std::cout << "cluster " << sum(minLineMergeTF).val[0] << "adjacent points by distance metric: minD= " << minD << endl;
		canny_dist_length = canny_dist_length + sum(minLineMergeTF).val[0] - 1;
		//dist_length = dist_length + s1(0) - 1;

	}

	cout << peaks_stage1.rows << " (original)- " << canny_dist_length << " (distance)= " << peaks_stage1.rows - canny_dist_length << "(final)" << endl;
	Mat stage1TF = Mat::ones(Size(peaks_stage1.rows, 1), CV_8UC1);
	Mat xTF, yTF;
	for (int ixtf = 0; ixtf < peaks_stage1.rows; ixtf++)
	{
		xTF = input.col(0) == peaks_stage1.at<int>(Point(0, ixtf));
		if (sum(xTF).val[0] == 0)
		{
			stage1TF.at<int>(Point(0, ixtf)) = 0;
			continue;
		}
		Mat temp_ytf = matlab_find_poly(input);
		for (int iytf = 0; iytf < temp_ytf.cols; iytf++)
		{
			yTF.push_back(input.at<int>(Point(1, temp_ytf.at<int>(Point(iytf, 0)))) == peaks_stage1.at<int>(Point(1, ixtf)));
		}
		if (sum(yTF).val[0] == 0)
			stage1TF.at<int>(0, ixtf) = 0;

	}
	return peaks_stage1;
}

template <class T>
bool initializationPhase::findValue(const cv::Mat &mat, T value) {
	for (int i = 0; i < mat.rows; i++) {
		const T* row = mat.ptr<T>(i);
		if (std::find(row, row + mat.cols, value) != row + mat.cols)
			return true;
	}
	return false;
}

template <typename T>
Mat initializationPhase::ismember(Mat_<T> mat1, Mat_<T> mat2)
{
	Mat oldMat = mat1;
	Mat newMat = Mat::zeros(oldMat.size(), oldMat.type());
	for (int r = 0; r < newMat.rows; r++)
	{
		for (int c = 0; c < newMat.cols; c++)
		{
			if (findValue(mat2, oldMat.at<T>(Point(c, r))))
				newMat.at<T>(Point(c, r)) = 1;
			else
				newMat.at<T>(Point(c, r)) = 0;
		}
	}
	return newMat;
}

Mat initializationPhase::ismember_poly(Mat mat1, Mat mat2)
{
	switch (mat1.type())
	{
	case CV_8U:
		cout << "case 1" << endl;
		return ismember<int>(mat1, mat2);
		break;
	case CV_32F:
		cout << "case 2" << endl;
		return ismember<float>(mat1, mat2);
		break;
	case CV_64F:
		cout << "case 3" << endl;
		return ismember<double>(mat1, mat2);
		break;
	default:
		cout << "default" << endl;
		return ismember<int>(mat1, mat2);
	}

}

template <typename T>
Mat initializationPhase::matlab_find(Mat_<T> mat1)
{
	Mat find_result;
	for (int r = 0; r < mat1.rows; r++)
	{
		for (int c = 0; c < mat1.cols; c++)
		{
			if (mat1.at<T>(Point(c, r)) != 0)
				find_result.push_back(c + r + 1);
		}
	}
	return find_result.t();
}

Mat initializationPhase::matlab_find_poly(Mat mat1)
{
	switch (mat1.type())
	{
	case CV_8U:
		return matlab_find<int>(mat1);
	case CV_32F:
		return matlab_find<float>(mat1);
	case CV_64F:
		return matlab_find<double>(mat1);
	default:
		cout << "default" << endl;
		return matlab_find<int>(mat1);

	}
}

vector<double> initializationPhase::linspace(double a, double b, int n) {
	vector<double> array;
	double epsilon = 0.0001;
	double step = (b - a) / (n - 1);
	if (a == b || n == 0 || n == 1)
	{
		for (int i = 0; i < n; i++)
		{
			array.push_back(a);
		}
	}
	else if (step >= 0)
	{
		while (a <= b + epsilon)
		{
			array.push_back(a);
			a += step;
		}
	}
	else
	{
		while (a + epsilon >= b)
		{
			array.push_back(a);
			a += step;
		}
	}
	return array;
}

Mat initializationPhase::matlab_pedge(Mat peaks, Mat edge_canny, Mat D, int minD)
{
	Mat LongDisTF = D > minD;
	int M = edge_canny.cols;
	int N = edge_canny.rows;
	int m = peaks.cols;
	int no_of_cols = m*(m - 1) / 2;
	Mat edgeTF = Mat::zeros(Size(no_of_cols, 1), CV_8UC1);
	float unit_d = 0;
	int ind = 1;
	for (int i = 0; i < m - 1; i++)
	{
		Mat p2 = peaks.row(i);
		for (int j = i + 1; j < m; j++)
		{
			if (LongDisTF.at<int>(Point(ind, 0))) {
				ind += 1;
				continue;
			}
			Mat p1 = peaks.row(j);
			Mat p1p2;
			pow(p1 - p2, 2, p1p2);
			Scalar s_sump1p2 = sum(p1p2);
			float v_sump1p2 = s_sump1p2(0);
			v_sump1p2 = sqrt(v_sump1p2);
			int n = ceil(v_sump1p2 / unit_d) + 2;
			vector<double> x = linspace(p1.at<int>(Point(0, 0)), p2.at<int>(Point(0, 0)), n);
			Mat x_mat;
			for (int ix = 0; ix < x.size(); ix++) {
				x[ix] = round(x[ix]);
				x_mat.push_back(x[ix]);
			}
			min(Mat::ones(x_mat.size(), x_mat.type()), x_mat, x_mat);
			max(Mat::ones(x_mat.size(), x_mat.type())*N, x_mat, x_mat);
			vector<double> y = linspace(p1.at<int>(Point(1, 0)), p2.at<int>(Point(1, 0)), n);
			Mat y_mat;
			for (int iy = 0; iy < y.size(); iy++) {
				y[iy] = round(y[iy]);
				y_mat.push_back(y[iy]);
			}
			min(Mat::ones(y_mat.size(), y_mat.type()), y_mat, y_mat);
			max(Mat::ones(y_mat.size(), y_mat.type())*M, y_mat, y_mat);
			Mat temp_edge_canny, temp;
			int y_idx, x_idx;
			for (int iy1 = 0; iy1 < y_mat.rows; iy1++)
			{
				y_idx = y_mat.at<int>(Point(0, iy1));
				for (int ix1 = 0; ix1 < x_mat.rows; ix1++)
				{
					x_idx = x_mat.at<int>(Point(0, ix1));
					temp.push_back(edge_canny.at<double>(Point(x_idx, y_idx)));
				}
				temp = temp.t();
				temp_edge_canny.push_back(temp);
				temp.release();
			}
			int diag_idx = 0;
			if (temp_edge_canny.cols >= temp_edge_canny.rows)
				diag_idx = temp_edge_canny.cols;
			else
				diag_idx = temp_edge_canny.rows;
			Mat TF;
			for (int tf_fill = 0; tf_fill < diag_idx; tf_fill++)
			{
				TF.push_back(temp_edge_canny.at<double>(Point(diag_idx, diag_idx)));
			}
			for (int mf = 0; mf < TF.cols; mf++)
			{
				Mat chk = matlab_find_poly(TF.col(mf));
				if (chk.cols >= 1)
				{
					edgeTF.at<int>(Point(ind, 0));
					break;
				}
			}
		}
		ind += 1;
	}
	return edgeTF;
}

Mat initializationPhase::im2vec(Mat I)
{
	int M = I.rows;
	int N = I.cols;
	vector<Mat> splitted;
	Mat vec;
	Mat temp;

	if (I.channels() == 3)
	{
		cv::split(I, splitted);
		for (int i = 0; i < 3; i++)
		{
			transpose(splitted[i], splitted[i]);
			temp = splitted[i].reshape(0, 1);
			temp.convertTo(temp, CV_64F);
			vec.push_back(temp);
		}
	}
	else if (I.channels() == 1)
	{
		transpose(I, I);
		I.convertTo(I, CV_64F);
		vec.push_back(I.reshape(0, 1));
	}
	else
	{
		vec.push_back(Mat::zeros(Size(1, 1), CV_64FC1));
	}
	std::cout << "im2vec size: " << vec.size() << endl;
	return vec;
}

Mat initializationPhase::colordeconv_normalize(Mat data)
{
	Mat denorm_deconv;
	double epsd = numeric_limits<double>::epsilon();
	data += epsd; data /= 255;
	log(data, denorm_deconv);
	denorm_deconv *= -1;
	return denorm_deconv;

}

Mat initializationPhase::colordeconv_denormalize(Mat data)
{
	Mat denorm_deconv;
	exp(-data, denorm_deconv);
	denorm_deconv *= 255;
	return denorm_deconv;
}

Mat initializationPhase::complement_contrast_smoothen(Mat hemat)
{
	Mat result, G, h;
	int kernel_size = 4;
	Mat gray_hemat;
	double alpha = 3.0; int beta = 30;
	if (hemat.channels()>1)
		cvtColor(hemat, gray_hemat, CV_BGR2GRAY);
	result = Mat::zeros(gray_hemat.size(), gray_hemat.type());
	cout << type2str(gray_hemat.type()) << endl;
	for (int y = 0; y < gray_hemat.rows; y++)
	{
		for (int x = 0; x < gray_hemat.cols; x++)
		{
			result.at<uchar>(y, x) =
				saturate_cast<uchar>(alpha*(gray_hemat.at<uchar>(y, x)) + beta);
		}
	}

	h = 255 - result;
	GaussianBlur(h, G, Size(2 * kernel_size + 1, 2 * kernel_size + 1), 1);
	return G;
}

Mat initializationPhase::diff_image(Mat smoothened)
{
	Mat result;
	int morph_size = 6;
	int morph_elem = MORPH_ELLIPSE;
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1));
	morphologyEx(smoothened, result, MORPH_OPEN, element);
	result = abs(smoothened - result);
	return result;
}

Mat initializationPhase::voting_map_const(Mat pp) {
	frangi2d_opts_t opts;
	frangi2d_createopts(&opts);
	Mat vesselness, scale, angles;
	return frangi2d_vote(pp, opts);

}

Mat initializationPhase::matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch)
{
	int old_row, old_col, old_ch;
	old_row = m.size().height;
	old_col = m.size().width;
	old_ch = m.channels();

	Mat m1(1, new_row*new_col*new_ch, m.depth());

	vector <Mat> p(old_ch);
	cv::split(m, p);
	for (int i = 0; i<p.size(); ++i) {
		Mat t(p[i].size().height, p[i].size().width, m1.type());
		t = p[i].t();
		Mat aux = m1.colRange(i*old_row*old_col, (i + 1)*old_row*old_col).rowRange(0, 1);
		t.reshape(0, 1).copyTo(aux);
	}

	vector <Mat> r(new_ch);
	for (int i = 0; i<r.size(); ++i) {
		Mat aux = m1.colRange(i*new_row*new_col, (i + 1)*new_row*new_col).rowRange(0, 1);
		r[i] = aux.reshape(0, new_col);
		r[i] = r[i].t();
	}

	Mat result;
	merge(r, result);
	return result;
}

bool initializationPhase::are_both_mats_same(Mat a, string filename, string variable_name)
{
	Mat local;
	FileStorage fs_in(filename, FileStorage::READ);
	fs_in[variable_name] >> local;
	cout << "opencv_type= " << endl << a.type() << "matlab_type= " << endl << local.type() << endl;
	cout << "opencv_size= " << endl << a.size() << "matlab_size= " << endl << local.size() << endl;
	local.convertTo(local, a.type());
	if (local.size() == a.size())
	{
		Scalar diff = sum(local - a);
		if (diff == Scalar(0, 0, 0, 0))
			return true;
		else {
			cout << "diff= " << endl << diff << endl;
			return false;
		}
	}
	else
		return false;
}

string initializationPhase::type2str(int type) {
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

Mat initializationPhase::bwareaopen(Mat img, int size)
{
	CBlobResult blobs;
	blobs = CBlobResult(img, Mat(), 4);
	blobs.Filter(blobs, B_INCLUDE, CBlobGetLength(), B_GREATER, size);

	Mat newimg(img.size(), img.type());
	newimg.setTo(0);
	for (int i = 0; i<blobs.GetNumBlobs(); i++)
	{
		blobs.GetBlob(i)->FillBlob(newimg, CV_RGB(255, 255, 255), 0, 0, true);
	}
	img = newimg;

	return newimg;
}