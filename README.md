# cellseg


Files in Nuclei Segmentation Project :

Main.cpp
initializationPhase.cpp
initializationPhase.h
LevelSegmentation.cpp
LevelSegmentation.h
Frangi.h
Frangi.cpp

Libraries Used:
OpenCV 3.2
Libfrangi
Pthread Library
Opencv Blobs Library
Mlpack Library
Armadillo Library


Description of the files and functions used in the Project :
Main.cpp :

Defines the entry point of the application, by calling all the functions related to the nuclei segmentation like the initialization Phase for getting the seed centers etc. and level segmentation for getting the contours.

initializationPhase.h :
Keeps the function definitions related to the color deconvolution, preprocessing and getting Seeds.

initializationPhase.cpp :
vector<Mat> initializationPhase::colordeconv(Mat I, Mat M, Mat stains)
Input: Mat I= Input RGB Image
       Mat M= stain matrix
       Mat stains= Which stain to use
Output: vector of two images one contains the Hematoxylin Image, other contains the Eosin image.


Mat initializationPhase::preprocess_hemat_generate_vote(Mat hemat)
It calls the function related to pre-processing needed to generate vote_map.
Input: Mat hemat = Hematoxylin Image
Output: Vote Map Image


Mat initializationPhase::voting_map_const(Mat pp)
Calls libfrangi library with necessary modification to calculate the vote map.
Input: Pre-processed Image
Output: Vote Map



Mat initializationPhase::merge1(Mat input,Mat vote_map)
Stage 1: Centers covered by connected component of upper level contours.
Input: Input Hemat Image, Vote Map Image
Output: Centroids after Merge Step1

Mat initializationPhase::merge2(Mat input,Mat im)
Stage 2: Merge peaks with distance and canny edge constraints.
Input: Input Hemat Image, Vote Map Image
Output: Centroids after Merge Step2

Mat initializationPhase::complement_contrast_smoothen(Mat hemat)
Input: Hematoxylin Image
Output: Pre-processed Image
Complements the input image, increase the contrast using alpha=3.0 and beta =30 saturate_contrast function.
Further smoothening image using Gaussian Blur of kernel size=7.

Mat initializationPhase::diff_image(Mat smoothened)
Input:Pre-processed Image
Output: Calculating the diff image.
The diff image is calculated by subtracting the result obtained by morphological reconstruction from pre-processed image. Morphological Element=ellipse Radius=6

Mat initializationPhase::colordeconv_normalize(Mat data)
Input: RGB Image
Output: Normalized Image
Using Euler number as the c++ default value and max pixel value as 255 for normalizing.

Mat initializationPhase::colordeconv_denormalize(Mat data)
Input: Normalized Image
Output: De Normalized Image
Using Euler number as the c++ default value and max pixel value as 255 for denormalizing.

Mat initializationPhase::bwareaopen(Mat img, int size)
C++ implementation of matlab bwareaopen.


string initializationPhase::type2str(int type) 
Input: Integer defined by MACRO Output: string describing the type of matrix
Used for comparing types, and accessing matrix of any type.


Mat initializationPhase::im2vec(Mat I)
C++ implementation of matlab im2vec, converts the 2d image to a linear vector.

vector<double> initializationPhase::linspace(double a, double b, int n)
C++ implementation of matlab linspace, creates uniformly spaced linear space array.

Mat initializationPhase::matlab_find_poly(Mat mat1)
Mat initializationPhase::matlab_find(Mat_<T> mat1)
C++ implementation of MATLAB’s find_poly function (poly is made to cater all datatypes)

Mat initializationPhase::ismember_poly(Mat mat1, Mat mat2)
Mat initializationPhase::ismember(Mat_<T> mat1, Mat_<T> mat2)
C++ implementation of MATLAB’s ismember function (poly is made to cater all datatypes)

bool initializationPhase::findValue(const cv::Mat &mat, T value)
C++ Implementation of Matlab’s findValue (Tells whether the value is present in the mat or not)

bool initializationPhase::are_both_mats_same(Mat a, string filename, string variable_name)
Compares the Matrices.(Used for debugging purposes to compare the results from OpenCV with that to Matlab.)



Mat initializationPhase::matlab_pedge(Mat peaks, Mat edge_canny, Mat Don't minD) -
C++ implementation for MATLAB pedge.

Mat initializationPhase::matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch) -
C++ implementation for MATLAB reshape.


Mat LevelSegmentation::div_norm(Mat in)  - 
Compute curvature for u with central difference scheme

static Mat gradientY(Mat & mat, float spacing), static Mat gradientX(Mat & mat, float spacing)
Compute the gradient in Y and X directions, respectively.
/// Internal method to get numerical gradient for x components. 
/// @param[in] mat Specify input matrix.
/// @param[in] spacing Specify input space.



static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat1i &X, cv::Mat1i &Y) -
Creates a meshgrid by repeating the two input vectors. Based on the MATLAB’s implementation.

pair<Mat, Mat> LevelSegmentation::gradient(Mat & img, float spaceX, float spaceY) -
Computes gradient using gradientX, gradientY.


void LevelSegmentation::updateLSF(Mat g,vector<Mat> transform) -
Computes imageTerm, lengthTerm, exclusiveTerm and penlaiseTerm.

void LevelSegmentation::updateF() - 
Computes Adaptive Occlusion Penalty Term.

void LevelSegmentation::updateSR() - 
Computes Sparse Reconstruction Term

Mat LevelSegmentation::readMat(string filename, string variable_name) -
Reads MATLAB’s .mat files into C++ Code using MATLABs header files.
Input: filename= Path of the .MAT file to be read.
variable_name= name of the variable to be read from that file.
Output: cv::Mat of the matrix intended to be read.

Mat LevelSegmentation::Heaviside(Mat x, int epsilon) -
C++ Implementation of MATLAB’s Heaviside function

Mat LevelSegmentation::Dirac(Mat x, double epsilon) -
C++ Implementation of MATLAB’s Dirac function

Mat LevelSegmentation::NeumannBoundCond(Mat f) -
Make a function satisfy Neumann boundary condition

Mat LevelSegmentation::distReg_p2(Mat phi) -
Compute the distance regularization term with the double - well potential p2 in equation(16)

Mat LevelSegmentation::post_process(Mat u, Mat peakX, Mat peakY) -
Computes the distance of each contour from the peaks and removes the ones within the biasThreshold of the RMS distance.

Description of Setup in a System:

Pre-requisites:
Install the latest version of OpenCV.
Install the latest version of mlpack. (installation of armadillo included in the link)
Install cvblobslib.
Install pthread.

Steps:
Clone the code from the repository.
In Visual Studio 2017, modify the properties of project-
In C/C++ section-> General -> Add Include Directories.
          i)OpenCV_PATH/include
          ii)mlpack_PATH/include
          iii)pthread_PATH/include
          iv)MATLAB_PATH/extern/include
          v) CVBLOBSLIB_PATH/include
          vi)armadillo_PATH/include
Add the corresponding libraries in the linker settings of properties.
       In Linker Section -> General -> Add library directories
          i)OpenCV_PATH/x64/lib
          ii)mlpack_PATH/lib
          iii)pthread_PATH/lib
          iv) CVBLOBSLIB_PATH/lib
          v)armadillo_PATH/lib
Add in additional dependencies section ->
      Opencv_world320.lib
      Opencvblobslib.lib
      pthreadVC2.lib
      Armadillo.lib
      mlpack.lib
Build the solution.
Start the solution without debugging it.

Attached are some screenshots for reference-




