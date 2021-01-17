#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

//寻找火焰位置并标记
void fire(Mat& src, Mat& fireMat) {
	//对待测火焰规定阈值范围
	float scale = 1;
	double i_minH = 11;
	double i_maxH = 33;
	double i_minS = 120;
	double i_maxS = 180;
	double i_minV = 150;
	double i_maxV = 255;

	Mat hsvMat;
	Mat detectMat;
	
	
	Size ResImgSiz = Size(src.cols * scale, src.rows * scale);
	Mat dst = Mat(ResImgSiz, src.type());
	resize(src, dst, ResImgSiz, INTER_LINEAR);
	cvtColor(dst, hsvMat, COLOR_BGR2HSV);
	dst.copyTo(detectMat);
	cv::inRange(hsvMat, Scalar(i_minH, i_minS, i_minV), Scalar(i_maxH, i_maxS, i_maxV),detectMat);
	
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element1 = getStructuringElement(MORPH_RECT, Size(25, 25));
	erode(detectMat, detectMat, element);//腐蚀
	dilate(detectMat, detectMat, element1);//膨胀
	std::vector<std::vector<Point>> contours;
	findContours(detectMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	imshow("dec",detectMat);
	for (int i = 0; i < contours.size(); i++) {
		RotatedRect rbox = minAreaRect(contours[i]);

		float width = (float)rbox.size.width;
		float height = (float)rbox.size.height;
		float ratio = width / height;
		if (width > 10) {

			cv::Point2f vtx[4];
			rbox.points(vtx);
			for (int i = 0; i < 4; i++) {
				cv::line(src, vtx[i], vtx[i < 3 ? i + 1 : 0], cv::Scalar(0, 0, 255), 2, CV_AA);
			}
		}
	}
}
//通过背景差分大致得到水的轨迹
int water_trance(Mat& frame, Mat& bgMat, Mat& bny_subMat)
{
	Mat subMat;
	absdiff(frame, bgMat, subMat);  
	threshold(subMat, bny_subMat, 90, 255, CV_THRESH_BINARY);
	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			if (j < 150 || (i < 90 && j>320)) {
				bny_subMat.at<uchar>(i, j) = 0;
			}
		}
	}
	return 0;
}
/*构建矩阵A，用于最小二乘法拟合，来源于CSDN https://blog.csdn.net/qq_20797273/article/details/83930101 */
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();
	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}
	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}
	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}
/*用最小二乘法拟合曲线,来源于CSDN https://blog.csdn.net/qq_20797273/article/details/83930101 */
int fit_plot(Mat& bny_mat, Mat& frame)
{
	std::vector<cv::Point> points;

	for (int i = 35; i < bny_mat.rows; i++) {
		for (int j = 120; j < bny_mat.cols; j++) {
			if (bny_mat.at<uchar>(i, j) == 255 && (i < 120)) {
				points.push_back(cv::Point(j, i));
			}

		}
	}
	cv::Mat A;
	polynomial_curve_fit(points, 2, A);
	//std::cout << "A = " << A << std::endl;
	std::vector<cv::Point> points_fitted;
	for (int x = 175; x < 500; x++)
	{
			double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x + A.at<double>(2, 0) * std::pow(x, 2);
			points_fitted.push_back(cv::Point(x, y));
	}
	cv::polylines(frame, points_fitted, false, cv::Scalar(255, 255, 0), 2, 8, 0);
	return 0;
}

int main() {
	Mat frame;
	Mat bgMat;
	Mat bny_subMat;
	VideoCapture cap;
	int cnt = 0;
	cap.open("附件3.mp4");
	if (!cap.isOpened())
	{
		std::cout << "不能打开视频文件" << std::endl;
		return -1;
	}
	while (1)
	{
		cap >> frame;
		Mat frame1;
		frame.copyTo(frame1);
		//fire(frame1, frame1);
		
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		if (cnt == 0) {
			frame.copyTo(bgMat);
		}
		else {
			water_trance(frame, bgMat, bny_subMat);
			fire(frame1, frame1);
			fit_plot(bny_subMat, frame1);
			imshow("src", frame1);

			waitKey(30);
		}
		cnt++;
		if (cnt == 250) return 0;
	}
}
