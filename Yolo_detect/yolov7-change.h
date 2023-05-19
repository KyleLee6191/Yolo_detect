#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <opencv2/dnn/dnn.hpp>      // 深度学习模块
#include <opencv2/imgproc.hpp>	// 图像处理模块
#include <opencv2/highgui.hpp>  // 高层GUI图形用户界面
#include <cmath>

class Yolov7
{
public:
	Yolov7(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda);
	void detect(cv::Mat &img, bool keepRatio);
	void setClass(std::vector<std::string> &s);
	cv::Mat resize_image(cv::Mat &input, int& newh, int& neww, int& padh, int& padw);
	void setAnchor(std::vector<std::vector<float>> anchors);
	virtual void postprocess(float ratioh, float ratiow, int padh, int padw);

private:
	void drawPred(int xl, int xr, int yl, int yr,float conf);
	void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid);

	std::vector<std::string> classes;
	int num_classes;
	int inputh;
	int inputw;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	cv::dnn::Net net;
	std::vector<std::vector<float>> Anchors = { {12.0, 16.0, 19.0, 36.0, 40.0, 28.0}, {36.0, 75.0, 76.0, 55.0, 72.0, 146.0}, {142.0, 110.0, 192.0, 243, 459.0, 401.0} };
	//float Anchors[3][6] = { {12.0, 16.0, 19.0, 36.0, 40.0, 28.0}, {36.0, 75.0, 76.0, 55.0, 72.0, 146.0}, {142.0, 110.0, 192.0, 243, 459.0, 401.0}};
	//float Anchors[3][6] = { {10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326} };

	std::vector<cv::Mat> pred;
	float Stride[4] = { 8.0, 16.0, 32.0, 64.0};


};


