#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <opencv2/dnn/dnn.hpp>      // 深度学习模块
#include <opencv2/imgproc.hpp>	// 图像处理模块
#include <opencv2/highgui.hpp>  // 高层GUI图形用户界面
#include <cmath>

class Yolo
{
public:
	Yolo(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda);
	void detect(cv::Mat &img, bool keepRatio);
	void setClass(const std::vector<std::string> &s);
	cv::Mat resize_image(cv::Mat &input, int& newh, int& neww, int& padh, int& padw);
	void setAnchors(std::vector<std::vector<float>> anchors);

	virtual void postprocess(std::vector<cv::Mat> &pred, cv::Mat& img);

	int  padw, padh;
	float ratioh, ratiow;

	int newh, neww;
	int img_col, img_row;

protected:
	virtual void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid);

	std::vector<std::string> classes;
	int num_classes;	
	int inputh;
	int inputw;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	cv::dnn::Net net;
	std::vector<std::vector<float>> Anchors;
	//float Anchors[3][6] = { {12.0, 16.0, 19.0, 36.0, 40.0, 28.0}, {36.0, 75.0, 76.0, 55.0, 72.0, 146.0}, {142.0, 110.0, 192.0, 243, 459.0, 401.0}};
	//float Anchors[3][6] = { {10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326} };

	float Stride[4] = { 8.0, 16.0, 32.0, 64.0};


}; 


class Yolov7_Pose : public Yolo
{
public:
	//using Yolo::Yolo;
	Yolov7_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda);

	Yolov7_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda, int num_points);

	void setPoints(int num_points);

	void postprocess(std::vector<cv::Mat>& pred, cv::Mat& img) override;



protected:

	int num_points;

	void drawPoints(std::vector<cv::Point> p, cv::Mat& frame);

	//void postprocess(std::vector<cv::Mat>& pred, std::vector<float>& confidences,
	//	std::vector<cv::Rect>& boxes, std::vector<int>& classIds) override;

};


class Yolov8_Pose : public Yolo
{
public:
	//using Yolo::Yolo;
	Yolov8_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda);

	Yolov8_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda, int num_points);

	void setPoints(int num_points);

	void postprocess(std::vector<cv::Mat>& pred, cv::Mat& img) override;



protected:

	int num_points;

	void drawPoints(std::vector<cv::Point> p, cv::Mat& frame);

	//void postprocess(std::vector<cv::Mat>& pred, std::vector<float>& confidences,
	//	std::vector<cv::Rect>& boxes, std::vector<int>& classIds) override;

};
