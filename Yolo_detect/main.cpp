#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include<time.h>
#include <opencv2/core/utils/logger.hpp>
#include "yolov7.h"


int main()
{
	//cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

	clock_t startTime, endTime;

	 ///////Yolo

	//std::string path = "./obj/yolov7.onnx";
	//Yolo v7_model(path, 0.3, 0.5, 0.3, true);
	//std::string img_path = "./obj/bus.jpg";
	//cv::Mat image = cv::imread(img_path);

	////std::vector<std::string> herb = { "LEYE", "REYE"};
	////v7_model.setClass(herb);

	////std::vector<std::vector<float>> anchors = { {10,12},{22,30} };
	////v7_model.setAnchors(anchors);

	

	//int newh, neww, padw, padh;
	//cv::Mat new_image = v7_model.resize_image(image, newh, neww, padh, padw);
	//cv::imshow("image2", new_image);
	//std::cout << newh << neww << padh << padw << std::endl;
	//
	//double timeStart = (double)cv::getTickCount();

	//cv::namedWindow("image1", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
	//cv::imshow("image1", image);
	//startTime = clock();//��ʱ��ʼ	
	////v7_model.detect(image,true);
	//v7_model.detect(image, 1);
	//endTime = clock();//��ʱ��ʼ
	//std::cout << double(endTime - startTime) / CLOCKS_PER_SEC << std::endl;

	//cv::namedWindow("detect_image", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
	//cv::imshow("detect_image", image);
	//cv::waitKey(0);

	////////////video

	//cv::VideoCapture capture(0);

	//while (true)
	//{
	//	startTime = clock();//��ʱ��ʼ	
	//	cv::Mat frame;
	//	capture >> frame;
	//	v7_model.detect(frame, true);

	//	endTime = clock(); //��ʱ����
	//	double fps = 1.0 / (double(endTime - startTime) / CLOCKS_PER_SEC);
	//	cv::String s = std::to_string((int)fps);

	//	cv::putText(frame, s, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));

	//	cv::imshow("��ȡ��Ƶ", frame);
	//	cv::waitKey(1);

	//}

	///////////////////////////////yolov7

//	std::string pose_path = "./obj/yolov7-eye-8.onnx";					//��ģ��
//	Yolov7_Pose v7_pose_model(pose_path, 0.3, 0.5, 0.3, true);
//	
//	v7_pose_model.setClass({ "eye" });
//	v7_pose_model.setPoints(17);
//
//	/////////picture
//
//	cv::Mat pose_image = cv::imread("./obj/image0501.jpg");
//	cv::namedWindow("image2", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
//	cv::imshow("image2", pose_image);
//	startTime = clock();//��ʱ��ʼ	
//;
//
//	v7_pose_model.detect(pose_image, 1);
//	endTime = clock();//��ʱ����
//	std::cout << double(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
//
//	cv::namedWindow("detect_image2", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
//	cv::imshow("detect_image2", pose_image);
//	cv::waitKey(0);

	/////video

	//cv::VideoCapture capture("D:\\1LZH\\C++\\test_opencv\\test_opencv\\111.MP4");

	//while (true)
	//{
	//	
	//	cv::Mat frame;
	//	capture.read(frame);
	//	startTime = clock();//��ʱ��ʼ	

	//	v7_pose_model.detect(frame, true);

	//	endTime = clock(); //��ʱ����
	//	double fps = 1.0 / (double(endTime - startTime) / CLOCKS_PER_SEC);
	//	cv::String s = std::to_string((int)fps);

	//	cv::putText(frame, s, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));

	//	cv::namedWindow("��ȡ��Ƶ", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
	//	cv::imshow("��ȡ��Ƶ", frame);
	//	char key = cv::waitKey(1);
	//	if (key == 'q') break;
	//}


	///////////////////////////////yolov8

	std::string pose_path = "./obj/yolov8s-eye-17.onnx";					//��ģ��
	Yolov8_Pose v8_pose_model(pose_path, 0.3, 0.5, 0.3, true);

	v8_pose_model.setClass({ "eye" });
	v8_pose_model.setPoints(17);

	/////////picture

	cv::Mat pose_image = cv::imread("./obj/image0502.bmp");
	cv::namedWindow("image2", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
	cv::imshow("image2", pose_image);
	startTime = clock();//��ʱ��ʼ	
	;

	v8_pose_model.detect(pose_image, 1);
	endTime = clock();//��ʱ����
	std::cout << double(endTime - startTime) / CLOCKS_PER_SEC << std::endl;

	cv::namedWindow("detect_image2", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
	cv::imshow("detect_image2", pose_image);
	cv::waitKey(0);


	/////video

cv::VideoCapture capture("D:\\1LZH\\C++\\test_opencv\\test_opencv\\111.MP4");

while (true)
{
	
	cv::Mat frame;
	capture.read(frame);
	startTime = clock();//��ʱ��ʼ	

	v8_pose_model.detect(frame, true);

	endTime = clock(); //��ʱ����
	double fps = 1.0 / (double(endTime - startTime) / CLOCKS_PER_SEC);
	cv::String s = std::to_string((int)fps);

	cv::putText(frame, s, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));

	cv::namedWindow("��ȡ��Ƶ", cv::WINDOW_FREERATIO);  // ����Ӧ���ڴ��ڴ�С
	cv::imshow("��ȡ��Ƶ", frame);
	char key = cv::waitKey(1);
	if (key == 'q') break;
}



	return 0;
}
