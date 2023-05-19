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

	//cv::namedWindow("image1", cv::WINDOW_FREERATIO);  // 自适应调节窗口大小
	//cv::imshow("image1", image);
	//startTime = clock();//计时开始	
	////v7_model.detect(image,true);
	//v7_model.detect(image, 1);
	//endTime = clock();//计时开始
	//std::cout << double(endTime - startTime) / CLOCKS_PER_SEC << std::endl;

	//cv::namedWindow("detect_image", cv::WINDOW_FREERATIO);  // 自适应调节窗口大小
	//cv::imshow("detect_image", image);
	//cv::waitKey(0);

	////////////video

	//cv::VideoCapture capture(0);

	//while (true)
	//{
	//	startTime = clock();//计时开始	
	//	cv::Mat frame;
	//	capture >> frame;
	//	v7_model.detect(frame, true);

	//	endTime = clock(); //计时结束
	//	double fps = 1.0 / (double(endTime - startTime) / CLOCKS_PER_SEC);
	//	cv::String s = std::to_string((int)fps);

	//	cv::putText(frame, s, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));

	//	cv::imshow("读取视频", frame);
	//	cv::waitKey(1);

	//}

	///////////////////////////////

	std::string pose_path = "./obj/yolov7-w6-pose.onnx";					//读模型
	Yolov7_Pose v7_pose_model(pose_path, 0.3, 0.5, 0.3, true);
	
	//v7_pose_model.setClass({ "Leye", "Reye" });
	//v7_pose_model.setPoints(8);

	/////////picture

	cv::Mat pose_image = cv::imread("./obj/bus.jpg");
	cv::namedWindow("image2", cv::WINDOW_FREERATIO);  // 自适应调节窗口大小
	cv::imshow("image2", pose_image);
	startTime = clock();//计时开始	
;

	v7_pose_model.detect(pose_image, 1);
	endTime = clock();//计时结束
	std::cout << double(endTime - startTime) / CLOCKS_PER_SEC << std::endl;

	cv::namedWindow("detect_image2", cv::WINDOW_FREERATIO);  // 自适应调节窗口大小
	cv::imshow("detect_image2", pose_image);
	cv::waitKey(0);

	/////video

	cv::VideoCapture capture(0);

	while (true)
	{
		startTime = clock();//计时开始	
		cv::Mat frame;
		capture >> frame;
		v7_pose_model.detect(frame, true);

		endTime = clock(); //计时结束
		double fps = 1.0 / (double(endTime - startTime) / CLOCKS_PER_SEC);
		cv::String s = std::to_string((int)fps);

		cv::putText(frame, s, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));

		cv::namedWindow("读取视频", cv::WINDOW_FREERATIO);  // 自适应调节窗口大小
		cv::imshow("读取视频", frame);
		char key = cv::waitKey(1);
		if (key == 'q') break;
	}



	return 0;
}
