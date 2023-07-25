#include "yolov7.h"
#include <time.h>

Yolo::Yolo(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda = false)		
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	this->objThreshold = objThreshold;

	this->net = cv::dnn::readNetFromONNX(modelpath);

	this->ratioh = 0;
	this->ratiow = 0;
	this->padh = 0;
	this->padw = 0;

	if (isCuda) 
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
		std::cout << "cuda" << std::endl;
	}
	else 
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}


	std::vector<std::string> cla = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
						"train", "truck", "boat", "traffic light", "fire hydrant",
						"stop sign", "parking meter", "bench", "bird", "cat", "dog",
						"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
						"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
						"skis", "snowboard", "sports ball", "kite", "baseball bat",
						"baseball glove", "skateboard", "surfboard", "tennis racket",
						"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
						"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
						"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
						"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
						"remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
						"sink", "refrigerator", "book", "clock", "vase", "scissors",
						"teddy bear", "hair drier", "toothbrush" };

	this->setClass(cla);

	std::vector<std::vector<float>> anchors = { {12.0, 16.0, 19.0, 36.0, 40.0, 28.0}, {36.0, 75.0, 76.0, 55.0, 72.0, 146.0}, {142.0, 110.0, 192.0, 243, 459.0, 401.0} };
	this->setAnchors(anchors);

	this->inputh = 640;
	this->inputw = 640;

}


void Yolo::setClass(const std::vector<std::string> &s)
{
	this->classes = s;
	this->num_classes = classes.size();
}


void Yolo::setAnchors(std::vector<std::vector<float>> anchors)
{
	this->Anchors.clear();
	std::vector<float> temp;
	for (int i = 0; i < anchors.size(); i++)
	{
		for (int j = 0; j < anchors[i].size(); j++)
		{
			temp.push_back(anchors[i][j]);
		}
		this->Anchors.push_back(temp);
		temp.clear();
	}

}


cv::Mat Yolo::resize_image(cv::Mat &input, int& newh, int& neww, int& padh, int& padw)
{
	int srch = input.rows;
	int srcw = input.cols;
	cv::Mat dstimg;
	if (srch == srcw)
	{
		cv::resize(input, dstimg, cv::Size(this->inputh, this->inputw), cv::INTER_AREA);
		neww = this->inputw;
		newh = this->inputh;
		padh = 0;
		padw = 0;
	}
	else
	{
		float ratio = (float)srch / srcw;
		if (ratio > 1)
		{
			newh = this->inputh;
			neww = int(this->inputw / ratio);
			padw = int((this->inputw - neww) / 2);
			padh = 0;
			cv::resize(input, dstimg, cv::Size(neww, newh), cv::INTER_AREA);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, padw, this->inputw - neww - padw,
				cv::BORDER_CONSTANT, 114);

		}
		else
		{
			newh = int(this->inputh * ratio);
			neww = this->inputw;
			padh = int((this->inputh - newh) / 2);
			padw = 0;
			cv::resize(input, dstimg, cv::Size(neww, newh), cv::INTER_AREA);
			cv::copyMakeBorder(dstimg, dstimg, padh, this->inputh - newh - padh, 0, 0,
				cv::BORDER_CONSTANT, 114);
		}
	}

	return dstimg;

}


void Yolo::detect(cv::Mat &img, bool keepRatio)
{
	//int neww = 0;
	//int newh = 0;
	cv::Mat new_image;
	if (keepRatio)
	{
		new_image = this->resize_image(img, this->newh, this->neww, this->padh, this->padw);
	}
	else
	{
		cv::resize(img, new_image, cv::Size(this->inputw, this->inputh));
		newh = this->inputh;
		neww = this->inputw;
		this->padw = 0;
		this->padh = 0;
	}

	this->ratioh = (float)img.rows / newh, this->ratiow = (float)img.cols / neww;

	this->img_col = img.cols, this->img_row = img.rows;

	cv::Mat blob = cv::dnn::blobFromImage(new_image, 1 / 255.0, cv::Size(this->inputw, this->inputh),
		cv::Scalar(0, 0, 0), true, false);


	int size4[4] = { 1, 1, 640, 640 };
	cv::Mat blob_one(4, size4, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < 640; i++)
		for (int j = 0; j < 640; j++)
			blob_one.ptr<uchar>(0, 0, i)[j] = blob.ptr<uchar>(0, 0, i)[j];

	this->net.setInput(blob);
	std::vector<cv::Mat> pred;


	net.enableWinograd(false);

	clock_t time0, time1, time2;

	time0 = clock();
	
	//for(int i = 0; i < this->net.getUnconnectedOutLayersNames().size(); i++)
	//	std::cout << this->net.getUnconnectedOutLayersNames()[i] << std::endl;

	this->net.forward(pred, this->net.getUnconnectedOutLayersNames());

	time1 = clock();

	this->postprocess(pred, img);

	time2 = clock();

	std::cout << "forward time = " << double(time1 - time0) / CLOCKS_PER_SEC << std::endl;
	std::cout << "postprocess time = " << double(time2 - time1) / CLOCKS_PER_SEC << std::endl;

	//cv::waitKey(0);
}


void Yolo::postprocess(std::vector<cv::Mat> &pred, cv::Mat& img)
{
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;

	for (int i = 0; i < pred.size(); i++)	// 输出结果的层数	1 3 x x nc+5
	{


		int h = pred[i].size[2];
		int w = pred[i].size[3];
		int num_proposal = h * w * pred[i].size[1];
		int out_dim = pred[i].size[4];
		pred[i] = pred[i].reshape(0, num_proposal);				//变成h*w*3 nc+5维度的矩阵
		float* pdata = (float*)pred[i].data;					//定义浮点型指针
		float stride = float(this->inputw / w);
		for (int x = 0; x < num_proposal; x++)
			for (int y = 0; y < out_dim; y++)
				pred[i].at<float>(x, y) = 1 / (1 + exp(-pred[i].at<float>(x, y)));

		for (int j = 0; j < num_proposal; j++)
		{
			int index = j * out_dim;

			float objconf = pdata[index + 4];
			if (objconf > this->objThreshold)
			{
				cv::Mat scores(1, this->num_classes, CV_32FC1, pdata + index + 5);
				cv::Point classIdPoint;
				double max_class_score;
				cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);

				if (max_class_score > this->confThreshold)
				{
					int anchor_num = j / (h * w);
					int idx = 0;

					for (int k = 0; k < 4; k++)
						if (abs(stride - this->Stride[k]) < 0.01)
							idx = k;
					float anchor_w = this->Anchors[idx][anchor_num * 2];
					float anchor_h = this->Anchors[idx][anchor_num * 2 + 1];

					const int class_idx = classIdPoint.x;

					float cx = (pdata[index] * 2.f - 0.5f + (j % h)) * stride;
					float cy = (pdata[index + 1] * 2.f - 0.5f + (j % (h * w)) / h) * stride;
					float boxw = powf(pdata[index + 2] * 2.f, 2.f) * anchor_w;
					float boxh = powf(pdata[index + 3] * 2.f, 2.f) * anchor_h;

					int left = int((cx - (float)padw - 0.5 * boxw) * ratiow);
					int top = int((cy - (float)padh - 0.5 * boxh) * ratioh);

					confidences.push_back((float)max_class_score);
					boxes.push_back(cv::Rect(left, top, (int)(boxw * ratiow), (int)(boxh * ratioh)));
					classIds.push_back(class_idx);

				}


			}





		}

	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, img, classIds[idx]);
	}
}


void Yolo::drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid)   // Draw the predicted bounding box
{
	using namespace std;
	using namespace cv;
	using namespace dnn;
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 4);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->classes[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
}





Yolov7_Pose::Yolov7_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda): Yolo(modelpath, confThreshold, nmsThreshold, objThreshold, isCuda)
{
	this->num_points = 17;
	this->setAnchors({{19, 27, 44, 40, 38, 94},
	{ 96, 68, 86, 152, 180, 137 },
	{ 140, 301, 303, 264, 238, 542 },
	{ 436, 615, 739, 380, 925, 792 },});
	this->setClass({ "person" });
}


Yolov7_Pose::Yolov7_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda, int num_points) : Yolo(modelpath, confThreshold, nmsThreshold, objThreshold, isCuda)
{
	this->num_points = num_points;
	this->setAnchors({ {19, 27, 44, 40, 38, 94},
	{ 96, 68, 86, 152, 180, 137 },
	{ 140, 301, 303, 264, 238, 542 },
	{ 436, 615, 739, 380, 925, 792 }, });
	this->setClass({ "person" });
}


void Yolov7_Pose::postprocess(std::vector<cv::Mat>& pred, cv::Mat& img)
{
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<std::vector<cv::Point>> points;


	for (int i = 0; i < pred.size(); i++)	// 输出结果的层数	1 3 x x nc+5+np*3
	{


		int h = pred[i].size[2];
		int w = pred[i].size[3];
		int num_proposal = h * w * pred[i].size[1];
		int out_dim = pred[i].size[4];
		pred[i] = pred[i].reshape(0, num_proposal);				//变成h*w*3 nc+5维度的矩阵
		float* pdata = (float*)pred[i].data;					//定义浮点型指针
		float stride = float(this->inputw / w);
		for (int x = 0; x < num_proposal; x++)
			for (int y = 0; y < out_dim; y++)
			{
				if (y <= 4 + this->num_classes)
					pred[i].at<float>(x, y) = 1 / (1 + exp(-pred[i].at<float>(x, y)));
				else if ((y - 4 - this->num_classes) % 3 == 0)
					pred[i].at<float>(x, y) = 1 / (1 + exp(-pred[i].at<float>(x, y)));
			}

		for (int j = 0; j < num_proposal; j++)
		{
			int index = j * out_dim;

			float objconf = pdata[index + 4];
			if (objconf > this->objThreshold)
			{
				cv::Mat scores(1, this->num_classes, CV_32FC1, pdata + index + 5);
				cv::Point classIdPoint;
				double max_class_score;
				cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);

				if (max_class_score > this->confThreshold)
				{
					int anchor_num = j / (h * w);
					int idx = 0;

					for (int k = 0; k < 4; k++)
						if (abs(stride - this->Stride[k]) < 0.01)
							idx = k;
					float anchor_w = this->Anchors[idx][anchor_num * 2];
					float anchor_h = this->Anchors[idx][anchor_num * 2 + 1];

					const int class_idx = classIdPoint.x;

					float cx = (pdata[index] * 2.f - 0.5f + (j % h)) * stride;
					float cy = (pdata[index + 1] * 2.f - 0.5f + (j % (h * w)) / h) * stride;
					float boxw = powf(pdata[index + 2] * 2.f, 2.f) * anchor_w;
					float boxh = powf(pdata[index + 3] * 2.f, 2.f) * anchor_h;

					int left = int((cx - (float)padw - 0.5 * boxw) * ratiow);
					int top = int((cy - (float)padh - 0.5 * boxh) * ratioh);

					confidences.push_back((float)max_class_score);
					boxes.push_back(cv::Rect(left, top, (int)(boxw * ratiow), (int)(boxh * ratioh)));
					classIds.push_back(class_idx);
					
					float new_ratioh = this->newh / boxh;
					float new_ratiow = this->neww / boxw;

					std::vector<cv::Point> temp;
					for (int np = 0; np < this->num_points; np++)
					{
						float conf_p = pdata[index + 5 + this->num_classes + np * 3 + 2];
						if (conf_p > 0.5)
						{
							float px = (pdata[index + 5 + this->num_classes + np * 3] * 2.f - 0.5f + (j % h)) * stride;
							float py = (pdata[index + 5 + this->num_classes + np * 3 + 1] * 2.f - 0.5f + (j % (h * w)) / h) * stride;
							int tempx = int((px - (float)padw) * this->ratiow);
							int tempy = int((py - (float)padh) * this->ratioh);
							//cv::Point p(cv::Point(tempx, tempy));
							temp.push_back(cv::Point(tempx, tempy));
						}
					}
					points.push_back(temp);

				}
			}
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, img, classIds[idx]);
		this->drawPoints(points[idx], img);
	}
}


void Yolov7_Pose::drawPoints(std::vector<cv::Point> p, cv::Mat& frame)
{
	for (int i = 0; i < p.size(); i++)
	{
		cv::circle(frame, p[i], 2, cv::Scalar(255,0,0), 2);
	}
}

void Yolov7_Pose::setPoints(int num_points)
{
	this->num_points = num_points;
}


Yolov8_Pose::Yolov8_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda) : Yolo(modelpath, confThreshold, nmsThreshold, objThreshold, isCuda)
{
	this->num_points = 17;
	this->setClass({ "person" });
}


Yolov8_Pose::Yolov8_Pose(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda, int num_points) : Yolo(modelpath, confThreshold, nmsThreshold, objThreshold, isCuda)
{
	this->num_points = num_points;
	this->setClass({ "person" });
}


void Yolov8_Pose::setPoints(int num_points)
{
	this->num_points = num_points;
}


void Yolov8_Pose::postprocess(std::vector<cv::Mat>& pred, cv::Mat& img)
{
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<std::vector<cv::Point>> points;

												// 输出结果的层数	1 xywh+ncconf+np*3 8400


		


	int num_proposal = pred[0].size[2];
	int out_dim = pred[0].size[1];

	pred[0] = pred[0].reshape(0, out_dim).t();				//变成h*w*3 nc+5维度的矩阵
	float* pdata = (float*)pred[0].data;					//定义浮点型指针


	for (int j = 0; j < num_proposal; j++)
	{
		int index = j * out_dim;

		cv::Mat scores(1, this->num_classes, CV_32FC1, pdata + index + 4);
		cv::Point classIdPoint;
		double max_class_score;
		cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);

		if (max_class_score > this->confThreshold)
		{
			const int class_idx = classIdPoint.x;

			float cx = pdata[index];
			float cy = pdata[index + 1];
			float boxw = pdata[index + 2];
			float boxh = pdata[index + 3];

			int left = int((cx - (float)padw - 0.5 * boxw) * ratiow);
			int top = int((cy - (float)padh - 0.5 * boxh) * ratioh);

			confidences.push_back((float)max_class_score);
			boxes.push_back(cv::Rect(left, top, (int)(boxw * ratiow), (int)(boxh * ratioh)));
			classIds.push_back(class_idx);

			float new_ratioh = this->newh / boxh;
			float new_ratiow = this->neww / boxw;

			std::vector<cv::Point> temp;
			for (int np = 0; np < this->num_points; np++)
			{
				float conf_p = pdata[index + 4 + this->num_classes + np * 3 + 2];
				if (conf_p > 0.5)
				{
					float px = pdata[index + 4 + this->num_classes + np * 3];
					float py = pdata[index + 4 + this->num_classes + np * 3 + 1];
					int tempx = int((px - (float)padw) * this->ratiow);
					int tempy = int((py - (float)padh) * this->ratioh);
					//cv::Point p(cv::Point(tempx, tempy));
					temp.push_back(cv::Point(tempx, tempy));
				}
			}
			points.push_back(temp);

		}


	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, img, classIds[idx]);
		this->drawPoints(points[idx], img);
	}
}


void Yolov8_Pose::drawPoints(std::vector<cv::Point> p, cv::Mat& frame)
{
	for (int i = 0; i < p.size(); i++)
	{
		cv::circle(frame, p[i], 2, cv::Scalar(255, 0, 0), 2);
	}
}