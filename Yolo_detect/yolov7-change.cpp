#include "yolov7.h"
#include <time.h>

Yolov7::Yolov7(std::string modelpath, float confThreshold, float nmsThreshold, float objThreshold, bool isCuda = false)		//Cuda后续添加
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	this->objThreshold = objThreshold;

	this->net = cv::dnn::readNetFromONNX(modelpath);

	if (isCuda) 
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
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

	this->inputh = 640;
	this->inputw = 640;

}


void Yolov7::setClass(std::vector<std::string> &s)
{
	this->classes = s;
	this->num_classes = classes.size();
}


cv::Mat Yolov7::resize_image(cv::Mat &input, int& newh, int& neww, int& padh, int& padw)
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


void Yolov7::detect(cv::Mat& img, bool keepRatio)
{
	int newh, neww, padw, padh;
	if (keepRatio)
	{
		cv::Mat new_image = this->resize_image(img, newh, neww, padh, padw);
		padh = padw = 0;
	}
	else
	{
		cv::Mat new_image;
		cv::resize(img, new_image, cv::Size(this->inputw, this->inputh));
		newh = this->inputh;
		neww = this->inputw;
		padw = padh = 0;
	}

	float ratioh = (float)img.rows / this->inputh, ratiow = (float)img.cols / this->inputw;

	cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 255.0, cv::Size(this->inputw, this->inputh),
		cv::Scalar(0, 0, 0), true, false);

	this->net.setInput(blob);

	net.enableWinograd(false);

	this->net.forward(this->pred, this->net.getUnconnectedOutLayersNames());

}

void Yolov7::postprocess(float ratioh, float ratiow, int padh, int padw)
{
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;    
	std::vector<int> classIds;



	if (this->pred.size() == 3)
	{
		for (int i = 0; i < this->pred.size(); i++)	// 输出结果的层数	1 3 x x nc+5
		{


			int h = this->pred[i].size[2];
			int w = this->pred[i].size[3];
			int num_proposal = h * w * this->pred[i].size[1];
			int out_dim = this->pred[i].size[4];
			this->pred[i] = this->pred[i].reshape(0, num_proposal);				//变成h*w*3 nc+5维度的矩阵
			float* pdata = (float*)this->pred[i].data;					//定义浮点型指针
			float stride = float(this->inputw / w);
			for (int x = 0; x < num_proposal; x++)
				for (int y = 0; y < out_dim; y++)
					this->pred[i].at<float>(x, y) = 1 / (1 + exp(-this->pred[i].at<float>(x, y)));

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
						float anchor_w = this->Anchors[idx][anchor_num*2];
						float anchor_h = this->Anchors[idx][anchor_num*2+1];

						const int class_idx = classIdPoint.x;

						float cx = (pdata[index] * 2.f - 0.5f + (j % h)) * stride;
						float cy = (pdata[index + 1] * 2.f - 0.5f + (j % (h*w)) / h) * stride;
						float boxw = powf(pdata[index + 2] * 2.f, 2.f) * anchor_w;
						float boxh = powf(pdata[index + 3] * 2.f, 2.f) * anchor_h;

						
						int left = int((cx - padw - 0.5 * boxw) * ratiow);
						int top = int((cy - padh - 0.5 * boxh) * ratioh);
						
						confidences.push_back((float)max_class_score);
						boxes.push_back(cv::Rect(left, top, (int)(boxw * ratiow), (int)(boxh * ratioh)));  
						classIds.push_back(class_idx);  

					}


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

	//cv::waitKey(0);
}


void Yolov7::drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid)   // Draw the predicted bounding box
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


void Yolov7::setAnchor(std::vector<std::vector<float>> anchors)
{
	this->Anchors = anchors;
}