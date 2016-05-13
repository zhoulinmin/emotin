//#include "stdafx.h"
#include "Light_Prep.h"
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>


//#include <iosfwd>
#include <memory>
#include <string>

#include <vector>
#include <iostream>
#include <fstream>
#include<opencv/cv.h>
#include <opencv/highgui.h>
//#include <windows.h>
#include <afx.h>  
#include <afxdlgs.h> 
#include <windows.h>
#include <atlconv.h>
#include<conio.h>
#include <utility>

#include <dlib-18.15/dlib/image_processing/frontal_face_detector.h>
#include <dlib-18.15/dlib/image_processing/render_face_detections.h>
#include <dlib-18.15/dlib/image_processing.h>
#include <dlib-18.15/dlib/gui_widgets.h>
#include <dlib-18.15/dlib/image_io.h>
#include <dlib-18.15/dlib/opencv.h>
//#include <dlib-18.15/dlib/opencv.h>
//#include <facedetect-dll.h>

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)
using namespace dlib;
using std::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;



/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
void faceAlign(Mat, Rect, shape_predictor&, Mat&);
Point2d getPointAffinedPos(const Point2d &, const Point2d, double);

void string_split(const string &str, char de, std::vector<string> & vs)
{
	size_t start = 0;
	while (true)
	{
		size_t pos = str.find_first_of(de, start);
		if (pos == string::npos)
		{
			vs.push_back(str.substr(start));
			break;
		}
		else{
			vs.push_back(str.substr(start, pos - start));
			start = pos + 1;
		}
	}
}

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 2);

private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	caffe::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	//Caffe::set_phase(Caffe::TEST);
	net_.reset(new Net<float>(model_file,TEST));
	//Net<float> caffe_net(model_file, caffe::TEST);
	LOG(INFO) << "setup all ready";
	net_->CopyTrainedLayersFrom(trained_file);


	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	//net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {

	::google::InitGoogleLogging(argv[0]);



	string model_file = "deploy_gender.prototxt";
	string trained_file = "caffenet_train_iter_5000.caffemodel";
	string mean_file ="imagenet_mean.binaryproto";
	string label_file = "genderlabel.txt";

	CascadeClassifier casc;
	if (!casc.load("haarcascade_frontalface_alt.xml"))//从指定的文件目录中加载级联分类器
	{
		printf("ERROR: Could not load classifier cascade");
		return 0;
	}

	char landmark_dat[256] = "E:\\caffe-windows-master\\buildVS2013\\classifer\\shape_predictor_68_face_landmarks.dat";
	shape_predictor sp;
	deserialize(landmark_dat) >> sp;

	frontal_face_detector  detector = get_frontal_face_detector();

	FLAGS_alsologtostderr = 1;
	Classifier classifier(model_file, trained_file, mean_file, label_file);
	VideoCapture c;
	if (!c.open(0)) {
		printf("Failed to open camera ");
	}
	Mat img;
	if (!c.isOpened())
		return -1;
	string result;
	while (c.isOpened())
	{
		c >> img;
	
		bool flag = false;
	
		int * pResults = NULL;
	
		std::vector<Rect> rects;
		rects.clear();



		//高斯平滑
		//cv::Mat dstimg;
		//dstimg = img.clone();
		//GaussianBlur(img, dstimg, Size(3, 3), 0, 0);
		//dstimg.copyTo(img);

		//光照调整
	/*	LightPrep   light;
		light.InitFilterKernel();
		light.InitLight(cvSize(img.size().width, img.size().height));
		CvMat cvimg = img;
		light.RunLightPrep(cvimg);*/

		//光照调整
		//LightPrep   light;
		//std::vector<Mat>  splitRGB(img.channels());
		//split(img, splitRGB);
		//light.InitFilterKernel();
		//light.InitLight(cvSize(img.size().width,img.size().height));
		//for (int i = 0; i < img.channels(); i++)
		//{
		//	CvMat* cvimg = cvCreateMat(img.size().width, img.size().height, CV_8UC1);
		//	//cvInitMatHeader(cvimg, img.size().width, img.size().height, CV_8UC1);
		//	CvMat temp = splitRGB[i];
		//	cvCopy(&temp, cvimg);
		//    //CvMat cvimg =splitRGB[i];
		//	light.RunLightPrep(cvimg);
		//}
		//merge(splitRGB, img);

		////直方图均衡
		//std::vector<Mat>  splitRGB(img.channels());
		//split(img, splitRGB);
		//for (int i = 0; i <img.channels(); i++)
		//	equalizeHist(splitRGB[i], splitRGB[i]);
		//merge(splitRGB, img);
		//double timebegin = (double)getTickCount();
		
		Mat extendBorder;
		Scalar value = Scalar(0, 0, 0);
		copyMakeBorder(img, extendBorder, 50, 50, 50, 50, BORDER_CONSTANT, value);


		casc.detectMultiScale(extendBorder, rects,
			1.1, 3, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			| CV_HAAR_SCALE_IMAGE,
			Size(20, 20));
		
	

		int x, y, w, h;
		int i = 0;
		Mat dstface;
		Rect rect;
		IplImage *  src = &IplImage(img);

	

		for (std::vector<Rect>::const_iterator r = rects.begin(); r != rects.end(); r++, i++)
		{
			if (0 == rects.size())
			{
				printf("could not find any face!\n");
				continue;
			}
			//Rect rect;
			if (r->x < 50 || r->y < 50 || r->x + rect.width>extendBorder.cols - 50 || r->y+ rect.height>extendBorder.rows - 50)
				continue;

			
				rect.x = r->x;
				rect.y = r->y;
				rect.width = r->width;
				rect.height = r->height;
		
			if (rect.x - rect.width / 2 < 0 || rect.y - rect.height / 2 < 0 || rect.x + rect.width *3 / 2 > extendBorder.cols || rect.y + rect.height *3 / 2 > extendBorder.rows)
				continue;			
			faceAlign(extendBorder, rect, sp, dstface);

			//	time_t timebegin = time(NULL);
			//	std::cout << timebegin << std::endl;
			std::vector<Prediction> predictions = classifier.Classify(dstface);
			/*double  timebegin = (double)getTickCount();
			
			double  timeend = (double)getTickCount();
			double timeused = (double)(timeend - timebegin) / getTickFrequency()*1000;
			std::cout << "used time - " << timeused << std::endl;*/
			// 	double timeused = (double)(timeend - timebegin)/1000;
			//	time_t  timeend = time(NULL);
			//	std::cout << timeend << std::endl;
			//	double timeused =difftime(timeend,timebegin);
	

			/* Print the top N predictions. */

			//for (size_t m = 0; m < predictions.size(); ++m) {
			//	Prediction p = predictions[m];

			//	std::cout << std::fixed << p.second << " - \""
			//		<< p.first << "\"" << std::endl;
			//}
			dstface.release();

			rect.x = rect.x - 50;
			rect.y = rect.y - 50;
			
			//	IplImage *src=cvLoadImage(file.c_str(),CV_LOAD_IMAGE_COLOR);
			Prediction   gender = predictions[0];
			string  showMsg = gender.first;

			CvScalar  color;
			int n = atoi(showMsg.c_str());
			
			switch (n)
			{
			case 0:
			color = Scalar(0, 255, 0);
			break;
			case 1:
			color = Scalar(255, 0, 0);
			break;
			case 2:
				color = Scalar(0, 0, 255);
				break;
			case 3:
				color = Scalar(255, 255, 255);
				break;
			case 4:
				 color = Scalar(0, 0, 0);
				break;
			case 5:
				color = Scalar(255, 0, 255);
				break;
				}
	/*		if (showMsg == "0")
			color =Scalar(0,255,0); 
			else 
			color =Scalar(255,0,0); */

			if (rect.x>0&&rect.y>0)
			cvRectangle(src, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height), color, 2, 8, 0);
	 
		}
	

			cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
			cvShowImage("image", src);

			int a = cvWaitKey(1);
			if (27==a)
		      break;
		}
	img.release();
	return 0;
}

void faceAlign(Mat image, Rect rect, shape_predictor& sp, Mat& dstimg)
{
	double ec_mc_y = 48;/*两个眼睛中间到两个嘴巴中间的距离*/
	double ec_y = 48;/*两个眼睛中间纵坐标*/
	Mat::zeros(rect.width * 2, rect.height * 2, CV_8U);
	Rect faceRect(rect.x - rect.width / 2, rect.y - rect.height / 2, rect.width * 2, rect.height * 2);
	Mat faceEx = image(faceRect);

	Mat face_gray;
	if (faceEx.channels() >= 3)
		cv::cvtColor(faceEx, face_gray, CV_BGR2GRAY);
	else
		face_gray = faceEx.clone();
	dlib::rectangle	rec(rect.width / 2, rect.height / 2, rect.width / 2 + rect.width, rect.height / 2 + rect.height);

	//dlib::rectangle rec(rect.x, rect.y, rect.width, rect.height);

	dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(face_gray), rec);
	//dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(image), rec);
	std::vector<Point2d> pts2d;
	for (size_t k = 0; k < shape.num_parts(); k++)
	{
		Point2d p(shape.part(k).x(), shape.part(k).y());
		pts2d.push_back(p);
	}

	//Mat inclone3 = inclone2.clone();
	//for (size_t i = 0; i<pts2d.size(); i++)
	//circle(faceEx, pts2d[i], 3, Scalar(255, 0, 255));
	Point2d eye_l = (pts2d[37] + pts2d[38] + pts2d[40] + pts2d[41]) * 0.25; // left eye center
	Point2d eye_r = (pts2d[43] + pts2d[44] + pts2d[46] + pts2d[47]) * 0.25; // right eye center
	Point2d lmouse = pts2d[48];
	Point2d rmouse = pts2d[54];

	Point2d ec, mc;
	ec.x = (eye_l.x + eye_r.x) / 2;
	ec.y = (eye_l.y + eye_r.y) / 2;
	mc.x = (lmouse.x + rmouse.x) / 2;
	mc.y = (lmouse.y + rmouse.y) / 2;

	//circle(faceEx, lmouse, 3, Scalar(255, 0, 255));
	//circle(faceEx, rmouse, 3, Scalar(255, 0, 255));

	double ang_tan = (eye_r.y - eye_l.y) / (eye_r.x - eye_l.x);
	double ang = atan(ang_tan) * 180 / CV_PI;

	Mat warppedFace;
	Point2f center(face_gray.cols / 2, face_gray.rows / 2);
	Mat rot = getRotationMatrix2D(center, ang, 1);
	warpAffine(face_gray, warppedFace, rot, Size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));

	Point2d ec_t = getPointAffinedPos(ec, center, ang * CV_PI / 180);
	//circle(faceEx, leye_t, 3, Scalar(255, 255, 255));
	Point2d mc_t = getPointAffinedPos(mc, center, ang * CV_PI / 180);
	//circle(faceEx, reye_t, 3, Scalar(255, 255, 255));

	double scale = ec_mc_y / sqrt((mc_t.x - ec_t.x)*(mc_t.x - ec_t.x) + (mc_t.y - ec_t.y)*(mc_t.y - ec_t.y));// 44.0 / (reye_t.x - leye_t.x);
	Mat resizedface;
	resize(warppedFace, resizedface, Size(warppedFace.rows*scale, warppedFace.cols*scale), 0, 0, CV_INTER_LINEAR);

	ec_t.x *= scale;
	ec_t.y *= scale;
	double crop_y = ec_t.y - ec_y;
	double crop_x = (ec_t.x - 144 / 2);
	Mat extendBorder;
	Scalar value = Scalar(0, 0, 0);
	//copyMakeBorder(resizedface, extendBorder, 120, 120, 120, 120, BORDER_CONSTANT, value);
	//if (crop_x + 159 > resizedface.rows || crop_y + 149> resizedface.cols||crop_x<15||crop_y<25)
	//	return;
	//else
	//{
		//Mat imgcropped = extendBorder(Rect(crop_x+120, crop_y+120, 128, 128));
		dstimg = resizedface(Rect(crop_x, crop_y, 144, 144));
	//}
	//return imgcropped;
}

Point2d getPointAffinedPos(const Point2d &src, const Point2d center, double angle)
{
	Point2d dst;
	int x = src.x - center.x;
	int y = src.y - center.y;

	dst.x = cvRound(x * cos(angle) + y * sin(angle) + center.x);
	dst.y = cvRound(-x * sin(angle) + y * cos(angle) + center.y);
	return dst;
}