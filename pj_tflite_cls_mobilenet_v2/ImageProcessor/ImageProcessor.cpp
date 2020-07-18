/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>


/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "InferenceHelper.h"
#include "ImageProcessor.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

/* Setting */
static const float PIXEL_MEAN[3] = { 0.5f, 0.5f, 0.5f };
static const float PIXEL_STD[3] = { 0.25f,  0.25f, 0.25f };

/*** Global variable ***/
static std::vector<std::string> s_labels;
static InferenceHelper *s_inferenceHelper;
static InferenceHelper::TENSOR_INFO s_inputTensor;
static InferenceHelper::TENSOR_INFO s_outputTensor;

/*** Function ***/
static void readLabel(const char* filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}


int ImageProcessor_initialize(const char *modelFilename, INPUT_PARAM *inputParam)
{
	s_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE);
	s_inferenceHelper->initialize(modelFilename, inputParam->numThreads);
	s_inferenceHelper->getTensorByName("input", &s_inputTensor);
	s_inferenceHelper->getTensorByName("MobilenetV2/Predictions/Reshape_1", &s_outputTensor);

	/* read label */
	readLabel(inputParam->labelFilename, s_labels);

	return 0;
}


int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	/*** PreProcess ***/
	cv::Mat inputImage;

	cv::resize(*mat, inputImage, cv::Size(s_inputTensor.width, s_inputTensor.height));
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
	if (s_inputTensor.type == InferenceHelper::TENSOR_TYPE_UINT8) {
		inputImage.convertTo(inputImage, CV_8UC3);
	} else {
		inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
		cv::subtract(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_MEAN)), inputImage);
		cv::divide(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_STD)), inputImage);
	}

	/* Set data to input tensor */
#if 1
	s_inferenceHelper->setBufferToTensorByIndex(s_inputTensor.index, (char*)inputImage.data, (int)(inputImage.total() * inputImage.elemSize()));
#else
	if (s_inputTensor.type == InferenceHelper::TENSOR_TYPE_UINT8) {
		memcpy(s_inputTensor.data, inputImage.reshape(0, 1).data, sizeof(uint8_t) * 1 * s_inputTensor.width * s_inputTensor.height * s_inputTensor.channel);
	} else {
		memcpy(s_inputTensor.data, inputImage.reshape(0, 1).data, sizeof(float) * 1 * s_inputTensor.width * s_inputTensor.height * s_inputTensor.channel);
	}
#endif
	
	/*** Inference ***/
	s_inferenceHelper->inference();

	/*** PostProcess ***/
	/* Retrieve the result */
	std::vector<float> outputScoreList;
	outputScoreList.resize(s_outputTensor.width * s_outputTensor.height * s_outputTensor.channel);
	if (s_outputTensor.type == InferenceHelper::TENSOR_TYPE_UINT8) {
		const uint8_t* valUint8 = (uint8_t*)s_outputTensor.data;
		for (int i = 0; i < outputScoreList.size(); i++) {
			float valFloat = (valUint8[i] - s_outputTensor.quant.zeroPoint) * s_outputTensor.quant.scale;
			outputScoreList[i] = valFloat;
		}
	} else {
		const float* valFloat = (float*)s_outputTensor.data;
		for (int i = 0; i < outputScoreList.size(); i++) {
			outputScoreList[i] = valFloat[i];
		}
	}

	/* Find the max score */
	int maxIndex = (int)(std::max_element(outputScoreList.begin(), outputScoreList.end()) - outputScoreList.begin());
	auto maxScore = *std::max_element(outputScoreList.begin(), outputScoreList.end());
	PRINT("Result = %s (%d) (%.3f)\n", s_labels[maxIndex].c_str(), maxIndex, maxScore);

	/* Draw the result */
	std::string resultStr;
	resultStr = "Result:" + s_labels[maxIndex] + " (score = " + std::to_string(maxScore) + ")";
	cv::putText(*mat, resultStr, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 3);
	cv::putText(*mat, resultStr, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);

	/* Return the results */
	outputParam->classId = maxIndex;
	snprintf(outputParam->label, sizeof(outputParam->label), s_labels[maxIndex].c_str());
	outputParam->score = maxScore;
	
	return 0;
}


int ImageProcessor_finalize(void)
{
	s_inferenceHelper->finalize();
	delete s_inferenceHelper;
	return 0;
}
