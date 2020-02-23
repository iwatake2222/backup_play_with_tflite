/*** Include ***/
/* for general */
#include <stdint.h>
#include <stdio.h>
#include <fstream> 
#include <vector>
#include <string>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for Tensorflow Lite */
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

/* for Edge TPU */
#ifdef USE_EDGETPU
#include "edgetpu.h"
#include "edgetpu_c.h"
#endif

/*** Macro ***/
/* Model parameters */
#define USE_EDGETPU_DELEGATE
#define MODEL_FILENAME RESOURCE"/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
//#define MODEL_FILENAME RESOURCE"/mobilenet_v2_1.0_224_quant.tflite"
// #define MODEL_FILENAME RESOURCE"/mobilenet_v2_1.0_224.tflite"
#define LABEL_NAME     RESOURCE"/imagenet_labels.txt"


/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 100

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/*** Function ***/
static void readLabel(const char* filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		printf("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}

int main()
{
	/*** Initialize ***/
	/* read label */
	std::vector<std::string> labels;
	readLabel(LABEL_NAME, labels);

	/* Create interpreter */
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
	TFLITE_MINIMAL_CHECK(model != nullptr);
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<tflite::Interpreter> interpreter;
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);
#ifdef USE_EDGETPU_DELEGATE
	size_t num_devices;
	std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
	TFLITE_MINIMAL_CHECK(num_devices > 0);
	const auto& device = devices.get()[0];
	auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
	interpreter->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});
#endif
	interpreter->SetNumThreads(4);
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
	// tflite::PrintInterpreterState(interpreter.get());

	/* Get model information */
	auto* inputTensor = interpreter->input_tensor(0);
	auto* outputTensor = interpreter->output_tensor(0);
	int   modelInputWidth = 0;
	int   modelInputHeight = 0;
	int   modelInputChannel = 0;
	int   modelOutputNum = 0;
	bool isQuantizedModel = false;
	float modelOutputQscale = 1.0;
	int   modelOutputZeroPoint = 0;

	for (int i = 0; i < inputTensor->dims->size; i++) printf("inputTensor->dims->size[%d]: %d\n", i, inputTensor->dims->data[i]);
	for (int i = 0; i < outputTensor->dims->size; i++) printf("outputTensor->dims->size[%d]: %d\n", i, outputTensor->dims->data[i]);
	modelInputHeight = inputTensor->dims->data[1];
	modelInputWidth = inputTensor->dims->data[2];
	modelInputChannel = inputTensor->dims->data[3];
	modelOutputNum = outputTensor->dims->data[1];
	
	if (outputTensor->type == kTfLiteUInt8) {
		isQuantizedModel = true;
		modelOutputQscale = outputTensor->params.scale;
		modelOutputZeroPoint = outputTensor->params.zero_point;
		printf("model is quantized. qcale = %f, zer_point = %d\n", modelOutputQscale, modelOutputZeroPoint);
	}

	/*** Process for each frame ***/
	/* Read input image data */
	cv::Mat inputImage = cv::imread(RESOURCE"/parrot.jpg");
	cv::imshow("test", inputImage); cv::waitKey(1);

	/* Pre-process */
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
	cv::resize(inputImage, inputImage, cv::Size(modelInputWidth, modelInputHeight));
	if (isQuantizedModel) {
		inputImage.convertTo(inputImage, CV_8UC3);
	} else {
		inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
	}

	/* Set data to input tensor */
	if (isQuantizedModel) {
		memcpy(inputTensor->data.int8, inputImage.reshape(0, 1).data, sizeof(int8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	} else {
		memcpy(inputTensor->data.f, inputImage.reshape(0, 1).data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	}

	/* Run inference */
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	// tflite::PrintInterpreterState(interpreter.get());

	/* Retrieve the result */
	int maxIndex = 0;
	float score  = -1;
	if (isQuantizedModel) {
		auto* scoresArray = outputTensor->data.uint8;
		std::vector<uint8_t> scores(scoresArray, scoresArray + modelOutputNum);
		maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
		auto maxScore = *std::max_element(scores.begin(), scores.end());
		score = (maxScore - modelOutputZeroPoint) * modelOutputQscale;
	} else {
		auto* scoresArray = outputTensor->data.f;
		std::vector<float> scores(scoresArray, scoresArray + modelOutputNum);
		maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
		auto maxScore = *std::max_element(scores.begin(), scores.end());
		score = maxScore;
	}
	printf("%s (%.3f)\n", labels[maxIndex].c_str(), score);


	/*** (Optional) Measure inference time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		interpreter->Invoke();
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);
	
	cv::waitKey(-1);
	return 0;
}
