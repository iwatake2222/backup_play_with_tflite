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
#include "edgetpu.h"
#include "edgetpu_c.h"

/*** Macro ***/
/* Model parameters */
#define MODEL_FILENAME RESOURCE"/mobilenet_v2_1.0_224_quant_edgetpu.tflite"
#define USE_EDGETPU
// #define MODEL_FILENAME RESOURCE"/mobilenet_v2_1.0_224_quant.tflite"
// #define MODEL_FILENAME RESOURCE"/mobilenet_v2_1.0_224.tflite"
#define LABEL_NAME     RESOURCE"/imagenet_labels.txt"
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL 3
#define MODEL_IS_QUANTIZED
#ifdef MODEL_IS_QUANTIZED
#define MODEL_Q_SCALE 0.00390625
#define MODEL_Q_ZERO_POINT 0
typedef uint8_t MODEL_TENSOR_TYPE;
#else
typedef float MODEL_TENSOR_TYPE;
#endif

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
#ifdef USE_EDGETPU
	size_t num_devices;
	std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
	TFLITE_MINIMAL_CHECK(num_devices > 0);
	const auto& device = devices.get()[0];
	auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
	interpreter->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});
#endif
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
	// tflite::PrintInterpreterState(interpreter.get());

	/*** Process for each frame ***/
	/* Read input image data */
	cv::Mat inputImage = cv::imread(RESOURCE"/parrot.jpg");
	cv::imshow("test", inputImage); cv::waitKey(1);
	/* Pre-process */
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
	cv::resize(inputImage, inputImage, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
#ifdef MODEL_IS_QUANTIZED
	inputImage.convertTo(inputImage, CV_8UC3);
#else
	inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
#endif

	/* Set data to input tensor */
	auto* input = interpreter->typed_input_tensor<MODEL_TENSOR_TYPE>(0);
	memcpy(input, inputImage.reshape(0, 1).data, sizeof(MODEL_TENSOR_TYPE) * 1 * MODEL_WIDTH * MODEL_WIDTH * MODEL_CHANNEL);

	/* Run inference */
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	// tflite::PrintInterpreterState(interpreter.get());

	/* Retrieve the result */
	auto* scoresArray = interpreter->typed_output_tensor<MODEL_TENSOR_TYPE>(0);
	std::vector<MODEL_TENSOR_TYPE> scores(scoresArray, scoresArray + 1000);
	int maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
	auto maxScore = *std::max_element(scores.begin(), scores.end());
#ifdef MODEL_IS_QUANTIZED
	float score = (maxScore - MODEL_Q_ZERO_POINT) * MODEL_Q_SCALE;
#else
	float score = maxScore;
#endif
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
