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
#define MODEL_FILENAME RESOURCE"/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_edgetpu.tflite"
//#define MODEL_FILENAME RESOURCE"/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite"
#define LABEL_NAME     RESOURCE"/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.txt"

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 100

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

typedef struct {
	double x;
	double y;
	double w;
	double h;
	int classId;
	std::string classIdName;
	double score;
} BBox;

/*** Function ***/
static void displayModelInfo(const tflite::Interpreter* interpreter)
{
	const auto& inputIndices = interpreter->inputs();
	int inputNum = inputIndices.size();
	printf("Input num = %d\n", inputNum);
	for (int i = 0; i < inputNum; i++) {
		auto* tensor = interpreter->tensor(inputIndices[i]);
		std::vector<int> tensorSize;
		for (int j = 0; j < tensor->dims->size; j++) {
			printf("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
			tensorSize.push_back(tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			printf("    tensor[%d]->type: quantized\n", i);
			printf("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			printf("    tensor[%d]->type: not quantized\n", i);
		}
	}

	const auto& outputIndices = interpreter->outputs();
	int outputNum = outputIndices.size();
	printf("Output num = %d\n", outputNum);
	for (int i = 0; i < outputNum; i++) {
		auto* tensor = interpreter->tensor(outputIndices[i]);
		std::vector<int> tensorSize;
		for (int j = 0; j < tensor->dims->size; j++) {
			printf("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
			tensorSize.push_back(tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			printf("    tensor[%d]->type: quantized\n", i);
			printf("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			printf("    tensor[%d]->type: not quantized\n", i);
		}
	}
}


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

static void getBBox(std::vector<BBox> &bboxList, const float *outputBoxList, const float *outputClassList, const float *outputScoreList, const int outputNum, const double threshold, const int imageWidth = 0, const int imageHeight = 0)
{
	for (int i = 0; i < outputNum; i++) {
		int classId = outputClassList[i] + 1;
		float score = outputScoreList[i];
		if (score < threshold) continue;
		float y0 = outputBoxList[4 * i + 0];
		float x0 = outputBoxList[4 * i + 1];
		float y1 = outputBoxList[4 * i + 2];
		float x1 = outputBoxList[4 * i + 3];
		float w = x1 - x0 + 1;
		float h = y1 - y0 + 1;
		if (imageWidth != 0) {
			x0 *= imageWidth;
			x1 *= imageWidth;
			y0 *= imageHeight;
			y1 *= imageHeight;
		}
		//printf("%d[%.2f]: %.3f %.3f %.3f %.3f\n", classId, score, x0, y0, x1, y1);
		BBox bbox;
		bbox.x = x0;
		bbox.y = y0;
		bbox.w = x1 - x0;
		bbox.h = y1 - y0;
		bbox.classId = classId;
		bbox.score = score;
		bboxList.push_back(bbox);
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
	interpreter->SetNumThreads(4);
#ifdef USE_EDGETPU_DELEGATE
	size_t num_devices;
	std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
	TFLITE_MINIMAL_CHECK(num_devices > 0);
	const auto& device = devices.get()[0];
	auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
	interpreter->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});
#endif
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);


	/* Get model information */
	displayModelInfo(interpreter.get());
	const TfLiteTensor* inputTensor = interpreter->input_tensor(0);
	const int modelInputHeight = inputTensor->dims->data[1];
	const int modelInputWidth = inputTensor->dims->data[2];
	const int modelInputChannel = inputTensor->dims->data[3];


	/*** Process for each frame ***/
	/* Read input image data */
	cv::Mat originalImage = cv::imread(RESOURCE"/cat_dog.jpg");
	cv::Mat inputImage;

	/* Pre-process and Set data to input tensor */
	cv::cvtColor(originalImage, inputImage, cv::COLOR_BGR2RGB);
	cv::resize(inputImage, inputImage, cv::Size(modelInputWidth, modelInputHeight));
	if (inputTensor->type == kTfLiteUInt8) {
		inputImage.convertTo(inputImage, CV_8UC3);
		memcpy(inputTensor->data.int8, inputImage.reshape(0, 1).data, sizeof(int8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	} else {
		inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
		memcpy(inputTensor->data.f, inputImage.reshape(0, 1).data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	}

	/* Run inference */
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

	/* Retrieve the result */
	const float *outputBoxList;
	const float *outputClassList;
	const float *outputScoreList;
	int outputNum;
	if (interpreter->output_tensor(0)->type == kTfLiteUInt8) {
		// todo
	} else {
		outputBoxList = interpreter->output_tensor(0)->data.f;
		outputClassList = interpreter->output_tensor(1)->data.f;
		outputScoreList = interpreter->output_tensor(2)->data.f;
		outputNum = (int)(*interpreter->output_tensor(3)->data.f);
	}

	/* Display bbox */
	std::vector<BBox> bboxList;
	getBBox(bboxList, outputBoxList, outputClassList, outputScoreList, outputNum, 0.5, originalImage.cols, originalImage.rows);
	for (int i = 0; i < bboxList.size(); i++) {
		const BBox bbox = bboxList[i];
		cv::rectangle(originalImage, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), cv::Scalar(255, 255, 0));
		cv::putText(originalImage, labels[bbox.classId], cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 3);
		cv::putText(originalImage, labels[bbox.classId], cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
	}
	cv::imshow("test", originalImage); cv::waitKey(1);


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
