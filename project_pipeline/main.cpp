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
#include "src/cpp/pipeline/pipelined_model_runner.h"
#include "src/cpp/pipeline/utils.h"
#endif

/*** Macro ***/
/* Model parameters */
#ifdef USE_EDGETPU
#define USE_EDGETPU_DELEGATE
#define MODEL_FILENAME RESOURCE"/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite"
#else
#define MODEL_FILENAME RESOURCE"/deeplabv3_mnv2_dm05_pascal_quant.tflite"
#endif
#define LABEL_NAME     RESOURCE"/pascal_voc_segmentation_labels.txt"

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 100

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }


/*** Function ***/
static void displayModelInfo(const tflite::Interpreter* interpreter)
{
	const auto& inputIndices = interpreter->inputs();
	int inputNum = (int)inputIndices.size();
	printf("Input num = %d\n", inputNum);
	for (int i = 0; i < inputNum; i++) {
		auto* tensor = interpreter->tensor(inputIndices[i]);
		for (int j = 0; j < tensor->dims->size; j++) {
			printf("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			printf("    tensor[%d]->type: quantized\n", i);
			printf("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			printf("    tensor[%d]->type: not quantized\n", i);
		}
	}

	const auto& outputIndices = interpreter->outputs();
	int outputNum = (int)outputIndices.size();
	printf("Output num = %d\n", outputNum);
	for (int i = 0; i < outputNum; i++) {
		auto* tensor = interpreter->tensor(outputIndices[i]);
		for (int j = 0; j < tensor->dims->size; j++) {
			printf("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			printf("    tensor[%d]->type: quantized\n", i);
			printf("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			printf("    tensor[%d]->type: not quantized\n", i);
		}
	}
}

static void extractTensorAsFloatVector(const TfLiteTensor* tensor, std::vector<float> &output)
{
	int dataNum = 1;
	for (int i = 0; i < tensor->dims->size; i++) {
		dataNum *= tensor->dims->data[i];
	}
	output.resize(dataNum);
	if (tensor->type == kTfLiteUInt8) {
		const auto *valUint8 = tensor->data.uint8;
		for (int i = 0; i < dataNum; i++) {
			float valFloat = (valUint8[i] - tensor->params.zero_point) * tensor->params.scale;
			output[i] = valFloat;
		}
	} else {
		for (int i = 0; i < dataNum; i++) {
			float valFloat = tensor->data.f[i];
			output[i] = valFloat;
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


int main()
{
	/*** Initialize ***/
	/* read label */
	std::vector<std::string> labels;
	readLabel(LABEL_NAME, labels);

	/* Create interpreters */
	int num_segments = 2;
	std::vector<tflite::Interpreter*> interpreters(num_segments);
	std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
	models[0] = tflite::FlatBufferModel::BuildFromFile("resource/deeplabv3_mnv2_dm05_pascal_quant_segment_0_of_2_edgetpu.tflite");
	models[1] = tflite::FlatBufferModel::BuildFromFile("resource/deeplabv3_mnv2_dm05_pascal_quant_segment_1_of_2_edgetpu.tflite");

	std::unique_ptr<coral::PipelinedModelRunner> runner(
		new coral::PipelinedModelRunner(interpreters));
	volatile auto* input_tensor_allocator = runner->GetInputTensorAllocator();
	volatile auto* output_tensor_allocator = runner->GetOutputTensorAllocator();

//	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
//	TFLITE_MINIMAL_CHECK(model != nullptr);
//	tflite::ops::builtin::BuiltinOpResolver resolver;
//	tflite::InterpreterBuilder builder(*model, resolver);
//	std::unique_ptr<tflite::Interpreter> interpreter;
//	builder(&interpreter);
//	TFLITE_MINIMAL_CHECK(interpreter != nullptr);
//	interpreter->SetNumThreads(4);
//#ifdef USE_EDGETPU_DELEGATE
//	size_t num_devices;
//	std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
//	TFLITE_MINIMAL_CHECK(num_devices > 0);
//	const auto& device = devices.get()[0];
//	auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
//	interpreter->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});
//#endif
//	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
//
//
//	/* Get model information */
//	displayModelInfo(interpreter.get());
//	const TfLiteTensor* inputTensor = interpreter->input_tensor(0);
//	const int modelInputHeight = inputTensor->dims->data[1];
//	const int modelInputWidth = inputTensor->dims->data[2];
//	const int modelInputChannel = inputTensor->dims->data[3];
//
//
//	/*** Process for each frame ***/
//	/* Read input image data */
//	cv::Mat originalImage = cv::imread(RESOURCE"/cat_dog.jpg");
//	cv::Mat inputImage;
//
//	/* Pre-process and Set data to input tensor */
//	cv::cvtColor(originalImage, inputImage, cv::COLOR_BGR2RGB);
//	cv::resize(inputImage, inputImage, cv::Size(modelInputWidth, modelInputHeight));
//	if (inputTensor->type == kTfLiteUInt8) {
//		inputImage.convertTo(inputImage, CV_8UC3);
//		memcpy(inputTensor->data.int8, inputImage.reshape(0, 1).data, sizeof(int8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
//	} else {
//		inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
//		memcpy(inputTensor->data.f, inputImage.reshape(0, 1).data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
//	}
//
//	/* Run inference */
//	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
//
//	/* Retrieve the result */
//	int ouputWidth = interpreter->output_tensor(0)->dims->data[2];
//	int ouputHeight = interpreter->output_tensor(0)->dims->data[1];
//	const int64_t *outputMap;
//	if (interpreter->output_tensor(0)->type == kTfLiteInt64) {
//		outputMap = interpreter->output_tensor(0)->data.i64;
//	} else {
//		// todo
//	}
//	
//	/* Display result */
//	cv::Mat outputImage = cv::Mat::zeros(ouputHeight, ouputWidth, CV_8UC3);
//	for (int y = 0; y < ouputHeight; y++) {
//		for (int x = 0; x < ouputWidth; x++) {
//			if (outputMap[y * ouputWidth + x] != 0) {
//				const int CHANNE_NUM = 3;
//				outputImage.data[(y * ouputWidth + x) * CHANNE_NUM] = 0xFF;
//			}
//		}
//	}
//	cv::imshow("originalImage", originalImage); cv::waitKey(1);
//	cv::imshow("outputImage", outputImage); cv::waitKey(1);
//
//
//	/*** (Optional) Measure inference time ***/
//	const auto& t0 = std::chrono::steady_clock::now();
//	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
//		interpreter->Invoke();
//	}
//	const auto& t1 = std::chrono::steady_clock::now();
//	std::chrono::duration<double> timeSpan = t1 - t0;
//	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);
//	
//	cv::waitKey(-1);
	return 0;
}
