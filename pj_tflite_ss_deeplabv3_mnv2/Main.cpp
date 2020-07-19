/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "ImageProcessor.h"

/*** Macro ***/
/* Model parameters */
#define MODEL_NAME   RESOURCE_DIR"/model/deeplabv3_mnv2_dm05_pascal_quant"
#define LABEL_NAME   RESOURCE_DIR"/model/label_PASCAL_VOC2012.txt"
#define IMAGE_NAME   RESOURCE_DIR"/cat.jpg"

/* Settings */
#define TEST_SPEED_ONLY
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10

int main()
{
	/*** Initialize ***/
	/* Initialize image processor library */
	INPUT_PARAM inputParam;
	snprintf(inputParam.modelFilename, sizeof(inputParam.modelFilename), MODEL_NAME);
	snprintf(inputParam.labelFilename, sizeof(inputParam.labelFilename), LABEL_NAME);
	inputParam.numThreads = 4;
	ImageProcessor_initialize(&inputParam);

#ifdef TEST_SPEED_ONLY
	/* Read an input image */
	cv::Mat originalImage = cv::imread(IMAGE_NAME);

	/* Call image processor library */
	OUTPUT_PARAM outputParam;
	ImageProcessor_process(&originalImage, &outputParam);

	cv::imshow("originalImage", originalImage);
	cv::waitKey(1);

	/*** (Optional) Measure inference time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		ImageProcessor_process(&originalImage, &outputParam);
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);
	cv::waitKey(-1);

#else
	/* Initialize camera */
	int originalImageWidth = 640;
	int originalImageHeight = 480;

	static cv::VideoCapture cap;
	cap = cv::VideoCapture(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, originalImageWidth);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, originalImageHeight);
	// cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
	cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
	while (1) {
		const auto& timeAll0 = std::chrono::steady_clock::now();
		/*** Read image ***/
		const auto& timeCap0 = std::chrono::steady_clock::now();
		cv::Mat originalImage;
		cap.read(originalImage);
		const auto& timeCap1 = std::chrono::steady_clock::now();

		/* Call image processor library */
		const auto& timeProcess0 = std::chrono::steady_clock::now();
		OUTPUT_PARAM outputParam;
		ImageProcessor_process(&originalImage, &outputParam);
		const auto& timeProcess1 = std::chrono::steady_clock::now();

		cv::imshow("test", originalImage);
		if (cv::waitKey(1) == 'q') break;

		const auto& timeAll1 = std::chrono::steady_clock::now();
		printf("Total time = %.3lf [msec]\n", (timeAll1 - timeAll0).count() / 1000000.0);
		printf("Capture time = %.3lf [msec]\n", (timeCap1 - timeCap0).count() / 1000000.0);
		printf("Image processing time = %.3lf [msec]\n", (timeProcess1 - timeProcess0).count() / 1000000.0);
		printf("========\n");
	}

#endif

	/* Fianlize image processor library */
	ImageProcessor_finalize();

	return 0;
}
