
#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

namespace cv {
	class Mat;
};

#define NUM_MAX_RESULT 100

typedef struct {
	char workDir[256];
	int  numThreads;
} INPUT_PARAM;

typedef struct {
	int resultNum;
	struct {
		int classId;
		char label[256];
		double score;
		int x;
		int y;
		int width;
		int height;
	} RESULTS[NUM_MAX_RESULT];
} OUTPUT_PARAM;

int ImageProcessor_initialize(const INPUT_PARAM *inputParam);
int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam);
int ImageProcessor_finalize(void);

#endif
