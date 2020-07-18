
#ifndef INFERENCE_HELPER_
#define INFERENCE_HELPER_

class InferenceHelper {
public:
	typedef enum {
		TENSORFLOW_LITE,
		TENSORFLOW_LITE_EDGETPU,
		TENSORFLOW_LITE_GPU,
		NCNN,
		NCNN_VULKAN,
		MNN,
		OPEN_CV,
		OPEN_CV_OPENCL,
		TENSOR_RT,
	} HELPER_TYPE;

	typedef enum {
		TENSOR_TYPE_UINT8,
		TENSOR_TYPE_FP32,
		TENSOR_TYPE_INT32,
		TENSOR_TYPE_INT64,
	} TENSOR_TYPE;

	typedef struct {
		int          index;
		TENSOR_TYPE  type;
		void         *data;
		int          width;
		int          height;
		int          channel;
		struct{
			float scale;
			int   zeroPoint;
		} quant;
	} TENSOR_INFO;

public:
	virtual ~InferenceHelper() {}
	virtual int initialize(const char *modelFilename, const int numThreads) = 0;
	virtual int finalize(void) = 0;
	virtual int inference(void) = 0;
	virtual int getTensorByName(const char *name, TENSOR_INFO *tensorInfo) = 0;
	virtual int getTensorByIndex(const int index, TENSOR_INFO *tensorInfo) = 0;
	virtual int setBufferToTensorByName(const char *name, const char *data, const unsigned int dataSize) = 0;
	virtual int setBufferToTensorByIndex(const int index, const char *data, const unsigned int dataSize) = 0;
	static InferenceHelper* create(const HELPER_TYPE typeFw);

protected:
	HELPER_TYPE m_helperType;
};

#endif
