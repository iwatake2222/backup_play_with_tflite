/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "InferenceHelper.h"
#include "InferenceHelperTensorflowLite.h"

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

InferenceHelper* InferenceHelper::create(const InferenceHelper::HELPER_TYPE type)
{
	InferenceHelper* p = NULL;
	switch (type) {
	case TENSORFLOW_LITE:
		p = new InferenceHelperTensorflowLite();
		break;
#ifdef TFLITE_DELEGATE_EDGETPU
	case TENSORFLOW_LITE_EDGETPU:
		p = new InferenceHelperTensorflowLite();
		break;
#endif
	default:
		PRINT("not supported yet");
		exit(1);
		break;
	}
	p->m_helperType = type;
	return p;
}


TensorInfo::TensorInfo()
{
	index = -1;
	type = TENSOR_TYPE_NONE;
	data = NULL;;
	dims.clear();
	quant.scale = 0;
	quant.zeroPoint = 0;
	m_dataFp32 = NULL;
}

TensorInfo::~TensorInfo()
{
	if (m_dataFp32 != NULL) {
		delete[] m_dataFp32;
	}
}

float* TensorInfo::getDataAsFloat()
{
	if (type == TENSOR_TYPE_UINT8) {
		int dataNum = 1;
		for (int i = 0; i < (int)dims.size(); i++) dataNum *= dims[i];
		if (m_dataFp32 == NULL) {
			m_dataFp32 = new float[dataNum];
		}
		for (int i = 0; i < dataNum; i++) {
			const uint8_t* valUint8 = (uint8_t*)data;
			float valFloat = (valUint8[i] - quant.zeroPoint) * quant.scale;
			m_dataFp32[i] = valFloat;
		}
		return m_dataFp32;
	} else if (type == TENSOR_TYPE_FP32) {
		return (float*)data;
	} else {
		PRINT("invalid call");
		return NULL;
	}
}

