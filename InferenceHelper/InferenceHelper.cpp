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
	InferenceHelper* p;
	switch (type) {
	case TENSORFLOW_LITE:
		p = new InferenceHelperTensorflowLite();
		break;
	default:
		PRINT("not supported yet");
		break;
	}
	p->m_helperType = type;
	return p;
}
