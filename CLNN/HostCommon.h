#pragma once
#include "CuCommon.h"
#include <cuda_fp16.h>
__declspec(dllexport) void HalfToFloatAsm(float* dst, __half* src, int count);
__declspec(dllexport) void FloatToHalfAsm(float* src, __half* dst, int count);
__declspec(dllexport) void PrintDataHalfDevice(const __half* data, size_t size, const char* label);
__declspec(dllexport) void PrintDataFloatDevice(const float* data, size_t size, const char* label);
__declspec(dllexport) void PrintDataFloatHost(const float* data, size_t size, const char* label);
__declspec(dllexport) void PrintDataCharHost(const unsigned char* data, size_t size, const char* label);
__declspec(dllexport) void SummarizeHalfDevice(const __half* data, size_t size, std::string label);
__declspec(dllexport) void SummarizeFloatDevice(const float* data, size_t size, std::string label);
__declspec(dllexport) void SummaryPrint();
__declspec(dllexport) void ClearScreen(char fill = ' ');
__declspec(dllexport) void OrthogonalInit(__half* dWeights, int rows, int cols, WeightInitMethod method);