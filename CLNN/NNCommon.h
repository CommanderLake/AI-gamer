#pragma once
#include "CuCommon.cuh"
#include <cuda_fp16.h>
#include <cudnn.h>
struct ConvolutionAlgorithms{
	cudnnConvolutionFwdAlgo_t fwdAlgo;
	cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
	size_t workspaceSize;
};
ConvolutionAlgorithms GetConvolutionAlgorithms(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, bool isTraining);
void HalfToFloatAsm(float* dst, __half* src, int count);
void FloatToHalfAsm(float* src, __half* dst, int count);
void PrintDataHalfDevice(const __half* data, size_t size, const char* label);
void PrintDataFloatDevice(const float* data, size_t size, const char* label);
void PrintDataFloatHost(const float* data, size_t size, const char* label);
void PrintDataCharHost(const unsigned char* data, size_t size, const char* label);
void SummarizeHalfDevice(const __half* data, size_t size, const char* label);
void SummarizeFloatDevice(const float* data, size_t size, const char* label);
void ClearScreen(char fill = ' ');
void OrthogonalInit(__half* weights, int rows, int cols, WeightInitMethod method);