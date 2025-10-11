#pragma once
#include "WeightInitMethod.h"
#include <cuda.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <iostream>
#include <string>
const char* cublasGetErrorString(cublasStatus_t status);
#define checkCUBLAS(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "\ncuBLAS error: " << cublasGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuBLAS error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cublasGetErrorString(status)); \
    } \
}
#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "\ncuDNN error: " << cudnnGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuDNN error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudnnGetErrorString(status)); \
    } \
}
#define checkCUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "\nCUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(status)); \
    } \
}
#define EPSILON_F 1e-6f
extern curandGenerator_t generator_;
extern int GS, BS, RPB, CPB, TPG, maxTPB, smemPB;
struct pixARGB{
	unsigned char B;
	unsigned char G;
	unsigned char R;
	unsigned char A;
};
struct pixRGB{
	unsigned char B;
	unsigned char G;
	unsigned char R;
};
float MseLoss(const __half* dPredictions, const float* dTargets, int size);
void MseLoss2(const __half* dPredictions, const float* dTargets, int numButs, int numCtrls, int batchSize, float* butLoss, float* axesLoss);
void BlockShiftHalf(__half* hPtr, int shiftBy, int blocksToShift);
void ConvertByteToHalf(const unsigned char* input, __half* output, size_t size, bool normalize);
void ConvertHalfToByte(const __half* input, unsigned char* output, size_t size, bool normalize);
void ConvertFloatToHalf(const float* input, __half* output, size_t size);
void ConvertHalfToFloat(const __half* input, float* output, size_t size);
void ConvertFloatToHalfScale(__half* halfWeights, const float* weights, size_t size, float scale);
void SGDHalf(__half* params, const __half* grads, int size, float learningRate, float weightDecay);
void SGDFloat(float* params, const float* grads, int size, float learningRate, float weightDecay);
void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, float lr, int t, float weightDecay, int size);
void AdamWFloat(float* params, const float* grads, float* m, float* v, float learningRate, int t, float weightDecay, int size);
void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, float clip, int size);
void SplitGradient(__half* dGradient, const __half* dPredictions, const float* dTargets, float clip, int size, int numCtrls, int numButs, int batchSize);
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, int numCtrls, int numButs, int size);
void GetPrediction(const __half* predBatch, float* prediction, int numCtrls, int batchSize);
void LeakyReluForward(const __half* dataIn, __half* dataOut, int size, float negativeSlope, cudaStream_t stream = nullptr);
void LeakyReluBackward(__half* grad, const __half* dataIn, int size, float negativeSlope, cudaStream_t stream = nullptr);
void SwishForward(const __half* dataIn, __half* outData, int size, cudaStream_t stream = nullptr);
void SwishBackward(__half* grad, const __half* dataIn, int size, cudaStream_t stream = nullptr);
void SigmoidForward(const __half* dataIn, __half* dataOut, int numCtrls, int numButs, int size, cudaStream_t cudaStream = nullptr);
void SigmoidBackward(__half* grad, const __half* dataIn, int numCtrls, int numButs, int size, cudaStream_t cudaStream = nullptr);
void GELUForward(const __half* dataIn, __half* dataOut, int size, cudaStream_t stream = nullptr);
void GELUBackward(__half* grad, const __half* dataIn, int size, cudaStream_t stream = nullptr);
void LayerNormForward(__half* y, const __half* x, const float* g, const float* b, float* mean, float* var, int N, int C, int HW);
void LayerNormBackward(__half* dx, const __half* dy, const __half* x, const float* g, float* dG, float* dB, const float* mean, const float* var, void* workspace, size_t workspace_size, int N, int C, int HW);
bool IsnanHalf(const __half* data, int size);
void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size, float scale);
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, int H, int W, int inC, int mosaicW, int tileW, int tileH, int gridW, cudaStream_t stream = nullptr);
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int B, int T, int D, int H);
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* Att, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H);
void ExtractPatches(const __half* in, __half* out, int B, int C, int H, int W, int P);
void CombinePatchGrads(const __half* dy, __half* dx, int B, int C, int H, int W, int P);
void SumPositionalGrad(const __half* grad, __half* out, int B, int C, int P, bool first);
void AttentionPoolForward(const __half* input, const __half* query, __half* output, float* attnWeights, float* tempBuffer, int batchSize, int tokens, int embedDim, float invSqrtDim);
void AttentionPoolBackward(const __half* grad, const __half* input, const __half* query, const float* attnWeights, float* tempBuffer, float* batchSums, __half* outGrad, __half* gradQuery, int batchSize, int tokens, int embedDim, float invSqrtDim);
int ConvertSmVer2Cores(int major, int minor);
int DivCeil(int a, int b);
void GetLaunchConfig(int n, int& blocks, int& tpb);
void InitCUDA();
void WeightInit(__half* weightHalf, int numWeights, int fanIn, WeightInitMethod method);
template <typename T>
void CUDAMallocZero(T** ptr, size_t size){
	checkCUDA(cudaMalloc(reinterpret_cast<void**>(ptr), size));
	checkCUDA(cudaMemset(*ptr, 0, size));
}