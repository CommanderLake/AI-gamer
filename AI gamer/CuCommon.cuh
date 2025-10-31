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
	const auto err = status; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "\ncuBLAS error: " << cublasGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuBLAS error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cublasGetErrorString(err)); \
    } \
}
#define checkCUDNN(status) { \
	const auto err = status; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "\ncuDNN error: " << cudnnGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuDNN error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudnnGetErrorString(err)); \
    } \
}
#define checkCUDA(status) { \
	const auto err = status; \
    if (err != cudaSuccess) { \
        std::cerr << "\nCUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
    } \
}
#define EPSILON_F 1e-6f
extern curandGenerator_t generator_;
extern size_t GS, BS, RPB, CPB, TPG, maxTPB, smemPB;
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
void Loss2(const __half* dPredictions, const float* dTargets, int numButs, int numCtrls, int batchSize, float* butLoss, float* axesLoss);
void BlockShiftHalf(__half* dPtr, int shiftBy, int blocksToShift);
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
void AsinhForward(const __half* dataIn, __half* dataOut, int size, float alpha, cudaStream_t stream = nullptr);
void AsinhBackward(__half* grad, const __half* activated, int size, float alpha, cudaStream_t stream = nullptr);
void LayerNormForward(__half* y, const __half* x, const float* g, const float* b, float* mean, float* var, int N, int C, int HW);
void LayerNormBackward(__half* dx, const __half* dy, const __half* x, const float* g, float* dG, float* dB, const float* mean, const float* var, void* workspace, size_t workspaceSize, int N, int C, int HW);
bool IsnanHalf(const __half* data, int size);
void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size, float scale);
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, int H, int W, int inC, int mosaicW, int tileW, int tileH, int gridW, float scale, cudaStream_t stream = nullptr);
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, __half* AttentionWeights, int batchSize, int tokens, int headDim, int heads);
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const __half* Att, __half* dQ, __half* dK, __half* dV, float* dAttWorkspace, size_t workspaceElements, int batchSize, int tokens, int headDim, int heads);
void ExtractPatches(const __half* in, __half* out, int B, int C, int H, int W, int P);
void CombinePatchGrads(const __half* dy, __half* dx, int B, int C, int H, int W, int P);
void SumPositionalGrad(const __half* grad, __half* out, int B, int C, int P, bool first, float scale);
void AddPerTokenEmbedding(__half* output, const __half* embed, int batch, int tokens, int embedDim);
void AttentionPoolForward(const __half* input, const __half* query, __half* output, float* attnWeights, float* tempBuffer, int batchSize, int tokens, int embedDim, float invSqrtDim);
void AttentionPoolBackward(const __half* grad, const __half* input, const __half* query, const float* attnWeights, float* tempBuffer, float* batchSums, __half* outGrad, __half* gradQuery, int batchSize, int tokens, int embedDim, float invSqrtDim);
void ScaleArrayHalf(__half* data, size_t count, float scale);
void AddBias(__half* output, const __half* bias, int channels, int batch);
void AccumulateBiasGrad(const __half* grad, __half* gradBias, int channels, int batch, float scale, bool reset);
void PackColumnsToHeads(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads);
void PackHeadsToColumns(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads);
void TokensToSpatial(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols);
void SpatialToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols);
int ConvertSmVer2Cores(int major, int minor);
size_t DivCeil(size_t a, size_t b);
void GetLaunchConfigGridStride(size_t n, size_t& blocks, size_t& tpb);
void InitCUDA();
void WeightInit(__half* weights, int elementCount, int fanIn, int fanOut, WeightInitMethod method);
template<class T>
void CUDAMallocZero(T** ptr, size_t size){
	checkCUDA(cudaMalloc(reinterpret_cast<void**>(ptr), size));
	checkCUDA(cudaMemset(*ptr, 0, size));
}