#pragma once
#include <cuda.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <iostream>
#include <string>
typedef enum{
	CLNN_OP_N = 0,
	CLNN_OP_T = 1,
	CLNN_OP_C = 2,
} CLNNOpT;
typedef enum{
	CLNN_STATUS_SUCCESS = 0,
	CLNN_STATUS_NOT_INITIALIZED = 1,
	CLNN_STATUS_ALLOC_FAILED = 3,
	CLNN_STATUS_INVALID_VALUE = 7,
	CLNN_STATUS_ARCH_MISMATCH = 8,
	CLNN_STATUS_MAPPING_ERROR = 11,
	CLNN_STATUS_EXECUTION_FAILED = 13,
	CLNN_STATUS_INTERNAL_ERROR = 14,
	CLNN_STATUS_NOT_SUPPORTED = 15
} CLNNStatusT;
enum WeightInitMethod{
	He, Xavier
};
const char* clnnGetErrorString(CLNNStatusT status);
#define checkCLNN(status) { \
	const auto err = status; \
    if (err != CLNN_STATUS_SUCCESS) { \
        std::cerr << "\nError: " << clnnGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("Error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + clnnGetErrorString(err)); \
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
extern curandGenerator_t generator_;
extern size_t MPC, GS, CPM;
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

struct AdamWHalfTask{
	__half* params;
	const __half* grads;
	__half* m;
	__half* v;
	int size;
	float weightDecay;
};
struct AdamWFloatTask{
	float* params;
	const float* grads;
	float* m;
	float* v;
	int size;
	float weightDecay;
};
void LossStats(const __half* dPredictions, const float* dTargets, int numButs, int numCtrls, int batchSize, float* butLoss, float* axesLoss);
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
void AdamWHalfMulti(const AdamWHalfTask* tasks, int taskCount, int totalSize, float lr, int t);
void AdamWFloatMulti(const AdamWFloatTask* tasks, int taskCount, int totalSize, float lr, int t);
void LossBackprop(__half* dGradient, const __half* dPredictions, const float* dTargets, float clip, int size, int numCtrls, int numButs, int batchSize);
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
void LayerNormForward(__half* y, const __half* x, const float* g, const float* b, float* mean, float* var, int N, int C, int HW, bool spatialMode);
void LayerNormBackward(__half* dx, const __half* dy, const __half* x, const float* g, float* dG, float* dB, const float* mean, const float* var, void* workspace, size_t workspaceSize, int N, int C, int HW, bool spatialMode);
bool IsnanHalf(const __half* data, int size);
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, int H, int W, int inC, int mosaicW, int tileW, int tileH, int gridW, float scale, cudaStream_t stream = nullptr);
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, __half* AttentionWeights, const float* attentionMask, const float* relPosBias, const int* relPosIndex, int relPosSize, int batchSize, int tokens, int headDim, int heads, int maskBatchSize, int maskHeads);
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const __half* Att, __half* dQ, __half* dK, __half* dV, float* dAttWorkspace, size_t workspaceElements, int batchSize, int tokens, int headDim, int heads);
void AccumulateRelPosBiasGrad(const float* dAtt, const int* relPosIndex, float* gradBias, int batchSize, int tokens, int heads, int relPosSize, float scale);
void ExtractPatches(const __half* in, __half* out, int B, int C, int H, int W, int P);
void CombinePatchGrads(const __half* dy, __half* dx, int B, int C, int H, int W, int P);
void SumPositionalGrad(const __half* grad, __half* out, int B, int C, int P, bool first, float scale);
void AddPerTokenEmbedding(__half* output, const __half* embed, int batch, int tokens, int embedDim);
void TanhInPlace(__half* data, int size);
void TanhBackward(__half* grad, const __half* activations, int size);
void AttentionPoolForward(const __half* input, const __half* query, __half* output, float* attnWeights, float* tempBuffer, int batchSize, int tokens, int embedDim, int numQueries, float invSqrtDim);
void AttentionPoolBackward(const __half* grad, const __half* input, const __half* query, const float* attnWeights, float* tempBuffer, float* batchSums, __half* outGrad, __half* gradQuery, int batchSize, int tokens, int embedDim, int numQueries, float invSqrtDim);
void ScaleArrayHalf(__half* data, size_t count, float scale);
void AddBias(__half* output, const __half* bias, int channels, int batchSize);
void AddTensor(float alpha, __half* A, float beta, const __half* B, int size);
void AddTensorBroadcast(float alpha, const __half* B, float beta, __half* C, int batch, int elementsPerBatch);
void AccumulateBiasGrad(const __half* grad, __half* gradBias, int channels, int batch, float scale, bool reset);
void PackColumnsToHeads(const __half* inputQ, const __half* inputK, const __half* inputV, __half* outputQ, __half* outputK, __half* outputV, int batch, int tokens, int embedDim, int numHeads);
void PackColumnsToHeads(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads);
void PackHeadsToColumns(const __half* inputQ, const __half* inputK, const __half* inputV, __half* outputQ, __half* outputK, __half* outputV, int batch, int tokens, int embedDim, int numHeads);
void PackHeadsToColumns(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads);
void PatchMerge(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols);
void PatchUnmerge(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols);
void TokensToSpatial(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols);
void SpatialToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols);
void TokensToWindows(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth);
void WindowsToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth);
void ScaleNearestNeighborForward(const __half* input, __half* output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth);
void ScaleNearestNeighborBackward(const __half* gradOut, __half* gradIn, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth);
void DropPathBuildMask(float* mask, int batch, float keepProb);
void DropPathApply(__half* data, const float* mask, int batch, int elementsPerBatch);
CLNNStatusT InitCublas();
CLNNStatusT CLNNGemmEx(CLNNOpT transa, CLNNOpT transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType);
CLNNStatusT CLNNGemmStridedBatchedEx(CLNNOpT transa, CLNNOpT transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType);
int ConvertSmVer2Cores(int major, int minor);
template<class Ta, class Tb>
Ta DivCeil(Ta a, Tb b){ return (a + b - 1)/b; }
void GetLaunchConfigGridStride(size_t n, size_t& blocks, size_t& tpb);
void InitCUDA();
void WeightInit(__half* weights, int elementCount, int fanIn, int fanOut, WeightInitMethod method);
template<class T>
void CUDAMallocZero(T** ptr, size_t size){
	checkCUDA(cudaMalloc(reinterpret_cast<void**>(ptr), size));
	checkCUDA(cudaMemset(*ptr, 0, size));
}
