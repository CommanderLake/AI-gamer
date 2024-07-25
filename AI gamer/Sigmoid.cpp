#include "Sigmoid.h"
#include "common.h"
#include <vector>
Sigmoid::Sigmoid(cudnnTensorDescriptor_t outDesc, int numSigmoidOutputs, int batchSize, const char* layerName) : numSigmoidOutputs_(numSigmoidOutputs), batchSize_(batchSize), data_(nullptr){
	layerName_ = layerName;
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outCHW_ = c*h*w;
	outNCHW_ = n*outCHW_;
	checkCUDNN(cudnnCreateTensorDescriptor(&gradDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(gradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w));
}
Sigmoid::~Sigmoid() {
    cudnnDestroyTensorDescriptor(gradDesc_);
}
__half* Sigmoid::Forward(__half* data, bool train) {
	data_ = data;
    SigmoidForward(data, numSigmoidOutputs_, batchSize_, outCHW_);
	cudaDeviceSynchronize();
	if(const cudaError_t err = cudaGetLastError(); err != cudaSuccess){
		printf("CUDA error in Forward: %s\n", cudaGetErrorString(err));
	}
    return data;
}
__half* Sigmoid::Backward(__half* grad) {
    SigmoidBackward(grad, data_, numSigmoidOutputs_, batchSize_, outCHW_);
	cudaDeviceSynchronize();
	if(const cudaError_t err = cudaGetLastError(); err != cudaSuccess){
		printf("CUDA error in Backward: %s\n", cudaGetErrorString(err));
	}
    return grad;
}