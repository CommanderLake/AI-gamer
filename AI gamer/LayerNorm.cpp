#include "LayerNorm.h"
#include "common.h"
#include <vector>
LayerNorm::LayerNorm(cudnnTensorDescriptor_t outDesc, int batchSize, const char* layerName) : batchSize_(batchSize), inData_(nullptr), epsilon_(1e-6){
	layerName_ = layerName;
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outC_ = c;
	outHW_ = h*w;
	outCHW_ = c*outHW_;
	outNCHW_ = n*outCHW_;
	checkCUDNN(cudnnCreateTensorDescriptor(&gradDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(gradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w));
	checkCUDA(cudaMalloc(&outData_, outNCHW_*sizeof(__half)));
	checkCUDA(cudaMemset(outData_, 0, outNCHW_*sizeof(__half)));
	const auto paramSizeBytes = c*sizeof(float);
	checkCUDA(cudaMalloc(&gamma_, paramSizeBytes));
	checkCUDA(cudaMalloc(&beta_, paramSizeBytes));
	checkCUDA(cudaMalloc(&gradGamma_, paramSizeBytes));
	checkCUDA(cudaMalloc(&gradBeta_, paramSizeBytes));
	checkCUDA(cudaMalloc(&mean_, paramSizeBytes));
	checkCUDA(cudaMalloc(&variance_, paramSizeBytes));
	checkCUDA(cudaMalloc(&mGamma_, paramSizeBytes));
	checkCUDA(cudaMalloc(&vGamma_, paramSizeBytes));
	checkCUDA(cudaMalloc(&mBeta_, paramSizeBytes));
	checkCUDA(cudaMalloc(&vBeta_, paramSizeBytes));
	const std::vector<float> gammaInit(c, 1.0f);
	checkCUDA(cudaMemcpy(gamma_, gammaInit.data(), paramSizeBytes, cudaMemcpyHostToDevice));
	checkCUDA(cudaMemset(beta_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(gradGamma_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(gradBeta_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(mean_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(variance_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(mGamma_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(vGamma_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(mBeta_, 0, paramSizeBytes));
	checkCUDA(cudaMemset(vBeta_, 0, paramSizeBytes));
}
LayerNorm::~LayerNorm(){
	cudaFree(outData_);
	cudaFree(gamma_);
	cudaFree(beta_);
	cudaFree(gradGamma_);
	cudaFree(gradBeta_);
	cudaFree(mean_);
	cudaFree(variance_);
	cudaFree(mGamma_);
	cudaFree(vGamma_);
	cudaFree(mBeta_);
	cudaFree(vBeta_);
	cudnnDestroyTensorDescriptor(gradDesc_);
}
__half* LayerNorm::Forward(__half* data){
	inData_ = data;
	LayerNormForward(outData_, data, gamma_, beta_, mean_, variance_, batchSize_, outC_, outHW_, epsilon_);
	cudaDeviceSynchronize();
	if(const cudaError_t err = cudaGetLastError(); err != cudaSuccess){
		printf("CUDA error in Forward: %s\n", cudaGetErrorString(err));
	}
	return outData_;
}
__half* LayerNorm::Backward(__half* grad){
	LayerNormBackward(grad, grad, inData_, gamma_, gradGamma_, gradBeta_, mean_, variance_, batchSize_, outC_, outHW_, epsilon_);
	cudaDeviceSynchronize();
	if(const cudaError_t err = cudaGetLastError(); err != cudaSuccess){
		printf("CUDA error in Backward: %s\n", cudaGetErrorString(err));
	}
	return grad;
}
void LayerNorm::UpdateParameters(float learningRate){
	AdamWFloat(gamma_, mGamma_, vGamma_, learningRate, gradGamma_, outNCHW_, t_, 0.001F);
	AdamWFloat(beta_, mBeta_, vBeta_, learningRate, gradBeta_, outNCHW_, t_, 0.001F);
	++t_;
}