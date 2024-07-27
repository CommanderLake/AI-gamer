#include "BatchNorm.h"
#include "common.h"
#include <vector>
BatchNorm::BatchNorm(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, cudnnBatchNormMode_t bnMode, int bitchSize, const char* layerName, bool train): cudnnHandle_(cudnnHandle), bnMode_(bnMode), bitchSize_(bitchSize), inData_(nullptr), epsilon_(1e-6){
	layerName_ = layerName;
	train_ = train;
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outNCHW_ = n*c*h*w;
	outC_ = c;
	checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasDesc_));
	checkCUDNN(cudnnDeriveBNTensorDescriptor(bnScaleBiasDesc_, outDesc, bnMode_));
	checkCUDA(cudaMalloc(&outData_, outNCHW_*sizeof(__half)));
	checkCUDA(cudaMemset(outData_, 0, outNCHW_*sizeof(__half)));
	const auto bnSizeBytes = outC_*sizeof(float);
	checkCUDA(cudaMalloc(&bnScale_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnBias_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnRunningMean_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnRunningVar_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnSavedMean_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnSavedInvVariance_, bnSizeBytes));
	const std::vector<float> bnScaleInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(bnScale_, bnScaleInit.data(), bnSizeBytes, cudaMemcpyHostToDevice));
	checkCUDA(cudaMemset(bnBias_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnRunningMean_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnRunningVar_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnSavedMean_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnSavedInvVariance_, 0, bnSizeBytes));
	if(train_){
		checkCUDNN(cudnnCreateTensorDescriptor(&gradDesc_));
		checkCUDNN(cudnnSetTensor4dDescriptor(gradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w));
		checkCUDA(cudaMalloc(&gradBnScale_, bnSizeBytes));
		checkCUDA(cudaMalloc(&gradBnBias_, bnSizeBytes));
		checkCUDA(cudaMemset(gradBnScale_, 0, bnSizeBytes));
		checkCUDA(cudaMemset(gradBnBias_, 0, bnSizeBytes));
		checkCUDA(cudaMalloc(&m_bnScale_, bnSizeBytes));
		checkCUDA(cudaMalloc(&v_bnScale_, bnSizeBytes));
		checkCUDA(cudaMalloc(&m_bnBias_, bnSizeBytes));
		checkCUDA(cudaMalloc(&v_bnBias_, bnSizeBytes));
		checkCUDA(cudaMemset(m_bnScale_, 0, bnSizeBytes));
		checkCUDA(cudaMemset(v_bnScale_, 0, bnSizeBytes));
		checkCUDA(cudaMemset(m_bnBias_, 0, bnSizeBytes));
		checkCUDA(cudaMemset(v_bnBias_, 0, bnSizeBytes));
	}
}
BatchNorm::~BatchNorm(){
	cudaFree(bnScale_);
	cudaFree(bnBias_);
	cudaFree(bnRunningMean_);
	cudaFree(bnRunningVar_);
	cudaFree(bnSavedMean_);
	cudaFree(bnSavedInvVariance_);
	checkCUDNN(cudnnDestroyTensorDescriptor(bnScaleBiasDesc_));
	if(train_){
		cudaFree(gradBnScale_);
		cudaFree(gradBnBias_);
		cudaFree(m_bnScale_);
		cudaFree(v_bnScale_);
		cudaFree(m_bnBias_);
		cudaFree(v_bnBias_);
	}
}
__half* BatchNorm::Forward(__half* data, bool train){
	inData_ = data;
	if(train){
		checkCUDNN(cudnnBatchNormalizationForwardTraining(cudnnHandle_, bnMode_, &alpha, &beta0, outDesc_, data, outDesc_, outData_, bnScaleBiasDesc_, bnScale_, bnBias_, 0.1, bnRunningMean_, bnRunningVar_, epsilon_, bnSavedMean_, bnSavedInvVariance_));
	} else{
		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle_, bnMode_, &alpha, &beta0, outDesc_, data, outDesc_, outData_, bnScaleBiasDesc_, bnScale_, bnBias_, bnRunningMean_, bnRunningVar_, epsilon_));
	}
	return outData_;
}
__half* BatchNorm::Backward(__half* grad){
	checkCUDNN(cudnnBatchNormalizationBackward(
		cudnnHandle_,
		bnMode_,
		&alpha, &beta0, &alpha, &beta1,
		outDesc_, inData_,	// x
		gradDesc_, grad,	// dy (backpropagated gradient input)
		gradDesc_, grad,	// dx (gradient output)
		bnScaleBiasDesc_,	// dBnScaleBiasDesc
		bnScale_,			// bnScaleData (batch normalization scale parameter)
		gradBnScale_,		// dBnScaleData (gradients of bnScaleData)
		gradBnBias_,		// dBnBiasData (gradients of bnBiasData)
		epsilon_,
		bnSavedMean_, bnSavedInvVariance_));
	return grad;
}
void BatchNorm::UpdateParameters(float learningRate){
	//SGDFloat(bnScale_, learningRate, gradBnScale_, outC_);
	//SGDFloat(bnBias_, learningRate, gradBnBias_, outC_);
	AdamWFloat(bnScale_, m_bnScale_, v_bnScale_, learningRate, gradBnScale_, outC_, t_, 0.0001F);
	AdamWFloat(bnBias_, m_bnBias_, v_bnBias_, learningRate, gradBnBias_, outC_, t_, 0.0001F);
	++t_;
}
void BatchNorm::SaveParameters(std::ofstream& file, float* buffer) const{
	cudaMemcpy(buffer, bnScale_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnBias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnRunningMean_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnRunningVar_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnSavedMean_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnSavedInvVariance_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}

void BatchNorm::LoadParameters(std::ifstream& file, float* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnScale_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnBias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnRunningMean_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnRunningVar_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnSavedMean_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnSavedInvVariance_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}

void BatchNorm::SaveOptimizerState(std::ofstream& file, float* buffer) const{
	cudaMemcpy(buffer, m_bnScale_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, v_bnScale_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, m_bnBias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, v_bnBias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}

void BatchNorm::LoadOptimizerState(std::ifstream& file, float* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(m_bnScale_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(v_bnScale_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(m_bnBias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(v_bnBias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
bool BatchNorm::HasParameters() const{
	return true;
}
bool BatchNorm::HasOptimizerState() const{
	return true;
}
size_t BatchNorm::GetParameterSize() const{
	return outC_*sizeof(float);
}
size_t BatchNorm::GetOptimizerStateSize() const{
	return outC_*sizeof(float);
}