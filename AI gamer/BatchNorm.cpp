#include "BatchNorm.h"
#include "common.h"
#include <vector>
BatchNorm::BatchNorm(cudnnHandle_t cudnnHandle, cudnnBatchNormMode_t bnMode, int batchSize, int channels, int height, int width, const char* layerName, bool train, float weightDecay): cudnnHandle_(cudnnHandle), bnMode_(bnMode), batchSize_(batchSize),
	outC_(channels), outHeight_(height), outWidth_(width), epsilon_(1e-6), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*outC_*outHeight_*outWidth_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, outHeight_, outWidth_));
	checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasDesc_));
	checkCUDNN(cudnnDeriveBNTensorDescriptor(bnScaleBiasDesc_, outDesc_, bnMode_));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	const auto bnSizeBytes = outC_*sizeof(float);
	CUDAMallocZero(&bnScale_, bnSizeBytes);
	CUDAMallocZero(&bnBias_, bnSizeBytes);
	CUDAMallocZero(&bnRunningMeanTrain_, bnSizeBytes);
	CUDAMallocZero(&bnRunningVarTrain_, bnSizeBytes);
	CUDAMallocZero(&bnRunningMeanInfer_, bnSizeBytes);
	CUDAMallocZero(&bnRunningVarInfer_, bnSizeBytes);
	CUDAMallocZero(&bnSavedMean_, bnSizeBytes);
	CUDAMallocZero(&bnSavedInvVariance_, bnSizeBytes);
	const std::vector<float> bnScaleInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(bnScale_, bnScaleInit.data(), bnSizeBytes, cudaMemcpyHostToDevice));
	if(train_){
		CUDAMallocZero(&gradBnScale_, bnSizeBytes);
		CUDAMallocZero(&gradBnBias_, bnSizeBytes);
		if(useAdamW_){
			CUDAMallocZero(&m_BnScale_, bnSizeBytes);
			CUDAMallocZero(&v_BnScale_, bnSizeBytes);
			CUDAMallocZero(&m_BnBias_, bnSizeBytes);
			CUDAMallocZero(&v_BnBias_, bnSizeBytes);
		}
	}
}
BatchNorm::~BatchNorm(){
	cudaFree(bnScale_);
	cudaFree(bnBias_);
	cudaFree(bnRunningMeanTrain_);
	cudaFree(bnRunningVarTrain_);
	cudaFree(bnRunningMeanInfer_);
	cudaFree(bnRunningVarInfer_);
	cudaFree(bnSavedMean_);
	cudaFree(bnSavedInvVariance_);
	checkCUDNN(cudnnDestroyTensorDescriptor(bnScaleBiasDesc_));
	if(train_){
		cudaFree(gradBnScale_);
		cudaFree(gradBnBias_);
		if(useAdamW_){
			cudaFree(m_BnScale_);
			cudaFree(v_BnScale_);
			cudaFree(m_BnBias_);
			cudaFree(v_BnBias_);
		}
	}
}
__half* BatchNorm::Forward(__half* data){
	inData_ = data;
	if(train_){
		checkCUDNN(cudnnBatchNormalizationForwardTraining(cudnnHandle_, bnMode_, &alpha, &beta0, outDesc_, data, outDesc_, outData_, bnScaleBiasDesc_, bnScale_, bnBias_, 1.0, bnRunningMeanTrain_, bnRunningVarTrain_, epsilon_, bnSavedMean_, bnSavedInvVariance_));
	} else{
		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle_, bnMode_, &alpha, &beta0, outDesc_, data, outDesc_, outData_, bnScaleBiasDesc_, bnScale_, bnBias_, bnRunningMeanInfer_, bnRunningVarInfer_, epsilon_));
	}
	return outData_;
}
__half* BatchNorm::Backward(__half* grad){
	checkCUDNN(cudnnBatchNormalizationBackward(
		cudnnHandle_,
		bnMode_,
		&alpha, &beta0, &alpha, &beta1,
		outDesc_, inData_,	 
		outDesc_, grad,	    
		outDesc_, grad,	   
		bnScaleBiasDesc_,	 
		bnScale_,			     
		gradBnScale_,		    
		gradBnBias_,		    
		epsilon_,
		bnSavedMean_, bnSavedInvVariance_));
	return grad;
}
void BatchNorm::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWFloat(bnScale_, gradBnScale_, m_BnScale_, v_BnScale_, learningRate, t_, weightDecay_, outC_);
		AdamWFloat(bnBias_, gradBnBias_, m_BnBias_, v_BnBias_, learningRate, t_, weightDecay_, outC_);
		++t_;
	} else{
		SGDFloat(bnScale_, gradBnScale_, outC_, learningRate, weightDecay_);
		SGDFloat(bnBias_, gradBnBias_, outC_, learningRate, weightDecay_);
	}
}
void BatchNorm::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, bnScale_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnBias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnRunningMeanTrain_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnRunningVarTrain_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnSavedMean_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, bnSavedInvVariance_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void BatchNorm::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnScale_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnBias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnRunningMeanTrain_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bnRunningMeanInfer_, bnRunningMeanTrain_, outC_*sizeof(float), cudaMemcpyDeviceToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnRunningVarTrain_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bnRunningVarInfer_, bnRunningVarTrain_, outC_*sizeof(float), cudaMemcpyDeviceToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnSavedMean_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(bnSavedInvVariance_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
void BatchNorm::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_BnScale_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, v_BnScale_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, m_BnBias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, v_BnBias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void BatchNorm::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(m_BnScale_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(v_BnScale_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(m_BnBias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(v_BnBias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
size_t BatchNorm::GetParameterSize(){
	return outC_*sizeof(float);
}
size_t BatchNorm::GetOptimizerStateSize(){
	return outC_*sizeof(float);
}
void BatchNorm::SetTrain(bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, outC_, outHeight_, outWidth_));
}