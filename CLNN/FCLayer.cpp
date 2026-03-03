#include "FCLayer.h"
#include "NNCommon.h"
#include "CuCommon.cuh"
#include <iostream>
#include <algorithm>
FCLayer::FCLayer(const int batchSize, const int inC, const int outC, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod, const bool useBias) :
	batchSize_(batchSize), inC_(inC), outC_(outC), useBias_(useBias), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*outC_;
	alphaWeights_ = 1.0f/(batchSize_*gradAccumLength_);
	weightCount_ = inC_*outC_;
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	if(useBias_){ CUDAMallocZero(&biases_, outC_*sizeof(__half)); }
	if(train_){
		WeightInit(weights_, weightCount_, inC_, outC_, weightInitMethod);
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&outGrad_, batchSize_*inC_*sizeof(__half));
		if(useBias_){ CUDAMallocZero(&gradBiases_, outC_*sizeof(__half)); }
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
			if(useBias_){
				CUDAMallocZero(&m_Biases_, outC_*sizeof(__half));
				CUDAMallocZero(&v_Biases_, outC_*sizeof(__half));
			}
		}
	}
}
FCLayer::~FCLayer(){
	cudaFree(weights_);
	cudaFree(outData_);
	if(useBias_){ cudaFree(biases_); }
	if(train_){
		cudaFree(outGrad_);
		cudaFree(gradWeights_);
		if(useBias_){ cudaFree(gradBiases_); }
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
			if(useBias_){
				cudaFree(m_Biases_);
				cudaFree(v_Biases_);
			}
		}
	}
}
__half* FCLayer::Forward(__half* data){
	inData_ = data;
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, outC_, batchSize_, inC_, &alpha_, weights_, CUDA_R_16F, outC_, data, CUDA_R_16F, inC_, &beta0_, outData_, CUDA_R_16F, outC_, CUDA_R_32F));
	if(useBias_){ AddBias(outData_, biases_, outC_, batchSize_); }
	return outData_;
}
__half* FCLayer::Backward(__half* grad){
	const float* betaWeights = accumCount_++%gradAccumLength_==0 ? &beta0_ : &beta1_;
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, outC_, inC_, batchSize_, &alphaWeights_, grad, CUDA_R_16F, outC_, inData_, CUDA_R_16F, inC_, betaWeights, gradWeights_, CUDA_R_16F, outC_, CUDA_R_32F));
	if(useBias_){
		const bool reset = betaWeights == &beta0_;
		AccumulateBiasGrad(grad, gradBiases_, outC_, batchSize_, alphaWeights_, reset);
	}
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, inC_, batchSize_, outC_, &alpha_, weights_, CUDA_R_16F, outC_, grad, CUDA_R_16F, outC_, &beta0_, outGrad_, CUDA_R_16F, inC_, CUDA_R_32F));
	return outGrad_;
}
void FCLayer::UpdateParameters(const float learningRate){
	if(accumCount_%gradAccumLength_>0) return;
	if(useAdamW_){
		++t_;
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
		if(useBias_){ AdamWHalf(biases_, gradBiases_, m_Biases_, v_Biases_, learningRate, t_, 0.0f, outC_); }
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, learningRate, weightDecay_);
		if(useBias_){ SGDHalf(biases_, gradBiases_, outC_, learningRate, 0.0f); }
	}
}
void FCLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	if(useBias_){
		cudaMemcpy(buffer, biases_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
	}
}
void FCLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	if(useBias_){
		file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
		cudaMemcpy(biases_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
	}
}
void FCLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	if(useBias_){
		cudaMemcpy(buffer, m_Biases_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
		cudaMemcpy(buffer, v_Biases_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
	}
	file.write(reinterpret_cast<char*>(&t_), sizeof(int));
}
void FCLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	if(useBias_){
		file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
		cudaMemcpy(m_Biases_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
		cudaMemcpy(v_Biases_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
	}
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t FCLayer::GetParameterSize(){
	const size_t biasCount = useBias_ ? outC_ : 0;
	return std::max(weightCount_, biasCount)*sizeof(__half);
}
size_t FCLayer::GetOptimizerStateSize(){
	const size_t biasCount = useBias_ ? outC_ : 0;
	return std::max(weightCount_, biasCount)*sizeof(__half);
}
void FCLayer::SetTrain(const bool enable){
	train_ = enable;
}

void FCLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!useAdamW_ || !train_) return;
	halfTasks.push_back({weights_, gradWeights_, m_Weights_, v_Weights_, static_cast<int>(weightCount_), weightDecay_});
	if(useBias_){ halfTasks.push_back({biases_, gradBiases_, m_Biases_, v_Biases_, outC_, 0.0f}); }
}
