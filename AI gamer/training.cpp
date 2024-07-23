#include "training.h"
#include "ConvLayer.h"
#include "FCLayer.h"
#include "LeakyReLU.h"
#include "BatchNorm.h"
#include "Sigmoid.h"
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <atomic>
NeuralNetwork::NeuralNetwork(): batchSize_(32), inWidth_(0), inHeight_(0), learningRate_(0.0001f){
	cudnnCreate(&cudnn_);
	cublasCreate(&cublas_);
}
NeuralNetwork::~NeuralNetwork(){
	cudnnDestroy(cudnn_);
	cublasDestroy(cublas_);
	if(gradient_) cudaFree(gradient_);
	if(ctrlBatchFloat_) cuMemFreeHost(ctrlBatchFloat_);
	if(ctrlBatchHalf_) cuMemFreeHost(ctrlBatchHalf_);
}
void NeuralNetwork::Initialize(int w, int h){
	inWidth_ = w;
	inHeight_ = h;
	auto inputSize = w*h*3;
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 3, 32, 8, 4, 2, &w, &h, &inputSize, 0, 0, "Conv0")); // Input: RGB image
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv0 BatchNorm"));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv0 LeakyReLU"));
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 32, 64, 4, 2, 1, &w, &h, &inputSize, 0, 0, "Conv1"));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv1 BatchNorm"));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv1 LeakyReLU"));
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 64, 128, 4, 2, 1, &w, &h, &inputSize, 0, 0, "Conv2"));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv2 BatchNorm"));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv2 LeakyReLU"));
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 128, 128, 3, 1, 0, &w, &h, &inputSize, 0, 0, "Conv3"));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv3 BatchNorm"));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv3 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, inputSize, 512, "FC0"));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC0 BatchNorm"));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC0 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 512, 256, "FC1"));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC1 BatchNorm"));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC1 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256, numCtrls_, "FC2"));
	layers_.push_back(new Sigmoid(layers_.back()->outDesc_, 16, batchSize_, "FC2 Sigmoid"));
	checkCUDA(cudaMalloc(&gradient_, ctrlBatchSize_*sizeof(__half)));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchFloat_), ctrlBatchSize_*sizeof(float));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchHalf_), ctrlBatchSize_*sizeof(__half));
	cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);//S
}
__half* NeuralNetwork::Forward(__half* data, bool train){
	//printDataHalf(data, 10, "data");
	for(const auto layer : layers_){
		//std::cout << layer->layerName_ << "\r\n";
		data = layer->Forward(data);
		//PrintDataHalf(data, 8, "data");
	}
	return data;
}
void NeuralNetwork::Backward(const __half* d_predictions, const float* d_targets){
	//printDataFloat(d_targets, numCtrls, "Targets");
	Gradient(gradient_, d_predictions, d_targets, batchSize_, numCtrls_, 128.0f);
	auto outGrad = gradient_;
	for(int i = layers_.size(); --i >= 0; ){
		//PrintDataHalf(outGrad, 4, "Gradient");
		outGrad = layers_[i]->Backward(outGrad);
	}
}
void NeuralNetwork::UpdateParams(){
	for(const auto layer : layers_){ layer->UpdateParameters(learningRate_); }
}
void NeuralNetwork::Train(InputRecord** data, size_t count){
	std::random_device rd;
	std::mt19937 gen(rd());
	const std::uniform_int_distribution<> dis(0, count - 1);
	const size_t inputSize = inWidth_ * inHeight_ * 3;
	const size_t totalInputSize = inputSize * batchSize_;
	unsigned char* h_batchInputBytes = nullptr;
	unsigned char* d_batchInputBytes = nullptr;
	__half* d_batchInputHalf = nullptr;
	checkCUDA(cudaMalloc(&d_batchInputBytes, totalInputSize * sizeof(unsigned char)));
	checkCUDA(cudaMalloc(&d_batchInputHalf, totalInputSize * sizeof(__half)));
	checkCUDA(cudaMallocHost(&h_batchInputBytes, totalInputSize * sizeof(unsigned char)));
	float* h_ctrlBatch = nullptr;
	checkCUDA(cudaMallocHost(&h_ctrlBatch, ctrlBatchSize_ * sizeof(float)));
	float* d_ctrlBatch;
	checkCUDA(cudaMalloc(&d_ctrlBatch, ctrlBatchSize_ * sizeof(float)));
	std::atomic<bool> batchReady(false);
	std::atomic<bool> trainingComplete(false);
	auto prepareBatch = [&](){
		while(!trainingComplete){
			if(!batchReady){
				for(size_t i = 0; i < batchSize_; ++i){
					const int index = dis(gen);
					const InputRecord* record = data[index];
					memcpy(h_batchInputBytes + i * inputSize, record->state_data, inputSize * sizeof(unsigned char));
					for(int j = 0; j < numButs_; ++j){
						h_ctrlBatch[i*numCtrls_ + j] = static_cast<float>(record->keyStates>>j & 1);
					}
					h_ctrlBatch[i * numCtrls_ + 14] = record->mouseDeltaX / 16384.0f + 0.5f;
					h_ctrlBatch[i * numCtrls_ + 15] = record->mouseDeltaY / 16384.0f + 0.5f;
				}
				batchReady = true;
			}
		}
	};
	std::thread batchPreparationThread(prepareBatch);
	batchPreparationThread.detach();
	for(size_t epoch = 0; epoch < 1000; ++epoch){
		for(size_t batch = 0; batch < count / batchSize_; ++batch){
			while(!batchReady){
				std::this_thread::yield();
			}
			checkCUDA(cudaMemcpy(d_batchInputBytes, h_batchInputBytes, totalInputSize*sizeof(unsigned char), cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(d_ctrlBatch, h_ctrlBatch, numCtrls_*batchSize_*sizeof(float), cudaMemcpyHostToDevice));
			batchReady = false;
			ConvertAndNormalize(d_batchInputBytes, d_batchInputHalf, totalInputSize);
			const auto output = Forward(d_batchInputHalf, true);
			ClearScreen();
			PrintDataFloatHost(h_ctrlBatch, numCtrls_, "Targets");
			PrintDataHalf(output, numCtrls_, "Predicted");
			const float loss = MseLoss(output, d_ctrlBatch, batchSize_*numCtrls_);
			std::cout << std::setprecision(10) << "Loss: " << loss << "\r\n\r\n";
			Backward(output, d_ctrlBatch);
			UpdateParams();
		}
	}
	trainingComplete = true;
	cudaFreeHost(h_ctrlBatch);
	cudaFreeHost(h_batchInputBytes);
	cudaFree(d_ctrlBatch);
	cudaFree(d_batchInputBytes);
	cudaFree(d_batchInputHalf);
}