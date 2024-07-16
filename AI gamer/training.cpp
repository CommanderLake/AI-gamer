#include "training.h"
#include "ConvLayer.h"
#include "FCLayer.h"
#include "BatchNorm.h"
#include "LeakyReLU.h"
#include "ScaleGrad.h"
#include <chrono>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
NeuralNetwork::NeuralNetwork(): inWidth(0), inHeight(0){
	cudnnCreate(&cudnn);
	cublasCreate(&cublas);
}
NeuralNetwork::~NeuralNetwork(){
	cudnnDestroy(cudnn);
	cublasDestroy(cublas);
	// Free preallocated memory
	if(gradient_) cudaFree(gradient_);
	if(ctrlBatchFloat) cuMemFreeHost(ctrlBatchFloat);
	if(ctrlBatchHalf) cuMemFreeHost(ctrlBatchHalf);
}
void NeuralNetwork::initialize(int w, int h){
	inWidth = w;
	inHeight = h;
	auto inputSize = w*h*3;
	layers.push_back(new ConvLayer(cudnn, cublas, batchSize, 3, 16, 8, 4, 2, &w, &h, &inputSize, 0, 0, "Conv0")); // Input: RGB image
	layers.push_back(new BatchNorm(cudnn, layers.back()->outDesc_, batchSize));
	layers.push_back(new LeakyReLU(layers.back()->outDesc_));
	layers.push_back(new ConvLayer(cudnn, cublas, batchSize, 32, 64, 4, 2, 1, &w, &h, &inputSize, 0, 0, "Conv1"));
	layers.push_back(new BatchNorm(cudnn, layers.back()->outDesc_, batchSize));
	layers.push_back(new LeakyReLU(layers.back()->outDesc_));
	layers.push_back(new ConvLayer(cudnn, cublas, batchSize, 64, 64, 3, 1, 1, &w, &h, &inputSize, 0, 0, "Conv2"));
	layers.push_back(new BatchNorm(cudnn, layers.back()->outDesc_, batchSize));
	layers.push_back(new LeakyReLU(layers.back()->outDesc_));
	layers.push_back(new FCLayer(cudnn, cublas, batchSize, inputSize, 256, "FC0"));
	layers.push_back(new BatchNorm(cudnn, layers.back()->outDesc_, batchSize));
	layers.push_back(new LeakyReLU(layers.back()->outDesc_));
	layers.push_back(new FCLayer(cudnn, cublas, batchSize, 256, 16, "FC1"));
	layers.push_back(new Activate(cudnn, layers.back()->outDesc_, CUDNN_ACTIVATION_TANH, 1.0));
	checkCUDA(cudaMalloc(&gradient_, ctrlBatchSize*sizeof(__half)));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchFloat), ctrlBatchSize*sizeof(float));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchHalf), ctrlBatchSize*sizeof(__half));
	cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
}
__half* NeuralNetwork::forward(__half* data, bool train){
	for(const auto layer : layers){
		data = layer->forward(data);
	}
	printDataHalf(data, numCtrls, "Predictions");
	return data;
}
void NeuralNetwork::backward(const __half* d_predictions, const float* d_targets){
	printDataFloat(d_targets, numCtrls, "Targets");
	gradient(gradient_, d_predictions, d_targets, batchSize, numCtrls);
	auto outGrad = gradient_;
	for(int i = layers.size(); --i >= 0; ){
		outGrad = layers[i]->backward(outGrad);
	}
}
void NeuralNetwork::updateParams(){
	for(const auto layer : layers){ layer->updateParameters(learningRate); }
}
void NeuralNetwork::train(InputRecord** data, size_t count){
	std::random_device rd;
	std::mt19937 gen(rd());
	const std::uniform_int_distribution<> dis(0, count - 1);
	const size_t inputSize = inWidth*inHeight*3;
	const size_t totalInputSize = inputSize*batchSize;
	unsigned char* d_batchInputBytes = nullptr;
	__half* d_batchInputHalf = nullptr;
	checkCUDA(cudaMalloc(&d_batchInputBytes, totalInputSize*sizeof(unsigned char)));
	checkCUDA(cudaMalloc(&d_batchInputHalf, totalInputSize*sizeof(__half)));
	float* h_ctrlBatch = nullptr;
	cudaMallocHost(&h_ctrlBatch, ctrlBatchSize*sizeof(float));
	float* d_ctrlBatch;
	checkCUDA(cudaMalloc(&d_ctrlBatch, ctrlBatchSize*sizeof(float)));
	//int max = 0;
	for(size_t epoch = 0; epoch < 1000; ++epoch){
		// Number of epochs
		for(size_t batch = 0; batch < count / batchSize; ++batch){
			for(size_t i = 0; i < batchSize; ++i){
				const int index = dis(gen);
				const InputRecord* record = data[index];
				checkCUDA(cudaMemcpy(d_batchInputBytes + i*inputSize, record->state_data, inputSize*sizeof(unsigned char), cudaMemcpyHostToDevice));
				for(const auto& [makeCode, bit] : keyMap){
					if(record->keyStates & 1 << bit){
						h_ctrlBatch[i*numCtrls + bit] = 1.0f;
					} else h_ctrlBatch[i*numCtrls + bit] = 0.0f;
				}
				//max = std::max(max, std::max(std::abs(record->mouseDeltaX), std::abs(record->mouseDeltaY)));
				h_ctrlBatch[i*numCtrls + 14] = record->mouseDeltaX/16384.0f;
				h_ctrlBatch[i*numCtrls + 15] = record->mouseDeltaY/16384.0f;
			}
			checkCUDA(cudaMemcpy(d_ctrlBatch, h_ctrlBatch, numCtrls*batchSize*sizeof(float), cudaMemcpyHostToDevice));
			// Launch kernel to convert and normalize input
			convertAndNormalize(d_batchInputBytes, d_batchInputHalf, totalInputSize);
			// Perform forward pass
			clear_screen();
			const auto output = forward(d_batchInputHalf, true);
			//printDataFloat(d_ctrlBatch, numCtrls, "Target");
			//printDataHalf(output, numCtrls, "Predicted");
			// Compute loss
			const float loss = mseLoss(output, d_ctrlBatch, batchSize*numCtrls);
			std::cout << std::setprecision(10) << "Loss: " << loss << "\r\n\r\n";
			//std::cout << "Max: " << max << "\r\n\r\n";
			// Perform backward pass
			backward(output, d_ctrlBatch);
			updateParams();
			//std::cout << "\r\n\r\n";
		}
	}
	cudaFreeHost(h_ctrlBatch);
	cudaFree(d_ctrlBatch);
	cudaFree(d_batchInputBytes);
	cudaFree(d_batchInputHalf);
}