#include "training.h"
#include "ConvLayer.h"
#include "FCLayer.h"
#include "LeakyReLU.h"
#include "BatchNorm.h"
#include "Sigmoid.h"
#include "NvDisplayCap.h"
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <atomic>
NeuralNetwork::NeuralNetwork(): cudnn_(nullptr), cublas_(nullptr), batchSize_(32), inWidth_(0), inHeight_(0), learningRate_(0.0001f), maxBufferSize_(0){}
NeuralNetwork::~NeuralNetwork(){
	cudnnDestroy(cudnn_);
	cublasDestroy(cublas_);
	if(gradient_) cudaFree(gradient_);
	if(ctrlBatchFloat_) cudaFreeHost(ctrlBatchFloat_);
	if(ctrlBatchHalf_) cudaFreeHost(ctrlBatchHalf_);
}
void NeuralNetwork::Initialize(int w, int h, bool train){
	if(!train) batchSize_ = 1;
	// Check if checkpoint file exists
	std::ifstream ckptFile(ckptFileName, std::ios::binary);
	const bool fileOpen = ckptFile.is_open();
	int modelWidth = w;
	int modelHeight = h;
	if(fileOpen){
		std::cout << "Checkpoint file found...\r\n";
		ckptFile.read(reinterpret_cast<char*>(&modelWidth), sizeof(int));
		ckptFile.read(reinterpret_cast<char*>(&modelHeight), sizeof(int));
	}
	if(train){
		if(fileOpen && (w != modelWidth || h != modelHeight)){
			std::cerr << "Error: Checkpoint resolution does not match training data resolution.\r\n";
			ckptFile.close();
			return;
		}
	}
	inWidth_ = modelWidth;
	inHeight_ = modelHeight;
	auto inputSize = modelWidth*modelHeight*3;
	InitCUDA();
	cudnnCreate(&cudnn_);
	cublasCreate(&cublas_);
	std::cout << "Initializing layers...\r\n";
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 3, 32, 8, 4, 2, &modelWidth, &modelHeight, &inputSize, "Conv0", train)); // Input: RGB image
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv0 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv0 LeakyReLU"));
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 32, 64, 4, 2, 1, &modelWidth, &modelHeight, &inputSize, "Conv1", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv1 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv1 LeakyReLU"));
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 64, 128, 4, 2, 1, &modelWidth, &modelHeight, &inputSize, "Conv2", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv2 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv2 LeakyReLU"));
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 128, 128, 3, 1, 0, &modelWidth, &modelHeight, &inputSize, "Conv3", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv3 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv3 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, inputSize, 512, "FC0", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC0 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC0 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 512, 256, "FC1", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC1 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC1 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256, numCtrls_, "FC2", train));
	layers_.push_back(new Sigmoid(layers_.back()->outDesc_, 16, batchSize_, "FC2 Sigmoid"));
	// Initialize memory for gradients and batch control
	checkCUDA(cudaMalloc(&gradient_, ctrlBatchSize_*sizeof(__half)));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchFloat_), ctrlBatchSize_*sizeof(float));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchHalf_), ctrlBatchSize_*sizeof(__half));
	cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);
	if(fileOpen){
		// Determine the maximum buffer size needed for loading parameters and optimizer state
		for(const auto& layer : layers_){
			if(layer->HasParameters()){
				maxBufferSize_ = max(maxBufferSize_, layer->GetParameterSize());
			}
			if(layer->HasOptimizerState()){
				maxBufferSize_ = max(maxBufferSize_, layer->GetOptimizerStateSize());
			}
		}
		std::cout << "Loading weights/bias...\r\n";
		float* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){
			if(layer->HasParameters()){
				layer->LoadParameters(ckptFile, buffer);
			}
		}
		ckptFile.close();
		if(train){
			std::cout << "Loading optimizer state...\r\n";
			std::ifstream optFile(optFileName, std::ios::binary);
			if(optFile.is_open()){
				for(const auto& layer : layers_){
					if(layer->HasOptimizerState()){
						layer->LoadOptimizerState(optFile, buffer);
					}
				}
				optFile.close();
			} else{
				std::cerr << "No optimizer state file: " << optFileName << "\r\n";
			}
		}
		cudaFreeHost(buffer);
	}
	std::cout << "Done.\r\n";
}
__half* NeuralNetwork::Forward(__half* data, bool train){
	for(const auto layer : layers_){
		//std::cout << layer->layerName_ << "\r\n";
		data = layer->Forward(data, train);
		//PrintDataHalf(data, 8, "data");
	}
	return data;
}
void NeuralNetwork::Backward(const __half* d_predictions, const float* d_targets){
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
void NeuralNetwork::SaveModel(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		float* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		file.write(reinterpret_cast<const char*>(&inWidth_), sizeof(inWidth_));
		file.write(reinterpret_cast<const char*>(&inHeight_), sizeof(inHeight_));
		for(const auto& layer : layers_){
			if(layer->HasParameters()){
				layer->SaveParameters(file, buffer);
			}
		}
		cudaFreeHost(buffer);
		file.close();
	} else{
		std::cerr << "Unable to open file for saving checkpoint: " << filename << "\r\n";
	}
}
void NeuralNetwork::SaveOptimizerState(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		float* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){
			if(layer->HasOptimizerState()){
				layer->SaveOptimizerState(file, buffer);
			}
		}
		cudaFreeHost(buffer);
		file.close();
	} else{
		std::cerr << "Unable to open file for saving optimizer state: " << filename << "\r\n";
	}
}
void NeuralNetwork::Train(InputRecord** data, size_t count, int epochs){
	std::random_device rd;
	std::mt19937 gen(rd());
	const std::uniform_int_distribution<> dis(0, count - 1);
	const size_t inputSize = inWidth_*inHeight_*3;
	const size_t totalInputSize = inputSize*batchSize_;
	unsigned char* h_batchInputBytes = nullptr;
	unsigned char* d_batchInputBytes = nullptr;
	__half* d_batchInputHalf = nullptr;
	checkCUDA(cudaMalloc(&d_batchInputBytes, totalInputSize*sizeof(unsigned char)));
	checkCUDA(cudaMalloc(&d_batchInputHalf, totalInputSize*sizeof(__half)));
	checkCUDA(cudaMallocHost(&h_batchInputBytes, totalInputSize*sizeof(unsigned char)));
	float* h_ctrlBatch = nullptr;
	checkCUDA(cudaMallocHost(&h_ctrlBatch, ctrlBatchSize_*sizeof(float)));
	float* d_ctrlBatch;
	checkCUDA(cudaMalloc(&d_ctrlBatch, ctrlBatchSize_*sizeof(float)));
	std::atomic<bool> batchReady(false);
	std::atomic<bool> trainingComplete(false);
	auto prepareBatch = [&](){
		while(!trainingComplete){
			if(!batchReady){
				for(size_t i = 0; i < batchSize_; ++i){
					const int index = dis(gen);
					const InputRecord* record = data[index];
					memcpy(h_batchInputBytes + i*inputSize, record->state_data, inputSize*sizeof(unsigned char));
					for(int j = 0; j < numButs_; ++j){
						h_ctrlBatch[i*numCtrls_ + j] = static_cast<float>(record->keyStates>>j & 1);
					}
					h_ctrlBatch[i*numCtrls_ + 14] = record->mouseDeltaX / 16384.0f + 0.5f;
					h_ctrlBatch[i*numCtrls_ + 15] = record->mouseDeltaY / 16384.0f + 0.5f;
				}
				batchReady = true;
			}
		}
	};
	std::thread batchPreparationThread(prepareBatch);
	batchPreparationThread.detach();
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		ClearScreen();
		std::cout << "Epoch: " << epoch << "\r\n";
		for(size_t batch = 0; batch < count / batchSize_; ++batch){
			while(!batchReady){
				std::this_thread::yield();
			}
			checkCUDA(cudaMemcpy(d_batchInputBytes, h_batchInputBytes, totalInputSize*sizeof(unsigned char), cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(d_ctrlBatch, h_ctrlBatch, numCtrls_*batchSize_*sizeof(float), cudaMemcpyHostToDevice));
			batchReady = false;
			ConvertAndNormalize(d_batchInputHalf, d_batchInputBytes, totalInputSize);
			const auto output = Forward(d_batchInputHalf, true);
			//ClearScreen();
			//PrintDataFloatHost(h_ctrlBatch, numCtrls_, "Targets");
			//PrintDataHalf(output, numCtrls_, "Predicted");
			const float loss = MseLoss(output, d_ctrlBatch, batchSize_*numCtrls_);
			std::cout << std::setprecision(10) << "Loss: " << loss << "\t\t\t\t\r";
			Backward(output, d_ctrlBatch);
			UpdateParams();
		}
	}
	trainingComplete = true;
	SaveModel(ckptFileName);
	SaveOptimizerState(optFileName);
	cudaFreeHost(h_ctrlBatch);
	cudaFreeHost(h_batchInputBytes);
	cudaFree(d_ctrlBatch);
	cudaFree(d_batchInputBytes);
	cudaFree(d_batchInputHalf);
}
void NeuralNetwork::ProcessOutput(const float* output){
	for(int i = 0; i < 14; ++i){
		if(output[i] > 0.5){
			keybd_event(keyMap[i], 0, 0, 0);
		} else{
			keybd_event(keyMap[i], 0, KEYEVENTF_KEYUP, 0);
		}
	}
	const int mouseX = static_cast<int>((output[14] - 0.5f)*16384.0f);
	const int mouseY = static_cast<int>((output[15] - 0.5f)*16384.0f);
	INPUT input = {0};
	input.type = INPUT_MOUSE;
	input.mi.dx = mouseX;
	input.mi.dy = mouseY;
	input.mi.dwFlags = MOUSEEVENTF_MOVE;
	SendInput(1, &input, sizeof(INPUT));
}
void NeuralNetwork::ListenForKey(){
	std::cout << "Press F9 to start AI input and Escape to pause.\r\n";
	while(true){
		if(GetAsyncKeyState(VK_F9) & 0x8000){
			simInput = true;
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			simInput = false;
			while(GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void NeuralNetwork::InferLoop(){
	Initialize(0, 0, false);
	InitNvFBC();
	AllocGPU();
	int capWidth = 0, capHeight = 0;
	float* h_predictionsF;
	checkCUDA(cudaMallocHost(&h_predictionsF, numCtrls_*sizeof(float)));
	float* d_predictionsF;
	checkCUDA(cudaMalloc(&d_predictionsF, numCtrls_*sizeof(float)));
	const auto inputSize = inWidth_*inHeight_*3;
	__half* frameHalf = nullptr;
	checkCUDA(cudaMalloc(&frameHalf, inputSize*sizeof(__half)));
	while(!stopInfer){
		while(!simInput){
			Sleep(1);
		}
		const auto frame = GrabFrame(&capWidth, &capHeight, true, false);
		if(capWidth != inWidth_ || capHeight != inHeight_){
			simInput = false;
			std::cerr << "Capture resolution mismatch, pausing AI input.\r\n";
			//continue;
		}
		ConvertAndNormalize(frameHalf, frame, inputSize);
		checkCUDA(cudaDeviceSynchronize());
		//PrintDataHalf(frameHalf, 100, "frameHalf");
		const auto output = Forward(frameHalf, false);
		ConvertHalfToFloat(output, d_predictionsF, numCtrls_);
		checkCUDA(cudaMemcpy(h_predictionsF, d_predictionsF, numCtrls_*sizeof(float), cudaMemcpyDeviceToHost));
		ProcessOutput(h_predictionsF);
	}
	DisposeNvFBC();
	cudaFreeHost(h_predictionsF);
	cudaFree(d_predictionsF);
	cudaFree(frameHalf);
}
void NeuralNetwork::Infer(){
	//std::thread inferThread(&NeuralNetwork::inferLoop, this);
	//inferThread.detach();
	std::thread listenKey(&NeuralNetwork::ListenForKey, this);
	listenKey.detach();
	InferLoop();
	MSG msg = {};
	while(GetMessage(&msg, nullptr, 0, 0)){
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	stopInfer = true;
}