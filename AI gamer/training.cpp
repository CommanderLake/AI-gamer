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
NeuralNetwork::NeuralNetwork(): cudnn_(nullptr), cublas_(nullptr), batchSize_(32), inWidth_(0), inHeight_(0), learningRate_(0.00001f), maxBufferSize_(0){}
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
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, inputSize, 1024, "FC0", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC0 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC0 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 1024, 512, "FC1", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC1 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC1 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 512, 256, "FC2", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC2 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC2 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256, 128, "FC2", train));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC2 BatchNorm", train));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC2 LeakyReLU"));
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 128, numCtrls_, "FC out", train));
	layers_.push_back(new Sigmoid(layers_.back()->outDesc_, numButs_, batchSize_, "FC out Sigmoid"));
	// Initialize memory for gradients and batch control
	checkCUDA(cudaMalloc(&gradient_, ctrlBatchSize_*sizeof(__half)));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchFloat_), ctrlBatchSize_*sizeof(float));
	cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchHalf_), ctrlBatchSize_*sizeof(__half));
	cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);
	// Determine the maximum buffer size needed for loading parameters and optimizer state
	for(const auto& layer : layers_){
		if(layer->HasParameters()){
			maxBufferSize_ = max(maxBufferSize_, layer->GetParameterSize());
		}
		if(layer->HasOptimizerState()){
			maxBufferSize_ = max(maxBufferSize_, layer->GetOptimizerStateSize());
		}
	}
	if(fileOpen){
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
	Gradient(gradient_, d_predictions, d_targets, batchSize_, numCtrls_, 512.0f);
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
void NeuralNetwork::Train(size_t count){
	int epochs = 10;
	std::cout << "How many epochs: ";
	std::cin >> epochs;
	StateBatch sb0(batchSize_, stateSize);
	StateBatch sb1(batchSize_, stateSize);
	const StateBatch* sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&]{
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([nextBatch]{
			LoadBatch(nextBatch);
		});
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch();
	unsigned char* dBatchInputBytes = nullptr;
	__half* dBatchInputHalf = nullptr;
	float* hCtrlBatch = nullptr;
	float* dCtrlBatch = nullptr;
	checkCUDA(cudaMalloc(&dBatchInputBytes, stateSize*batchSize_*sizeof(unsigned char)));
	checkCUDA(cudaMalloc(&dBatchInputHalf, stateSize*batchSize_*sizeof(__half)));
	checkCUDA(cudaMallocHost(&hCtrlBatch, batchSize_*numCtrls_*sizeof(float)));
	checkCUDA(cudaMalloc(&dCtrlBatch, numCtrls_*batchSize_*sizeof(float)));
	std::cout << std::fixed << std::setprecision(8);
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		ClearScreen();
		std::cout << "Epoch: " << epoch << "\r\n";
		for(size_t batch = 0; batch < count/batchSize_; ++batch){
			threadPool.WaitAll();
			fetchBatch();
			for(size_t i = 0; i < batchSize_; ++i){
				for(int j = 0; j < numButs_; ++j){
					hCtrlBatch[i*numCtrls_ + j] = static_cast<float>(sbRead->keyStates[i] >> j & 1);
				}
				hCtrlBatch[i*numCtrls_ + 14] = static_cast<float>(sbRead->mouseDeltaX[i])/128.0f;
				hCtrlBatch[i*numCtrls_ + 15] = static_cast<float>(sbRead->mouseDeltaY[i])/128.0f;
			}
			checkCUDA(cudaMemcpy(dBatchInputBytes, sbRead->stateData, stateSize*batchSize_*sizeof(unsigned char), cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(dCtrlBatch, hCtrlBatch, numCtrls_*batchSize_*sizeof(float), cudaMemcpyHostToDevice));
			ConvertAndNormalize(dBatchInputHalf, dBatchInputBytes, batchSize_*stateSize);
			//PrintDataHalf(dBatchInputHalf, 16, "Batch input");
			const auto output = Forward(dBatchInputHalf, true);
			const float loss = MseLoss(output, dCtrlBatch, numCtrls_*batchSize_);
			std::cout << "Loss: " << loss << "\t\t\t\r";
			Backward(output, dCtrlBatch);
			UpdateParams();
		}
	}
	SaveModel(ckptFileName);
	SaveOptimizerState(optFileName);
	checkCUDA(cudaFree(dBatchInputBytes));
	checkCUDA(cudaFree(dBatchInputHalf));
	checkCUDA(cudaFreeHost(hCtrlBatch));
	checkCUDA(cudaFree(dCtrlBatch));
}
void NeuralNetwork::ProcessOutput(const float* output){
	INPUT inputs[numCtrls_] = {};
	int inputIndex = 0;
	// Simulate letter key inputs using scancodes (W, A, S, D, etc.)
	for(int i = 0; i < 11; ++i){
		inputs[inputIndex].type = INPUT_KEYBOARD;
		inputs[inputIndex].ki.wScan = keyMap[i];
		inputs[inputIndex].ki.dwFlags = KEYEVENTF_SCANCODE;
		if(output[i] > 0.5){
			inputs[inputIndex].ki.dwFlags |= 0;  // Key press
		} else{
			inputs[inputIndex].ki.dwFlags |= KEYEVENTF_KEYUP;  // Key release
		}
		inputIndex++;
	}
	// Handle mouse buttons
	if(output[11] > 0.5){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_LEFTUP;
		inputIndex++;
	}
	if(output[12] > 0.5){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
		inputIndex++;
	}
	if(output[13] > 0.5){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
		inputIndex++;
	}
	// Handle mouse movement
	const int mouseX = static_cast<int>(output[14]*128.0f);
	const int mouseY = static_cast<int>(output[15]*128.0f);
	inputs[inputIndex].type = INPUT_MOUSE;
	inputs[inputIndex].mi.dx = mouseX;
	inputs[inputIndex].mi.dy = mouseY;
	inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MOVE;
	// Send all inputs at once
	SendInput(inputIndex + 1, inputs, sizeof(INPUT));
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
void NeuralNetwork::Infer(){
	InitNvFBC();
	AllocGPU();
	Initialize(0, 0, false);
	std::thread listenKey(&NeuralNetwork::ListenForKey, this);
	listenKey.detach();
	while(!simInput){
		Sleep(1);
	}
	int capWidth = 0, capHeight = 0;
	float* h_predictionsF;
	checkCUDA(cudaMallocHost(&h_predictionsF, numCtrls_*sizeof(float)));
	float* d_predictionsF;
	checkCUDA(cudaMalloc(&d_predictionsF, numCtrls_*sizeof(float)));
	const auto inputSize = inWidth_*inHeight_*3;
	__half* frameHalf = nullptr;
	checkCUDA(cudaMalloc(&frameHalf, inputSize*sizeof(__half)));
	checkCUDA(cudaMemset(frameHalf, 0, inputSize*sizeof(__half)));
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	while(!stopInfer){
		while(!simInput){
			Sleep(1);
		}
		nextFrameTime += frameDuration;
		std::this_thread::sleep_until(nextFrameTime);
		const auto frame = GrabFrame(&capWidth, &capHeight, true, false);
		if(capWidth != inWidth_ || capHeight != inHeight_){
			simInput = false;
			std::cerr << "Capture resolution mismatch, pausing AI input.\r\n";
			continue;
		}
		ConvertAndNormalize(frameHalf, frame, inputSize);
		checkCUDA(cudaDeviceSynchronize());
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