#include "Train.h"
#include "NN.h"
#include "CuCommon.cuh"
#include <iostream>
#include <string>
#include <cstring>
Train::Train(){}
Train::~Train(){}
void Train::Allocate(const int batchSize, const int stateSize, const int framesPerSample){
	framesPerSample_ = framesPerSample;
	const size_t stackedStateSize = static_cast<size_t>(batchSize)*stateSize*framesPerSample_;
	checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&hStateBatchStackBytes), stackedStateSize*sizeof(unsigned char)));
	CUDAMallocZero(&dStateBatchBytes, stackedStateSize*sizeof(unsigned char));
	CUDAMallocZero(&dStateBatchHalf, stackedStateSize*sizeof(__half));
	checkCUDA(cudaMallocHost(&hTargetBatchFloat, batchSize*NUM_CTRLS_*sizeof(float)));
	CUDAMallocZero(&dTargetBatchFloat, batchSize*NUM_CTRLS_*sizeof(float));
	CUDAMallocZero(&dGradient_, batchSize*NUM_CTRLS_*sizeof(__half));
}
void Train::Free(){
	cudaFree(dGradient_);
	cudaFree(dTargetBatchFloat);
	cudaFreeHost(hTargetBatchFloat);
	cudaFree(dStateBatchHalf);
	cudaFree(dStateBatchBytes);
	cudaFreeHost(hStateBatchStackBytes);
}
float GetLearningRate(const int epoch, const int batch, const int epochBatchCount, const int epochs){
	constexpr auto baseLr = 0.00001f;
	constexpr auto minLr = 0.000001f;
	const auto warmupSteps = epochBatchCount;
	const auto totalSteps = epochBatchCount*epochs;
	const auto currentStep = epoch*epochBatchCount + batch;
	if(currentStep < warmupSteps){
		const auto lr = baseLr*static_cast<float>(currentStep)/static_cast<float>(warmupSteps);
		return lr < minLr ? minLr : lr;
	}
	const auto progress = static_cast<float>(currentStep - warmupSteps)/static_cast<float>(totalSteps - warmupSteps);
	const auto cosineDecay = 0.5f*(1.0f + cosf(3.14159f*progress));
	return minLr + (baseLr - minLr)*cosineDecay;
}
static void BuildRecentFrameStack(const unsigned char* sourceFrames, unsigned char* stackedFrames, int batchSize, int stateSize, int framesPerSample){
	for(int sample = 0; sample < batchSize; ++sample){
		for(int frame = 0; frame < framesPerSample; ++frame){
			const int sourceSample = sample - (framesPerSample - 1 - frame);
			const int clampedSource = sourceSample < 0 ? 0 : sourceSample;
			std::memcpy(stackedFrames + static_cast<size_t>(sample*framesPerSample + frame)*stateSize, sourceFrames + static_cast<size_t>(clampedSource)*stateSize, stateSize);
		}
	}
}
double GetRate(){
	static auto lastTime = std::chrono::high_resolution_clock::now();
	static int callCount = 0;
	static bool isFirstCall = true;
	const auto currentTime = std::chrono::high_resolution_clock::now();
	callCount++;
	const double elapsedTime = std::chrono::duration<double>(currentTime - lastTime).count();
	if(isFirstCall){
		isFirstCall = false;
		lastTime = currentTime;
		return 0.0;
	}
	if(elapsedTime > 0){
		const double rate = callCount / elapsedTime;
		callCount = 0;
		lastTime = currentTime;
		return rate;
	}
	return 0.0;
}
int Train::TrainBatch(NN* nn, const StateBatch* sb, const bool smoothLoss, const float lr, const int batchIndex, const int epochBatchCount){
	for(auto i = 0; i < nn->batchSize_; ++i){
		for(auto j = 0; j < NUM_BUTS_; ++j){ hTargetBatchFloat[i*NUM_CTRLS_ + j] = static_cast<float>(sb->inputStates[i].keyStates >> j & 1); }
		const auto axisBase = i*NUM_CTRLS_ + NUM_BUTS_;
		hTargetBatchFloat[axisBase] = std::asinh(static_cast<float>(sb->inputStates[i].deltaX)/AXIS_SCALE_);
		hTargetBatchFloat[axisBase + 1] = std::asinh(static_cast<float>(sb->inputStates[i].deltaY)/AXIS_SCALE_);
	}
	BuildRecentFrameStack(sb->stateData, hStateBatchStackBytes, nn->batchSize_, nn->inWidth_*nn->inHeight_*nn->channelsPerFrame_, nn->framesPerSample_);
	const size_t stackedStateSize = static_cast<size_t>(nn->stateSize_)*nn->batchSize_;
	checkCUDA(cudaMemcpy(dStateBatchBytes, hStateBatchStackBytes, stackedStateSize, cudaMemcpyHostToDevice));
	ConvertByteToHalf(dStateBatchBytes, dStateBatchHalf, stackedStateSize, true);
	checkCUDA(cudaMemcpy(dTargetBatchFloat, hTargetBatchFloat, NUM_CTRLS_*nn->batchSize_*sizeof(float), cudaMemcpyHostToDevice));
	const auto dPredictions = nn->Forward(dStateBatchHalf);
	if(IsnanHalf(dPredictions, NUM_CTRLS_*nn->batchSize_)){
		std::cout << " NaN in predictions\n";
		return -1;
	}
	LossStats(dPredictions, dTargetBatchFloat, NUM_BUTS_, NUM_CTRLS_, nn->batchSize_, &lossButs_, &lossAxes_);
	if(smoothLoss){
		constexpr float smoothing = 0.99f;
		emaLossButs_ = smoothing*emaLossButs_ + (1.0f - smoothing)*lossButs_;
		emaLossAxes_ = smoothing*emaLossAxes_ + (1.0f - smoothing)*lossAxes_;
	} else{
		emaLossButs_ = lossButs_;
		emaLossAxes_ = lossAxes_;
	}
	std::cout << "\rLR: " << lr << " Batch " << batchIndex + 1 << "/" << epochBatchCount << " Buts: " << emaLossButs_ << " Axes: " << emaLossAxes_ << " Batch rate: " << GetRate();
	if(lr <= 0.0f) return 0;
	LossBackprop(dGradient_, dPredictions, dTargetBatchFloat, 8.0f, NUM_CTRLS_*nn->batchSize_, NUM_CTRLS_, NUM_BUTS_, nn->batchSize_);
	if(IsnanHalf(nn->Backward(dGradient_), nn->stateSize_*nn->batchSize_)){
		std::cout << " NaN in gradient\n";
		return -1;
	}
	nn->UpdateParams(lr);
	return 0;
}
void Train::TrainModel(const int width, const int height){
	int epochs = 10;
	std::cout << "How many epochs: ";
	std::cin >> epochs;
	std::cout << "\n";
	InitCUDA();
	cudnnContext* cudnn;
	cudnnCreate(&cudnn);
	const auto nn = new NN(cudnn, width, height, true);
	const int frameStateSize = nn->inWidth_*nn->inHeight_*nn->channelsPerFrame_;
	StateBatch sb0(nn->batchSize_, frameStateSize);
	StateBatch sb1(nn->batchSize_, frameStateSize);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&](const bool validation){
		ResetLoadBatchFailureCount();
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch, validation, frameStateSize]{ LoadBatch(nextBatch, nn->batchSize_, frameStateSize, validation); });
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	Allocate(nn->batchSize_, nn->inWidth_*nn->inHeight_*nn->channelsPerFrame_, nn->framesPerSample_);
	bool stopTraining = false;
	const auto epochBatchCount = (trainRecordIndices.size() + nn->batchSize_ - 1)/nn->batchSize_;
	const auto epochBatchCountVal = (valRecordIndices.size() + nn->batchSize_ - 1)/nn->batchSize_;
	for(auto epoch = 0; epoch < epochs; ++epoch){
		ShuffleBatchOrder(false);
		fetchBatch(false);
		emaLossButs_ = emaLossAxes_ = 0;
		std::cout << "\nEpoch: " << epoch << "\n";
		for(auto batch = 0; batch < epochBatchCount && !stopTraining; ++batch){
			threadPool.WaitAll();
			const auto loadBatchFailures = GetLoadBatchFailureCount();
			fetchBatch(false);
			if(loadBatchFailures > 0){
				std::cerr << "\nWarning: skipped batch " << (batch + 1) << "/" << epochBatchCount << " due to " << loadBatchFailures << " load failures\n";
				continue;
			}
			const float lr = GetLearningRate(epoch, batch, epochBatchCount, epochs);
			const auto result = TrainBatch(nn, sbRead, true, lr, batch, epochBatchCount);
			if(result == -1){ stopTraining = true; }
		}
		if(stopTraining){
			std::cout << "\nNaN encountered during training. Stopping.\n";
			break;
		}
		threadPool.WaitAll();
		nn->SaveModel(ckptFileName);
		nn->SaveOptimizerState(optFileName);
		emaLossButs_ = emaLossAxes_ = 0;
		std::cout << "\nRunning validation...\n";
		nn->SetTrain(false);
		ShuffleBatchOrder(true);
		fetchBatch(true);
		for(auto batch = 0; batch < epochBatchCountVal && !stopTraining; ++batch){
			threadPool.WaitAll();
			fetchBatch(true);
			const auto result = TrainBatch(nn, sbRead, true, 0.0f, batch, epochBatchCountVal);
			if(result == -1){ stopTraining = true; }
		}
		nn->SetTrain(true);
		if(stopTraining){
			std::cout << "\nNaN encountered during validation. Stopping.\n";
			break;
		}
	}
	threadPool.WaitAll();
	Free();
	delete nn;
	cudnnDestroy(cudnn);
}
