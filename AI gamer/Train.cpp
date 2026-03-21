#include "Train.h"
#include "NN.h"
#include "CuCommon.cuh"
#include <iostream>
#include <string>
Train::Train(){}
Train::~Train(){}
void Train::Allocate(const int batchSize, const int stateSize){
	CUDAMallocZero(&dStateBatchBytes, batchSize*stateSize*sizeof(unsigned char));
	CUDAMallocZero(&dStateBatchHalf, batchSize*stateSize*sizeof(__half));
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
	checkCUDA(cudaMemcpy(dStateBatchBytes, sb->stateData, nn->stateSize_*nn->batchSize_, cudaMemcpyHostToDevice));
	ConvertByteToHalf(dStateBatchBytes, dStateBatchHalf, nn->stateSize_*nn->batchSize_, true);
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
	constexpr int kTemporalSequenceLength = 4;
	int epochs = 10;
	std::cout << "How many epochs: ";
	std::cin >> epochs;
	std::cout << "\n";
	InitCUDA();
	cudnnContext* cudnn;
	cudnnCreate(&cudnn);
	const auto nn = new NN(cudnn, width, height, true);
	StateBatch sb0(nn->batchSize_, nn->stateSize_);
	StateBatch sb1(nn->batchSize_, nn->stateSize_);
	Allocate(nn->batchSize_, nn->stateSize_);
	bool stopTraining = false;
	const auto trainSequenceStartCount = GetSequenceStartCount(false, kTemporalSequenceLength);
	const auto valSequenceStartCount = GetSequenceStartCount(true, kTemporalSequenceLength);
	const auto epochBatchCount = std::max(1, static_cast<int>((trainSequenceStartCount + static_cast<size_t>(nn->batchSize_) - 1)/static_cast<size_t>(nn->batchSize_)));
	const auto epochBatchCountVal = std::max(1, static_cast<int>((valSequenceStartCount + static_cast<size_t>(nn->batchSize_) - 1)/static_cast<size_t>(nn->batchSize_)));
	auto primeSequenceBatch = [&](StateBatch* batch, std::vector<size_t>* sequenceStarts, const bool validation){
		if(!GetSequenceBatchStarts(sequenceStarts, nn->batchSize_, kTemporalSequenceLength, validation)){ return false; }
		ResetLoadBatchFailureCount();
		LoadSequenceBatchStep(batch, *sequenceStarts, 0, nn->stateSize_, validation);
		return true;
	};
	auto runSequencePhase = [&](const bool validation, const int phaseBatchCount, const auto& getBatchLr){
		std::vector<size_t> currentStarts;
		std::vector<size_t> nextStarts;
		ShuffleSequenceOrder(validation, kTemporalSequenceLength);
		auto* sbRead = &sb0;
		auto* sbPrefetch = &sb1;
		bool batchPrimed = primeSequenceBatch(sbRead, &currentStarts, validation);
		for(int batch = 0; batch < phaseBatchCount && !stopTraining; ++batch){
			if(!batchPrimed && !primeSequenceBatch(sbRead, &currentStarts, validation)){
				std::cerr << (validation ? "\nNo valid validation sequences available\n" : "\nNo valid training sequences available\n");
				stopTraining = true;
				break;
			}
			batchPrimed = false;
			nn->ResetState();
			const float lr = getBatchLr(batch);
			for(int step = 0; step < kTemporalSequenceLength && !stopTraining; ++step){
				threadPool.WaitAll();
				const auto loadBatchFailures = GetLoadBatchFailureCount();
				if(loadBatchFailures > 0){
					std::cerr << (validation ? "\nWarning: skipped validation sequence batch " : "\nWarning: skipped sequence batch ") << (batch + 1) << "/" << phaseBatchCount << " at step " << (step + 1) << " due to " << loadBatchFailures << " load failures\n";
					break;
				}
				bool prefetchedNextBatch = false;
				if(step + 1 < kTemporalSequenceLength){
					ResetLoadBatchFailureCount();
					LoadSequenceBatchStep(sbPrefetch, currentStarts, step + 1, nn->stateSize_, validation);
				} else if(batch + 1 < phaseBatchCount && GetSequenceBatchStarts(&nextStarts, nn->batchSize_, kTemporalSequenceLength, validation)){
					ResetLoadBatchFailureCount();
					LoadSequenceBatchStep(sbPrefetch, nextStarts, 0, nn->stateSize_, validation);
					prefetchedNextBatch = true;
				}
				const auto result = TrainBatch(nn, sbRead, true, lr, batch, phaseBatchCount);
				if(result == -1){ stopTraining = true; }
				if(step + 1 < kTemporalSequenceLength){
					std::swap(sbRead, sbPrefetch);
				} else if(prefetchedNextBatch){
					currentStarts.swap(nextStarts);
					std::swap(sbRead, sbPrefetch);
					batchPrimed = true;
				}
			}
		}
		return !stopTraining;
	};
	for(auto epoch = 0; epoch < epochs; ++epoch){
		emaLossButs_ = emaLossAxes_ = 0;
		std::cout << "\nEpoch: " << epoch << "\n";
		if(!runSequencePhase(false, epochBatchCount, [&](const int batch){ return GetLearningRate(epoch, batch, epochBatchCount, epochs); })){
			std::cout << "\nNaN encountered during training. Stopping.\n";
			break;
		}
		threadPool.WaitAll();
		nn->SaveModel(ckptFileName);
		nn->SaveOptimizerState(optFileName);
		emaLossButs_ = emaLossAxes_ = 0;
		std::cout << "\nRunning validation...\n";
		nn->SetTrain(false);
		runSequencePhase(true, epochBatchCountVal, [](const int){ return 0.0f; });
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
