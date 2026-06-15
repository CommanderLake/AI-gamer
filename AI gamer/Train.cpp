#include "Train.h"
#include "NN.h"
#include "APICommon.h"
#include "HostCommon.h"
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
		//hTargetBatchFloat[axisBase] = CompressAxisDelta(static_cast<float>(sb->inputStates[i].deltaX));
		//hTargetBatchFloat[axisBase + 1] = CompressAxisDelta(static_cast<float>(sb->inputStates[i].deltaY));
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
	LossBackprop(dGradient_, dPredictions, dTargetBatchFloat, 4.0f, NUM_CTRLS_*nn->batchSize_, NUM_CTRLS_, NUM_BUTS_, nn->batchSize_);
	const auto result = IsnanHalf(nn->Backward(dGradient_), nn->stateSize_*nn->batchSize_);
	SummaryPrint();
	if(result){
		std::cout << " NaN in gradient\n";
		return -1;
	}
	nn->UpdateParams(lr);
	return 0;
}
void Train::TrainModel(const int width, const int height, const bool validate){
	int epochs = 10;
	std::cout << "How many epochs: ";
	std::cin >> epochs;
	std::cout << "\n";
	InitCUDA();
	const auto nn = new NN(width, height, true);
	if(trainRecordIndices.empty()){
		delete nn;
		throw std::runtime_error("No training records were loaded");
	}
	if(validate && valRecordIndices.empty()){
		delete nn;
		throw std::runtime_error("Validation was requested but no validation records were loaded");
	}
	StateBatch sb0(nn->batchSize_, nn->stateSize_);
	StateBatch sb1(nn->batchSize_, nn->stateSize_);
	auto queueBatch = [&](StateBatch* batch, const bool validation){
		ResetLoadBatchFailureCount();
		LoadBatch(batch, nn->batchSize_, nn->stateSize_, validation);
	};
	Allocate(nn->batchSize_, nn->stateSize_);
	bool stopTraining = false;
	const auto epochBatchCount = (trainRecordIndices.size() + nn->batchSize_ - 1)/nn->batchSize_;
	const auto epochBatchCountVal = (valRecordIndices.size() + nn->batchSize_ - 1)/nn->batchSize_;
	auto runBatches = [&](const bool validation, const size_t batchCount, const int epoch){
		if(batchCount == 0){ return true; }
		StateBatch* currentBatch = &sb0;
		StateBatch* nextBatch = &sb1;
		queueBatch(currentBatch, validation);
		for(size_t batch = 0; batch < batchCount; ++batch){
			threadPool.WaitAll();
			const auto loadBatchFailures = GetLoadBatchFailureCount();
			const bool hasNextBatch = batch + 1 < batchCount;
			if(hasNextBatch){ queueBatch(nextBatch, validation); }
			if(loadBatchFailures > 0){
				std::cerr << "\nWarning: skipped batch " << (batch + 1) << "/" << batchCount << " due to " << loadBatchFailures << " load failures\n";
			} else{
				const float lr = validation ? 0.0f : GetLearningRate(epoch, static_cast<int>(batch), static_cast<int>(batchCount), epochs);
				const auto result = TrainBatch(nn, currentBatch, true, lr, static_cast<int>(batch), static_cast<int>(batchCount));
				if(result == -1){
					threadPool.WaitAll();
					return false;
				}
			}
			if(hasNextBatch){ std::swap(currentBatch, nextBatch); }
		}
		return true;
	};
	for(auto epoch = 0; epoch < epochs; ++epoch){
		ShuffleBatchOrder(false);
		emaLossButs_ = emaLossAxes_ = 0;
		std::cout << "\nEpoch: " << epoch << "\n";
		stopTraining = !runBatches(false, epochBatchCount, epoch);
		if(stopTraining){
			std::cout << "\nNaN encountered during training. Stopping.\n";
			break;
		}
		threadPool.WaitAll();
		nn->SaveModel(ckptFileName);
		nn->SaveOptimizerState(optFileName);
		if(!validate) continue;
		emaLossButs_ = emaLossAxes_ = 0;
		std::cout << "\nRunning validation...\n";
		nn->SetTrain(false);
		ShuffleBatchOrder(true);
		stopTraining = !runBatches(true, epochBatchCountVal, epoch);
		nn->SetTrain(true);
		if(stopTraining){
			std::cout << "\nNaN encountered during validation. Stopping.\n";
			break;
		}
	}
	threadPool.WaitAll();
	Free();
	delete nn;
}
