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
float GetLearningRate(size_t epoch, size_t batch, size_t epochBatchCount){
	constexpr float baseLr = 0.00002f;
	constexpr float minLr = 0.0000001f;
	const size_t warmupSteps = epochBatchCount*1;
	const size_t totalSteps = epochBatchCount*10;
	const size_t currentStep = epoch*epochBatchCount + batch;
	if(currentStep < warmupSteps){ return baseLr*static_cast<float>(currentStep) / static_cast<float>(warmupSteps); }
	const float progress = static_cast<float>(currentStep - warmupSteps) / static_cast<float>(totalSteps - warmupSteps);
	const float cosineDecay = 0.5f*(1.0f + cosf(3.14159f*progress));
	return minLr + (baseLr - minLr)*cosineDecay;
}
int Train::TrainBatch(NN* nn, const StateBatch* sb, const bool smoothLoss, const float lr, const std::size_t batchIndex, const std::size_t epochBatchCount){
	for(size_t i = 0; i < nn->batchSize_; ++i){
		for(int j = 0; j < NUM_BUTS_; ++j){ hTargetBatchFloat[i*NUM_CTRLS_ + j] = static_cast<float>(sb->inputStates[i].keyStates >> j & 1); }
		hTargetBatchFloat[i*NUM_CTRLS_ + 14] = std::asinh(static_cast<float>(sb->inputStates[i].deltaX) / AXIS_SCALE_);
		hTargetBatchFloat[i*NUM_CTRLS_ + 15] = std::asinh(static_cast<float>(sb->inputStates[i].deltaY) / AXIS_SCALE_);
	}
	checkCUDA(cudaMemcpy(dStateBatchBytes, sb->stateData, nn->stateSize_*nn->batchSize_, cudaMemcpyHostToDevice));
	ConvertByteToHalf(dStateBatchBytes, dStateBatchHalf, nn->stateSize_*nn->batchSize_, true);
	checkCUDA(cudaMemcpy(dTargetBatchFloat, hTargetBatchFloat, NUM_CTRLS_*nn->batchSize_*sizeof(float), cudaMemcpyHostToDevice));
	const auto dPredictions = nn->Forward(dStateBatchHalf);
	if(IsnanHalf(dPredictions, NUM_CTRLS_*nn->batchSize_)){
		std::cout << " NaN in predictions\n";
		return -1;
	}
	Loss2(dPredictions, dTargetBatchFloat, NUM_BUTS_, NUM_CTRLS_, nn->batchSize_, &lossButs_, &lossAxes_);
	if(smoothLoss){
		constexpr float smoothing = 0.98f;
		emaLossButs_ = smoothing*emaLossButs_ + (1.0f - smoothing)*lossButs_;
		emaLossAxes_ = smoothing*emaLossAxes_ + (1.0f - smoothing)*lossAxes_;
	} else{
		emaLossButs_ = lossButs_;
		emaLossAxes_ = lossAxes_;
	}
	std::cout << "\rLR: " << lr << " Batch " << (batchIndex + 1) << "/" << epochBatchCount << " Buts: " << emaLossButs_ << " Axes: " << emaLossAxes_;
	if(lr == 0.0f) return 0;
	SplitGradient(dGradient_, dPredictions, dTargetBatchFloat, 32.0f, NUM_CTRLS_*nn->batchSize_, NUM_CTRLS_, NUM_BUTS_, nn->batchSize_);
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
	cublasContext* cublas;
	cudnnCreate(&cudnn);
	cublasCreate(&cublas);
	cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH); //S
	const auto nn = new NN(cudnn, cublas, width, height, true);
	StateBatch sb0(nn->batchSize_, nn->stateSize_);
	StateBatch sb1(nn->batchSize_, nn->stateSize_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&](const bool validation){
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch, validation]{ LoadBatch(nextBatch, nn->batchSize_, nn->stateSize_, validation); });
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch(false);
	Allocate(nn->batchSize_, nn->stateSize_);
	bool stopTraining = false;
	const auto epochBatchCount = trainRecordIndices.size() / nn->batchSize_;
	const auto epochBatchCountVal = valRecordIndices.size() / nn->batchSize_;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		emaLossButs_ = emaLossAxes_ = 0;
		std::cout << "\nEpoch: " << epoch << "\n";
		for(size_t batch = 0; batch < epochBatchCount && !stopTraining; ++batch){
			threadPool.WaitAll();
			fetchBatch(false);
			const float lr = GetLearningRate(epoch, batch, epochBatchCount);
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
		fetchBatch(true);
		for(size_t batch = 0; batch < epochBatchCountVal && !stopTraining; ++batch){
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
	cublasDestroy(cublas);
	cudnnDestroy(cudnn);
}