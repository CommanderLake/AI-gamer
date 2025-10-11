#include "Train.h"
#include "NN.h"
#include "CuCommon.cuh"
#include <iostream>
#include <string>
Train::Train(){}
Train::~Train(){}
void Train::Allocate(const int batchSize, const int seqLength, const int stateSize){
	CUDAMallocZero(&dStateBatchBytes, batchSize*seqLength*stateSize*sizeof(unsigned char));
	CUDAMallocZero(&dStateBatchHalf, batchSize*seqLength*stateSize*sizeof(__half));
	checkCUDA(cudaMallocHost(&hTargetBatchFloat, batchSize*seqLength*NUM_CTRLS_*sizeof(float)));
	CUDAMallocZero(&dTargetBatchFloat, batchSize*seqLength*NUM_CTRLS_*sizeof(float));
	CUDAMallocZero(&dy_, batchSize*seqLength*NUM_CTRLS_*sizeof(__half));
}
void Train::Free(){
	cudaFree(dy_);
	cudaFree(dTargetBatchFloat);
	cudaFreeHost(hTargetBatchFloat);
	cudaFree(dStateBatchHalf);
	cudaFree(dStateBatchBytes);
}
float GetLearningRate(size_t epoch, size_t batch, size_t epochBatchCount){
	constexpr float baseLr = 0.001f;
	const float minLr = 0.00001f;
	const size_t warmupSteps = epochBatchCount*1;
	const size_t totalSteps = epochBatchCount*10;
	const size_t currentStep = epoch * epochBatchCount + batch;
	if(currentStep < warmupSteps){
		return baseLr * static_cast<float>(currentStep) / static_cast<float>(warmupSteps);
	}
	const float progress = static_cast<float>(currentStep - warmupSteps) / static_cast<float>(totalSteps - warmupSteps);
	const float cosineDecay = 0.5f * (1.0f + cosf(3.14159f*progress));
	return minLr + (baseLr - minLr) * cosineDecay;
}
int Train::TrainBatch(NN* nn, const StateBatch* sb, const bool smoothLoss, const float lr){
	for(size_t i = 0; i < nn->batchStateTotal_; ++i){
		for(int j = 0; j < NUM_BUTS_; ++j){
			hTargetBatchFloat[i*NUM_CTRLS_ + j] = static_cast<float>(sb->inputStates[i].keyStates >> j & 1);
		}
		hTargetBatchFloat[i*NUM_CTRLS_ + 14] = static_cast<float>(sb->inputStates[i].deltaX)/1024.0f;
		hTargetBatchFloat[i*NUM_CTRLS_ + 15] = static_cast<float>(sb->inputStates[i].deltaY)/1024.0f;
	}
	checkCUDA(cudaMemcpy(dStateBatchBytes, sb->stateData, nn->stateSize_*nn->batchStateTotal_, cudaMemcpyHostToDevice));
	ConvertByteToHalf(dStateBatchBytes, dStateBatchHalf, nn->stateSize_*nn->batchStateTotal_, true);
	checkCUDA(cudaMemcpy(dTargetBatchFloat, hTargetBatchFloat, NUM_CTRLS_*nn->batchStateTotal_*sizeof(float), cudaMemcpyHostToDevice));
	const auto dPredictions = nn->Forward(dStateBatchHalf);
	if(IsnanHalf(dPredictions, NUM_CTRLS_*nn->batchStateTotal_)){
		std::cout << " NaN in predictions\n";
		return -1;
	}
	MseLoss2(dPredictions, dTargetBatchFloat, NUM_BUTS_, NUM_CTRLS_, nn->batchStateTotal_, &lossButs_, &lossAxes_);
	if(smoothLoss){
		constexpr float smoothing = 0.98f;
		emaLossButs_ = smoothing*emaLossButs_ + (1.0f - smoothing)*lossButs_;
		emaLossAxes_ = smoothing*emaLossAxes_ + (1.0f - smoothing)*lossAxes_;
	} else{
		emaLossButs_ = lossButs_;
		emaLossAxes_ = lossAxes_;
	}
	std::cout << "\rButs: " << emaLossButs_ << " Axes: " << emaLossAxes_;
	if(lr == 0.0f) return 0;
	SplitGradient(dy_, dPredictions, dTargetBatchFloat, 32.0f, NUM_CTRLS_*nn->batchStateTotal_, NUM_CTRLS_, NUM_BUTS_, nn->batchStateTotal_);
	if(IsnanHalf(nn->Backward(dy_), nn->stateSize_*nn->batchStateTotal_)){
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
	StateBatch sb0(nn->batchStateTotal_, nn->stateSize_);
	StateBatch sb1(nn->batchStateTotal_, nn->stateSize_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&](const bool validation){
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch, validation]{
			LoadBatch(nextBatch, nn->batchSize_, nn->stateSize_, validation);
			//LoadBatchLSTM(nextBatch, nn->batchSize_, nn->seqLength_, nn->stateSize_, validation);
		});
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch(false);
	Allocate(nn->batchSize_, nn->seqLength_, nn->stateSize_);
	bool nan = false;
	const auto epochBatchCount = trainRecordIndices.size()/nn->batchStateTotal_;
	const auto epochBatchCountVal = valRecordIndices.size()/nn->batchStateTotal_;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		std::cout << "\nEpoch: " << epoch << "\n";
		for(size_t batch = 0; batch < epochBatchCount; ++batch){
			nan = false;
			threadPool.WaitAll();
			fetchBatch(false);
			const float lr = GetLearningRate(epoch, batch, epochBatchCount);
			const auto result = TrainBatch(nn, sbRead, true, lr);
			if(result == -1) nan = true;
		}
		threadPool.WaitAll();
		std::cout << "\nRunning validation...\n";
		nn->SetDropout(false);
		fetchBatch(true);
		for(size_t batch = 0; batch < epochBatchCountVal; ++batch){
			nan = false;
			threadPool.WaitAll();
			fetchBatch(true);
			const auto result = TrainBatch(nn, sbRead, true, 0.0f);
			if(result == -1) nan = true;
		}
		nn->SetDropout(true);
		if(!nan){
			nn->SaveModel(ckptFileName);
			nn->SaveOptimizerState(optFileName);
		}
	}
	Free();
	delete nn;
	cublasDestroy(cublas);
	cudnnDestroy(cudnn);
}
void Train::TuneModel(NN* nn, const std::vector<StateSingle*>& states, const int epochs, const float lr){
	StateBatch sb0(nn->batchStateTotal_, nn->stateSize_);
	StateBatch sb1(nn->batchStateTotal_, nn->stateSize_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&]{
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch]{
			LoadBatchFromVector(states, nextBatch, nn->batchSize_, nn->stateSize_);
		});
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch();
	Allocate(nn->batchSize_, nn->seqLength_, nn->stateSize_);
	const auto epochBatchCount = states.size()/nn->batchStateTotal_;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		std::cout << "Epoch: " << epoch << "\n";
		for(size_t batch = 0; batch < epochBatchCount; ++batch){
			threadPool.WaitAll();
			fetchBatch();
			const auto result = TrainBatch(nn, sbRead, false, lr);
			std::cout << "\n";
		}
	}
	Free();
}