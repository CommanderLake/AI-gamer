#include "Train.h"
#include "NN.h"
#include <iostream>
#include <string>
Train::Train(){}
Train::~Train(){}
void Train::Allocate(const int batchSize, const int sequenceLength, const int stateSize){
	CUDAMallocZero(&dStateBatchBytes, stateSize*batchSize*sequenceLength*sizeof(unsigned char));
	CUDAMallocZero(&dstateBatchHalf, stateSize*batchSize*sequenceLength*sizeof(__half));
	checkCUDA(cudaMallocHost(&hTargetBatchFloat, numCtrls_*batchSize*sizeof(float)));
	CUDAMallocZero(&dTargetBatchFloat, numCtrls_*batchSize*sizeof(float));
	CUDAMallocZero(&dTargetBatchHalf, numCtrls_*batchSize*sizeof(__half));
	CUDAMallocZero(&dGeneratorGrad, numCtrls_*batchSize*sizeof(__half));
}
void Train::Free(){
	cudaFree(dGeneratorGrad);
	cudaFree(dTargetBatchHalf);
	cudaFree(dTargetBatchFloat);
	cudaFreeHost(hTargetBatchFloat);
	cudaFree(dstateBatchHalf);
	cudaFree(dStateBatchBytes);
}
int Train::TrainBatch(NN* generator, const StateBatch* sb, const int stateSize, bool averageLoss, float lr){
	for(size_t i = 0; i < generator->batchSize_; ++i){
		for(int j = 0; j < numButs_; ++j){
			hTargetBatchFloat[i*numCtrls_ + j] = static_cast<float>(sb->inputStates[i].keyStates >> j & 1);
		}
		hTargetBatchFloat[i*numCtrls_ + 14] = static_cast<float>(sb->inputStates[i].deltaX)/128.0f;
		hTargetBatchFloat[i*numCtrls_ + 15] = static_cast<float>(sb->inputStates[i].deltaY)/128.0f;
	}
	checkCUDA(cudaMemcpy(dStateBatchBytes, sb->stateData, stateSize*generator->batchStateTotal_, cudaMemcpyHostToDevice));
	ConvertAndNormalize(dstateBatchHalf, dStateBatchBytes, stateSize*generator->batchStateTotal_);
	checkCUDA(cudaMemcpy(dTargetBatchFloat, hTargetBatchFloat, numCtrls_*generator->batchSize_*sizeof(float), cudaMemcpyHostToDevice));
	ConvertFloatToHalf(dTargetBatchFloat, dTargetBatchHalf, numCtrls_*generator->batchSize_);
	const auto dPredictions = generator->Forward(dstateBatchHalf);
	if(IsnanHalf(dPredictions, numCtrls_*generator->batchSize_)){
		std::cout << " NaN in predictions\r\n";
		return -1;
	}
	MseLoss2(dPredictions, dTargetBatchFloat, numButs_, numCtrls_, generator->batchSize_, &lossButs_, &lossAxes_);
	if(averageLoss){
		constexpr float smoothing = 0.95f;
		emaLossButs_ = smoothing*emaLossButs_ + (1.0f - smoothing)*lossButs_;
		emaLossAxes_ = smoothing*emaLossAxes_ + (1.0f - smoothing)*lossAxes_;
	} else{
		emaLossButs_ = lossButs_;
		emaLossAxes_ = lossAxes_;
	}
	std::cout << "\rButs: " << emaLossButs_ << " Axes: " << emaLossAxes_;
	SplitGradient(dGeneratorGrad, dPredictions, dTargetBatchHalf, 128.0f, numCtrls_*generator->batchSize_, numCtrls_, numButs_, generator->batchSize_);
	if(IsnanHalf(generator->Backward(dGeneratorGrad), stateSize*generator->batchStateTotal_)){
		std::cout << " NaN in gradient\r\n";
		return -1;
	}
	generator->UpdateParams(lr);
	return 0;
}
void Train::TrainModel(const int width, const int height){
	int epochs = 10;
	std::cout << "How many epochs: ";
	std::cin >> epochs;
	std::cout << "\r\n";
	InitCUDA();
	cudnnContext* cudnn;
	cublasContext* cublas;
	cudnnCreate(&cudnn);
	cublasCreate(&cublas);
	cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH); //S
	const auto generator = new NN(cudnn, cublas, width, height, true);
	//viewer->InitializeWindow(width, height);
	StateBatch sb0(generator->batchStateTotal_, generator->stateSize_);
	StateBatch sb1(generator->batchStateTotal_, generator->stateSize_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&]{
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch]{
			LoadBatch(nextBatch, generator->batchSize_, generator->stateSize_);
		});
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch();
	Allocate(generator->batchSize_, generator->seqLength_, generator->stateSize_);
	bool nan = false;
	const auto epochBatchCount = recordIndices.size()/generator->batchStateTotal_;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		std::cout << "\r\nEpoch: " << epoch << "\r\n";
		for(size_t batch = 0; batch < epochBatchCount; ++batch){
			nan = false;
			threadPool.WaitAll();
			fetchBatch();
			//for(int i = 0; i<nn->batchSize_*nn->seqLength_; ++i){
			//	viewer->ShowImage(sbRead->stateData + i*stateSize_, width, height);
			//	Sleep(500);
			//}
			const auto result = TrainBatch(generator, sbRead, generator->stateSize_, true, 0.000005f);
			if(result == -1) nan = true;
		}
		if(!nan){
			generator->SaveModel(ckptFileName);
			generator->SaveOptimizerState(optFileName);
		}
	}
	Free();
	delete generator;
	cublasDestroy(cublas);
	cudnnDestroy(cudnn);
}
void Train::TuneModel(NN* generator, const std::vector<StateSingle*>& states, int epochs, float lr){
	StateBatch sb0(generator->batchStateTotal_, generator->stateSize_);
	StateBatch sb1(generator->batchStateTotal_, generator->stateSize_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&]{
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch]{
			LoadBatchFromVector(states, nextBatch, generator->batchSize_, generator->stateSize_);
		});
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch();
	Allocate(generator->batchSize_, generator->seqLength_, generator->stateSize_);
	const auto epochBatchCount = states.size()/generator->batchStateTotal_;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		std::cout << "Epoch: " << epoch << "\r\n";
		for(size_t batch = 0; batch < epochBatchCount; ++batch){
			threadPool.WaitAll();
			fetchBatch();
			const auto result = TrainBatch(generator, sbRead, generator->stateSize_, false, lr);
			std::cout << "\r\n";
		}
	}
	Free();
}