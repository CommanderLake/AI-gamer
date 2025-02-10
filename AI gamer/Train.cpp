#include "Train.h"
#include "NN.h"
#include <iostream>
#include <string>
Train::Train(){}
Train::~Train(){}
void Train::Allocate(const int batchSize, const int sequenceLength, const int stateSize){
	CUDAMallocZero(&dStateBatchBytes, stateSize*batchSize*sequenceLength*sizeof(unsigned char));
	CUDAMallocZero(&dstateBatchHalf, stateSize*batchSize*sequenceLength*sizeof(__half));
	checkCUDA(cudaMallocHost(&hTargetBatchFloat, NUM_CTRLS_*batchSize*sizeof(float)));
	CUDAMallocZero(&dTargetBatchFloat, NUM_CTRLS_*batchSize*sizeof(float));
	CUDAMallocZero(&dTargetBatchHalf, NUM_CTRLS_*batchSize*sizeof(__half));
	CUDAMallocZero(&dGeneratorGrad, NUM_CTRLS_*batchSize*sizeof(__half));
}
void Train::Free(){
	cudaFree(dGeneratorGrad);
	cudaFree(dTargetBatchHalf);
	cudaFree(dTargetBatchFloat);
	cudaFreeHost(hTargetBatchFloat);
	cudaFree(dstateBatchHalf);
	cudaFree(dStateBatchBytes);
}
int Train::TrainBatch(NN* nn, const StateBatch* sb, const int stateSize, bool averageLoss, float lr){
	for(size_t i = 0; i < nn->batchSize_; ++i){
		for(int j = 0; j < NUM_BUTS_; ++j){
			hTargetBatchFloat[i*NUM_CTRLS_ + j] = static_cast<float>(sb->inputStates[i].keyStates >> j & 1);
		}
		hTargetBatchFloat[i*NUM_CTRLS_ + 14] = static_cast<float>(sb->inputStates[i].deltaX)/1024.0f;
		hTargetBatchFloat[i*NUM_CTRLS_ + 15] = static_cast<float>(sb->inputStates[i].deltaY)/1024.0f;
	}
	checkCUDA(cudaMemcpy(dStateBatchBytes, sb->stateData, stateSize*nn->batchStateTotal_, cudaMemcpyHostToDevice));
	ConvertByteToHalf(dstateBatchHalf, dStateBatchBytes, stateSize*nn->batchStateTotal_, true);
	checkCUDA(cudaMemcpy(dTargetBatchFloat, hTargetBatchFloat, NUM_CTRLS_*nn->batchSize_*sizeof(float), cudaMemcpyHostToDevice));
	ConvertFloatToHalf(dTargetBatchFloat, dTargetBatchHalf, NUM_CTRLS_*nn->batchSize_);
	const auto dPredictions = nn->Forward(dstateBatchHalf);
	if(IsnanHalf(dPredictions, NUM_CTRLS_*nn->batchSize_)){
		std::cout << " NaN in predictions\n";
		return -1;
	}
	MseLoss2(dPredictions, dTargetBatchFloat, NUM_BUTS_, NUM_CTRLS_, nn->batchSize_, &lossButs_, &lossAxes_);
	if(averageLoss){
		constexpr float smoothing = 0.98f;
		emaLossButs_ = smoothing*emaLossButs_ + (1.0f - smoothing)*lossButs_;
		emaLossAxes_ = smoothing*emaLossAxes_ + (1.0f - smoothing)*lossAxes_;
	} else{
		emaLossButs_ = lossButs_;
		emaLossAxes_ = lossAxes_;
	}
	std::cout << "\rButs: " << emaLossButs_ << " Axes: " << emaLossAxes_;
	if(lr == 0.0f) return 0;
	SplitGradient(dGeneratorGrad, dPredictions, dTargetBatchHalf, 128.0f, NUM_CTRLS_*nn->batchSize_, NUM_CTRLS_, NUM_BUTS_, nn->batchSize_);
	if(IsnanHalf(nn->Backward(dGeneratorGrad), stateSize*nn->batchStateTotal_)){
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
	//viewer->InitializeWindow(width, height);
	StateBatch sb0(nn->batchStateTotal_, nn->stateSize_);
	StateBatch sb1(nn->batchStateTotal_, nn->stateSize_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&](const bool validation){
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch, validation]{
			LoadBatch(nextBatch, nn->batchSize_, nn->stateSize_, validation);
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
			//for(int i = 0; i<nn->batchSize_*nn->seqLength_; ++i){
			//	viewer->ShowImage(sbRead->stateData + i*stateSize_, width, height);
			//	Sleep(500);
			//}
			const auto result = TrainBatch(nn, sbRead, nn->stateSize_, true, 0.00001f);
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
			const auto result = TrainBatch(nn, sbRead, nn->stateSize_, true, 0.0f);
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
		std::cout << "Epoch: " << epoch << "\n";
		for(size_t batch = 0; batch < epochBatchCount; ++batch){
			threadPool.WaitAll();
			fetchBatch();
			const auto result = TrainBatch(generator, sbRead, generator->stateSize_, false, lr);
			std::cout << "\n";
		}
	}
	Free();
}