#include "Train.h"
#include "NN.h"
#include "CuCommon.cuh"
#include <iostream>
#include <string>
Train::Train(){}
Train::~Train(){}
void Train::Allocate(const int batchSize, const int seqLength, const int stateSize, const int controlTokens){
	CUDAMallocZero(&dStateBatchBytes, batchSize*seqLength*stateSize*sizeof(unsigned char));
	stateHalfStride_ = static_cast<size_t>(stateSize) + static_cast<size_t>(controlTokens)*NUM_CTRLS_;
	stateHalfCount_ = static_cast<size_t>(batchSize)*seqLength*stateHalfStride_;
	CUDAMallocZero(&dStateBatchHalf, stateHalfCount_*sizeof(__half));
	checkCUDA(cudaMallocHost(&hTargetBatchFloat, batchSize*seqLength*NUM_CTRLS_*sizeof(float)));
	CUDAMallocZero(&dTargetBatchFloat, batchSize*seqLength*NUM_CTRLS_*sizeof(float));
	CUDAMallocZero(&dy_, batchSize*seqLength*NUM_CTRLS_*sizeof(__half));
	controlTokenCount_ = controlTokens;
	controlUploadCount_ = 0;
	if(controlTokenCount_>0){
		controlUploadCount_ = static_cast<size_t>(batchSize)*seqLength*controlTokenCount_*NUM_CTRLS_;
		checkCUDA(cudaMallocHost(&hControlTokensFloat_, controlUploadCount_*sizeof(float)));
		checkCUDA(cudaMallocHost(&hControlTokensHalf_, controlUploadCount_*sizeof(__half)));
	}
}
void Train::Free(){
	cudaFree(dy_);
	cudaFree(dTargetBatchFloat);
	cudaFreeHost(hTargetBatchFloat);
	cudaFree(dStateBatchHalf);
	cudaFree(dStateBatchBytes);
	if(hControlTokensFloat_){ cudaFreeHost(hControlTokensFloat_); hControlTokensFloat_ = nullptr; }
	if(hControlTokensHalf_){ cudaFreeHost(hControlTokensHalf_); hControlTokensHalf_ = nullptr; }
	controlTokenCount_ = 0;
	controlUploadCount_ = 0;
	stateHalfStride_ = 0;
	stateHalfCount_ = 0;
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
	auto encodeInputState = [](const InputState& inputState, float* dst){
		for(int j = 0; j < NUM_BUTS_; ++j){ dst[j] = static_cast<float>((inputState.keyStates >> j) & 1); }
		dst[NUM_BUTS_] = std::asinh(static_cast<float>(inputState.deltaX) / AXIS_SCALE_);
		dst[NUM_BUTS_ + 1] = std::asinh(static_cast<float>(inputState.deltaY) / AXIS_SCALE_);
	};
	for(size_t i = 0; i < nn->batchStateTotal_; ++i){
		encodeInputState(sb->inputStates[i], &hTargetBatchFloat[i*NUM_CTRLS_]);
	}
	if(controlTokenCount_>0 && sb->controlHistory){
		const size_t historyLen = static_cast<size_t>(controlTokenCount_);
		for(size_t i = 0; i < nn->batchStateTotal_; ++i){
			const auto* historyBase = sb->controlHistory + i*historyLen;
			for(size_t h = 0; h < historyLen; ++h){
				float features[NUM_CTRLS_] = {};
				encodeInputState(historyBase[h], features);
				auto* dst = hControlTokensFloat_ + (i*historyLen + h)*NUM_CTRLS_;
				memcpy(dst, features, NUM_CTRLS_*sizeof(float));
			}
		}
		const size_t controlCount = nn->batchStateTotal_*historyLen*NUM_CTRLS_;
		FloatToHalfAsm(hControlTokensFloat_, hControlTokensHalf_, static_cast<int>(controlCount));
		auto* controlDst = dStateBatchHalf + static_cast<size_t>(nn->stateSize_)*nn->batchStateTotal_;
		checkCUDA(cudaMemcpy(controlDst, hControlTokensHalf_, controlCount*sizeof(__half), cudaMemcpyHostToDevice));
	} else if(controlTokenCount_>0){
		auto* controlDst = dStateBatchHalf + static_cast<size_t>(nn->stateSize_)*nn->batchStateTotal_;
		const size_t controlCount = static_cast<size_t>(nn->batchStateTotal_)*controlTokenCount_*NUM_CTRLS_;
		checkCUDA(cudaMemset(controlDst, 0, controlCount*sizeof(__half)));
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
		constexpr float smoothing = 0.95f;
		emaLossButs_ = smoothing*emaLossButs_ + (1.0f - smoothing)*lossButs_;
		emaLossAxes_ = smoothing*emaLossAxes_ + (1.0f - smoothing)*lossAxes_;
	} else{
		emaLossButs_ = lossButs_;
		emaLossAxes_ = lossAxes_;
	}
	std::cout << "\rLR: " << lr << " Batch " << (batchIndex + 1) << "/" << epochBatchCount << " Buts: " << emaLossButs_ << " Axes: " << emaLossAxes_;
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
	StateBatch sb0(nn->batchSize_, nn->stateSize_, nn->seqLength_, nn->controlTokenCount_);
	StateBatch sb1(nn->batchSize_, nn->stateSize_, nn->seqLength_, nn->controlTokenCount_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&](const bool validation){
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch, validation]{ LoadBatchLSTM(nextBatch, nn->batchSize_, nn->seqLength_, nn->stateSize_, validation); });
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch(false);
	Allocate(nn->batchSize_, nn->seqLength_, nn->stateSize_, nn->controlTokenCount_);
	bool stopTraining = false;
	const auto epochBatchCount = trainRecordIndices.size() / nn->batchStateTotal_;
	const auto epochBatchCountVal = valRecordIndices.size() / nn->batchStateTotal_;
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