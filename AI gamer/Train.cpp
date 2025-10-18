#include "Train.h"
#include "NN.h"
#include "CuCommon.cuh"
#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <algorithm>
Train::Train(){}
Train::~Train(){}
void Train::Allocate(const int batchSize, const int seqLength, const int stateSize){
	CUDAMallocZero(&dStateBatchBytes, batchSize * seqLength * stateSize * sizeof(unsigned char));
	CUDAMallocZero(&dStateBatchHalf, batchSize * seqLength * stateSize * sizeof(__half));
	checkCUDA(cudaMallocHost(&hTargetBatchFloat, batchSize*seqLength*NUM_CTRLS_*sizeof(float)));
	CUDAMallocZero(&dTargetBatchFloat, batchSize * seqLength * NUM_CTRLS_ * sizeof(float));
	CUDAMallocZero(&dy_, batchSize * seqLength * NUM_CTRLS_ * sizeof(__half));
}
void Train::Free(){
	cudaFree(dy_);
	cudaFree(dTargetBatchFloat);
	cudaFreeHost(hTargetBatchFloat);
	cudaFree(dStateBatchHalf);
	cudaFree(dStateBatchBytes);
}
float GetLearningRate(size_t epoch, size_t batch, size_t epochBatchCount){
	constexpr float baseLr = 0.0001f;
	const float minLr = 0.000001f;
	const size_t warmupSteps = epochBatchCount * 1;
	const size_t totalSteps = epochBatchCount * 10;
	const size_t currentStep = epoch * epochBatchCount + batch;
	if(currentStep < warmupSteps){ return baseLr * static_cast<float>(currentStep) / static_cast<float>(warmupSteps); }
	const float progress = static_cast<float>(currentStep - warmupSteps) / static_cast<float>(totalSteps - warmupSteps);
	const float cosineDecay = 0.5f * (1.0f + cosf(3.14159f * progress));
	return minLr + (baseLr - minLr) * cosineDecay;
}
int Train::TrainBatch(NN* nn, const StateBatch* sb, const bool smoothLoss, const float lr, const std::size_t batchIndex, const std::size_t epochBatchCount){
	for(size_t i = 0; i < nn->batchStateTotal_; ++i){
		for(int j = 0; j < NUM_BUTS_; ++j){ hTargetBatchFloat[i * NUM_CTRLS_ + j] = static_cast<float>(sb->inputStates[i].keyStates >> j & 1); }
		hTargetBatchFloat[i * NUM_CTRLS_ + 14] = static_cast<float>(sb->inputStates[i].deltaX) / 1024.0f;
		hTargetBatchFloat[i * NUM_CTRLS_ + 15] = static_cast<float>(sb->inputStates[i].deltaY) / 1024.0f;
	}
	checkCUDA(cudaMemcpy(dStateBatchBytes, sb->stateData, nn->stateSize_*nn->batchStateTotal_, cudaMemcpyHostToDevice));
	ConvertByteToHalf(dStateBatchBytes, dStateBatchHalf, nn->stateSize_ * nn->batchStateTotal_, true);
	checkCUDA(cudaMemcpy(dTargetBatchFloat, hTargetBatchFloat, NUM_CTRLS_*nn->batchStateTotal_*sizeof(float), cudaMemcpyHostToDevice));
        gDebugOptions.patchNormLogged = false;
        if(gDebugOptions.useReferenceAttention){ gDebugOptions.referenceAttentionTriggered = false; }
        const auto dPredictions = nn->Forward(dStateBatchHalf);
        if(IsnanHalf(dPredictions, NUM_CTRLS_ * nn->batchStateTotal_)){
                std::cout << " NaN in predictions\n";
                return -1;
        }
        const int batchCount = nn->batchStateTotal_;
        const int totalCtrls = NUM_CTRLS_ * batchCount;
        const int totalButtons = NUM_BUTS_ * batchCount;
        const int totalAxes = NUM_AXES_ * batchCount;
        if(gDebugOptions.gradientStripeTest){
                gDebugOptions.gradientStripePending = true;
                checkCUDA(cudaMemset(dy_, 0, totalCtrls*sizeof(__half)));
                const int sample = std::min(gDebugOptions.gradientStripeSampleIndex, batchCount - 1);
                const int logit = std::min(gDebugOptions.gradientStripeLogitIndex, NUM_CTRLS_ - 1);
                size_t offset = 0;
                if(logit < NUM_BUTS_){ offset = static_cast<size_t>(sample)*NUM_BUTS_ + logit; }
                else{ offset = totalButtons + static_cast<size_t>(sample)*NUM_AXES_ + (logit - NUM_BUTS_); }
                float gradVal = 1.0f;
                if(gDebugOptions.applyStaticLossScale){ gradVal *= gDebugOptions.staticLossScale; }
                const __half gradHalf = __float2half(gradVal);
                checkCUDA(cudaMemcpy(dy_ + offset, &gradHalf, sizeof(__half), cudaMemcpyHostToDevice));
                lossButs_ = 0.0f;
                lossAxes_ = 0.0f;
                emaLossButs_ = lossButs_;
                emaLossAxes_ = lossAxes_;
                std::cout << "\rLR: " << lr << " Batch " << (batchIndex + 1) << "/" << epochBatchCount << " GradStripe active";
        } else if(gDebugOptions.useBceWithLogits){
                std::vector<__half> predHost(totalCtrls);
                checkCUDA(cudaMemcpy(predHost.data(), dPredictions, totalCtrls*sizeof(__half), cudaMemcpyDeviceToHost));
                std::vector<__half> gradHost(totalCtrls);
                double buttonLossSum = 0.0;
                double axisLossSum = 0.0;
                for(int b = 0; b < batchCount; ++b){
                        for(int j = 0; j < NUM_BUTS_; ++j){
                                const int predIdx = b*NUM_CTRLS_ + j;
                                const float logit = __half2float(predHost[predIdx]);
                                const float target = hTargetBatchFloat[predIdx];
                                const float negAbs = -std::fabs(logit);
                                const float loss = fmaxf(logit, 0.0f) - logit*target + logf(1.0f + expf(negAbs));
                                buttonLossSum += loss;
                                const float prob = 1.0f / (1.0f + expf(-logit));
                                float grad = prob - target;
                                grad = fmaxf(fminf(grad, 32.0f), -32.0f);
                                gradHost[b*NUM_BUTS_ + j] = __float2half(grad);
                        }
                        for(int a = 0; a < NUM_AXES_; ++a){
                                const int predIdx = b*NUM_CTRLS_ + NUM_BUTS_ + a;
                                const float predVal = __half2float(predHost[predIdx]);
                                const float target = hTargetBatchFloat[predIdx];
                                const float diff = fmaxf(fminf(predVal - target, 32.0f), -32.0f);
                                axisLossSum += (predVal - target)*(predVal - target);
                                gradHost[totalButtons + b*NUM_AXES_ + a] = __float2half(diff);
                        }
                }
                lossButs_ = static_cast<float>(buttonLossSum / (NUM_BUTS_ * batchCount));
                lossAxes_ = static_cast<float>(axisLossSum / (NUM_AXES_ * batchCount));
                if(smoothLoss){
                        constexpr float smoothing = 0.95f;
                        emaLossButs_ = smoothing * emaLossButs_ + (1.0f - smoothing) * lossButs_;
                        emaLossAxes_ = smoothing * emaLossAxes_ + (1.0f - smoothing) * lossAxes_;
                } else{
                        emaLossButs_ = lossButs_;
                        emaLossAxes_ = lossAxes_;
                }
                std::cout << "\rLR: " << lr << " Batch " << (batchIndex + 1) << "/" << epochBatchCount << " Buts: " << emaLossButs_ << " Axes: " << emaLossAxes_;
                checkCUDA(cudaMemcpy(dy_, gradHost.data(), gradHost.size()*sizeof(__half), cudaMemcpyHostToDevice));
        } else{
                MseLoss2(dPredictions, dTargetBatchFloat, NUM_BUTS_, NUM_CTRLS_, batchCount, &lossButs_, &lossAxes_);
                if(smoothLoss){
                        constexpr float smoothing = 0.95f;
                        emaLossButs_ = smoothing * emaLossButs_ + (1.0f - smoothing) * lossButs_;
                        emaLossAxes_ = smoothing * emaLossAxes_ + (1.0f - smoothing) * lossAxes_;
                } else{
                        emaLossButs_ = lossButs_;
                        emaLossAxes_ = lossAxes_;
                }
                std::cout << "\rLR: " << lr << " Batch " << (batchIndex + 1) << "/" << epochBatchCount << " Buts: " << emaLossButs_ << " Axes: " << emaLossAxes_;
                if(lr != 0.0f){
                        SplitGradient(dy_, dPredictions, dTargetBatchFloat, 32.0f, NUM_CTRLS_ * batchCount, NUM_CTRLS_, NUM_BUTS_, batchCount);
                }
        }
        if(lr == 0.0f && !gDebugOptions.gradientStripeTest && !gDebugOptions.useBceWithLogits) return 0;
        if(gDebugOptions.gradientStripeTest){
                if(IsnanHalf(nn->Backward(dy_), nn->stateSize_ * batchCount)){
                        std::cout << " NaN in gradient\n";
                        return -1;
                }
                return 0;
        }
        if(gDebugOptions.applyStaticLossScale && lr != 0.0f){ ScaleArrayHalf(dy_, totalCtrls, gDebugOptions.staticLossScale); }
        if(IsnanHalf(nn->Backward(dy_), nn->stateSize_ * batchCount)){
                std::cout << " NaN in gradient\n";
                return -1;
        }
        if(lr == 0.0f) return 0;
        nn->UpdateParams(lr);
        return 0;
}
void Train::TrainModel(const int width, const int height){
        InitializeDebugOptionsFromEnv();
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
	bool stopTraining = false;
	const auto epochBatchCount = trainRecordIndices.size() / nn->batchStateTotal_;
	const auto epochBatchCountVal = valRecordIndices.size() / nn->batchStateTotal_;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
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
		std::cout << "\nRunning validation...\n";
		nn->SetDropout(false);
		fetchBatch(true);
		for(size_t batch = 0; batch < epochBatchCountVal && !stopTraining; ++batch){
			threadPool.WaitAll();
			fetchBatch(true);
			const auto result = TrainBatch(nn, sbRead, true, 0.0f, batch, epochBatchCountVal);
			if(result == -1){ stopTraining = true; }
		}
		nn->SetDropout(true);
		if(stopTraining){
			std::cout << "\nNaN encountered during validation. Stopping.\n";
			break;
		}
		nn->SaveModel(ckptFileName);
		nn->SaveOptimizerState(optFileName);
	}
	threadPool.WaitAll();
	Free();
	delete nn;
	cublasDestroy(cublas);
	cudnnDestroy(cudnn);
}
void Train::TuneModel(NN* nn, const std::vector<StateSingle*>& states, const int epochs, const float lr){
        InitializeDebugOptionsFromEnv();
        StateBatch sb0(nn->batchStateTotal_, nn->stateSize_);
	StateBatch sb1(nn->batchStateTotal_, nn->stateSize_);
	auto sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&]{
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch]{ LoadBatchFromVector(states, nextBatch, nn->batchSize_, nn->stateSize_); });
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch();
	Allocate(nn->batchSize_, nn->seqLength_, nn->stateSize_);
	const auto epochBatchCount = states.size() / nn->batchStateTotal_;
	bool stopTuning = false;
	for(size_t epoch = 0; epoch < epochs && !stopTuning; ++epoch){
		std::cout << "Epoch: " << epoch << "\n";
		for(size_t batch = 0; batch < epochBatchCount && !stopTuning; ++batch){
			threadPool.WaitAll();
			fetchBatch();
			if(TrainBatch(nn, sbRead, false, lr, batch, epochBatchCount) == -1){
				std::cout << "\nNaN encountered during tuning. Stopping.\n";
				stopTuning = true;
				break;
			}
			std::cout << "\n";
		}
	}
	Free();
}