#include "Train.h"
#include "NN.h"
#include "Discriminator.h"
#include <iostream>
#include <string>
Train::Train(){}
Train::~Train(){}
void Train::TrainModel(size_t count, int width, int height, Viewer* viewer){
	InitCUDA();
	const auto nn = new NN(width, height, true);
	const auto discriminator = new Discriminator(nn->batchSize_, numCtrls_, true);
	int epochs = 10;
	std::cout << "How many epochs: ";
	std::cin >> epochs;
	std::cout << "\r\n";
	StateBatch sb0(nn->batchSize_, stateSize);
	StateBatch sb1(nn->batchSize_, stateSize);
	const StateBatch* sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&]{
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch]{
			LoadBatch(nextBatch, nn->seqLength_, nn->batchSize_/nn->seqLength_);
		});
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch();
	unsigned char* dBatchInputBytes = nullptr;
	__half* dBatchInputHalf = nullptr;
	float* hCtrlBatchFloat = nullptr;
	float* dCtrlBatchFloat = nullptr;
	__half* dCtrlBatchHalf = nullptr;
	checkCUDA(cudaMalloc(&dBatchInputBytes, stateSize*nn->batchSize_*sizeof(unsigned char)));
	checkCUDA(cudaMalloc(&dBatchInputHalf, stateSize*nn->batchSize_*sizeof(__half)));
	checkCUDA(cudaMallocHost(&hCtrlBatchFloat, nn->batchSize_*numCtrls_*sizeof(float)));
	checkCUDA(cudaMalloc(&dCtrlBatchFloat, numCtrls_*nn->batchSize_*sizeof(float)));
	checkCUDA(cudaMalloc(&dCtrlBatchHalf, numCtrls_*nn->batchSize_*sizeof(__half)));
	__half* dRealLabels = nullptr;
	__half* dFakeLabels = nullptr;
	checkCUDA(cudaMalloc(&dRealLabels, nn->batchSize_*sizeof(__half)));
	checkCUDA(cudaMalloc(&dFakeLabels, nn->batchSize_*sizeof(__half)));
	const std::vector<__half> hRealLabels(nn->batchSize_, __float2half(1.0f));
	checkCUDA(cudaMemcpy(dRealLabels, hRealLabels.data(), nn->batchSize_*sizeof(__half), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemset(dFakeLabels, 0, nn->batchSize_*sizeof(__half)));
	__half* dGeneratorGrad = nullptr;
	checkCUDA(cudaMalloc(&dGeneratorGrad, numCtrls_*nn->batchSize_*sizeof(__half)));
	bool nan = false;
	auto lossButs = 0.0f;
	auto lossAxes = 0.0f;
	auto emaLossButs = 0.0f;
	auto emaLossAxes = 0.0f;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		constexpr auto lrLower = 0.000002f, lrUpper = 0.00002f;
		nn->learningRate_ = epoch%5==0 ? lrUpper : lrLower;
		discriminator->learningRate_ = nn->learningRate_/2.0f;
		std::cout << "\r\nEpoch: " << epoch << "\r\n";
		for(size_t batch = 0; batch < count/nn->batchSize_; ++batch){
			threadPool.WaitAll();
			fetchBatch();
			for(size_t i = 0; i < nn->batchSize_; ++i){
				for(int j = 0; j < numButs_; ++j){
					hCtrlBatchFloat[i*numCtrls_ + j] = static_cast<float>(sbRead->keyStates[i] >> j & 1);
				}
				hCtrlBatchFloat[i*numCtrls_ + 14] = static_cast<float>(sbRead->mouseDeltaX[i])/256.0f;
				hCtrlBatchFloat[i*numCtrls_ + 15] = static_cast<float>(sbRead->mouseDeltaY[i])/256.0f;
			}
			checkCUDA(cudaMemcpy(dBatchInputBytes, sbRead->stateData, stateSize*nn->batchSize_*sizeof(unsigned char), cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(dCtrlBatchFloat, hCtrlBatchFloat, numCtrls_*nn->batchSize_*sizeof(float), cudaMemcpyHostToDevice));
			ConvertFloatToHalf(dCtrlBatchFloat, dCtrlBatchHalf, numCtrls_*nn->batchSize_);
			ConvertAndNormalize(dBatchInputHalf, dBatchInputBytes, nn->batchSize_*stateSize);
			const auto dPredictions = nn->Forward(dBatchInputHalf);
			discriminator->Backward(discriminator->Forward(dCtrlBatchHalf), dRealLabels);
			const auto discFakePred = discriminator->Forward(dPredictions);
			discriminator->Backward(discFakePred, dFakeLabels);
			discriminator->UpdateParams();
			GAILGradient(dGeneratorGrad, dPredictions, discFakePred, dCtrlBatchFloat, nn->batchSize_, numCtrls_, numButs_, 0.25f, 0.025f, 128.0f, 128.0f, 128.0f);
			MseLoss2(dPredictions, dCtrlBatchFloat, numButs_, numCtrls_, nn->batchSize_, &lossButs, &lossAxes);
			if(batch == 0 && epoch == 0){
				emaLossButs = lossButs;
				emaLossAxes = lossAxes;
			} else{
				constexpr float smoothing = 0.99f;
				emaLossButs = smoothing*emaLossButs + (1.0f - smoothing)*lossButs;
				emaLossAxes = smoothing*emaLossAxes + (1.0f - smoothing)*lossAxes;
			}
			std::cout << "Buts: " << emaLossButs << " Axes: " << emaLossAxes << "\r";
			if(isnan(lossButs) || isnan(lossAxes)){
				std::cout << "Loss is NaN         \r\n";
				nan = true;
				break;
			}
			const auto gradOut = nn->Backward(dGeneratorGrad);
			if(IsnanHalf(gradOut, nn->inWidth_*nn->inHeight_*3)){
				std::cout << "NaN in gradient     \r\n";
				nan = true;
				break;
			}
			nn->UpdateParams();
		}
		if(nan) break;
		nn->SaveModel(ckptFileName);
		nn->SaveOptimizerState(optFileName);
		discriminator->SaveModel(ckptFileNameDisc);
		discriminator->SaveOptimizerState(optFileNameDisc);
	}
	checkCUDA(cudaFree(dBatchInputBytes));
	checkCUDA(cudaFree(dBatchInputHalf));
	checkCUDA(cudaFreeHost(hCtrlBatchFloat));
	checkCUDA(cudaFree(dCtrlBatchFloat));
	checkCUDA(cudaFree(dCtrlBatchHalf));
	checkCUDA(cudaFree(dRealLabels));
	checkCUDA(cudaFree(dFakeLabels));
	checkCUDA(cudaFree(dGeneratorGrad));
	delete nn;
	delete discriminator;
}