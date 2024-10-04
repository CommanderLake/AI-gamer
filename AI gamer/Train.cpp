#include "Train.h"
#include "NN.h"
#include "Discriminator.h"
#include <iostream>
#include <string>
Train::Train(){}
Train::~Train(){}
void Train::TrainModel(size_t count, int width, int height, Viewer* viewer){
	InitCUDA();
	cudnnContext* cudnn;
	cudnnCreate(&cudnn);
	cublasContext* cublas;
	cublasCreate(&cublas);
	cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);//S
	const auto generator = new NN(cudnn, cublas, width, height, true);
	//const auto discriminator = new Discriminator(cudnn, cublas, generator->batchSize_, numCtrls_, true);
	int epochs = 10;
	std::cout << "How many epochs: ";
	std::cin >> epochs;
	std::cout << "\r\n";
	//viewer->InitializeWindow(width, height);
	StateBatch sb0(generator->batchSize_, stateSize_);
	StateBatch sb1(generator->batchSize_, stateSize_);
	const StateBatch* sbRead = &sb0;
	bool sbSwitch = false;
	auto fetchBatch = [&]{
		sbSwitch = !sbSwitch;
		StateBatch* nextBatch = sbSwitch ? &sb1 : &sb0;
		threadPool.Enqueue([&, nextBatch]{
			LoadBatch(nextBatch, generator->batchSize_);
		});
		sbRead = sbSwitch ? &sb0 : &sb1;
	};
	fetchBatch();
	unsigned char* dStateBatchBytes = nullptr;
	__half* dstateBatchHalf = nullptr;
	float* hCtrlBatchFloat = nullptr;
	float* dCtrlBatchFloat = nullptr;
	__half* dCtrlBatchHalf = nullptr;
	//__half* dRealLabels = nullptr;
	//__half* dFakeLabels = nullptr;
	__half* dGeneratorGrad = nullptr;
	CUDAMallocZero(&dStateBatchBytes, stateSize_*generator->batchSize_*sizeof(unsigned char));
	CUDAMallocZero(&dstateBatchHalf, stateSize_*generator->batchSize_*sizeof(__half));
	checkCUDA(cudaMallocHost(&hCtrlBatchFloat, numCtrls_*generator->batchSize_*sizeof(float)));
	CUDAMallocZero(&dCtrlBatchFloat, numCtrls_*generator->batchSize_*sizeof(float));
	CUDAMallocZero(&dCtrlBatchHalf, numCtrls_*generator->batchSize_*sizeof(__half));
	//CUDAMallocZero(&dRealLabels, numCtrls_*generator->batchSize_*sizeof(__half));
	//CUDAMallocZero(&dFakeLabels, numCtrls_*generator->batchSize_*sizeof(__half));
	//const std::vector<__half> hRealLabels(numCtrls_*generator->batchSize_, __float2half(1.0f));
	//checkCUDA(cudaMemcpy(dRealLabels, hRealLabels.data(), numCtrls_*generator->batchSize_*sizeof(__half), cudaMemcpyHostToDevice));
	CUDAMallocZero(&dGeneratorGrad, numCtrls_*generator->batchSize_*sizeof(__half));
	bool nan = false;
	auto lossButs = 0.0f;
	auto lossAxes = 0.0f;
	auto emaLossButs = 0.0f;
	auto emaLossAxes = 0.0f;
	generator->learningRate_ = 0.000001f;
	//discriminator->learningRate_ = generator->learningRate_/2.0f;
	for(size_t epoch = 0; epoch < epochs; ++epoch){
		std::cout << "\r\nEpoch: " << epoch << "\r\n";
		for(size_t batch = 0; batch < count/generator->batchSize_; ++batch){
			nan = false;
			threadPool.WaitAll();
			fetchBatch();
			//for(int i = 0; i<nn->batchSize_*nn->seqLength_; ++i){
			//	viewer->ShowImage(sbRead->stateData + i*stateSize_, width, height);
			//	Sleep(500);
			//}
			for(size_t i = 0; i < generator->batchSize_; ++i){
				for(int j = 0; j < numButs_; ++j){
					hCtrlBatchFloat[i*numCtrls_ + j] = static_cast<float>(sbRead->keyStates[i] >> j & 1);
				}
				hCtrlBatchFloat[i*numCtrls_ + 14] = static_cast<float>(sbRead->mouseDeltaX[i])/256.0f;
				hCtrlBatchFloat[i*numCtrls_ + 15] = static_cast<float>(sbRead->mouseDeltaY[i])/256.0f;
			}
			checkCUDA(cudaMemcpy(dStateBatchBytes, sbRead->stateData, stateSize_*generator->batchSize_, cudaMemcpyHostToDevice));
			ConvertAndNormalize(dstateBatchHalf, dStateBatchBytes, stateSize_*generator->batchSize_);
			checkCUDA(cudaMemcpy(dCtrlBatchFloat, hCtrlBatchFloat, numCtrls_*generator->batchSize_*sizeof(float), cudaMemcpyHostToDevice));
			ConvertFloatToHalf(dCtrlBatchFloat, dCtrlBatchHalf, numCtrls_*generator->batchSize_);
			const auto dPredictions = generator->Forward(dstateBatchHalf);
			if(IsnanHalf(dPredictions, numCtrls_*generator->batchSize_)){
				std::cout << "\r\nNaN in predictions\r\n";
				nan = true;
				continue;
			}
			MseLoss2(dPredictions, dCtrlBatchFloat, numButs_, numCtrls_, generator->batchSize_, &lossButs, &lossAxes);
			constexpr float smoothing = 0.99f;
			emaLossButs = smoothing*emaLossButs + (1.0f - smoothing)*lossButs;
			emaLossAxes = smoothing*emaLossAxes + (1.0f - smoothing)*lossAxes;
			std::cout << "\rButs: " << emaLossButs << " Axes: " << emaLossAxes;
			//ClearScreen(' ');
			//PrintDataFloat(dCtrlBatchFloat, 16, "\r\n\r\nTargets");
			//PrintDataHalf(dPredictions, 16, "Predictions");
			//discriminator->Backward(discriminator->Forward(dCtrlBatchHalf), dRealLabels);
			//const auto discFakePred = discriminator->Forward(dPredictions);
			//discriminator->Backward(discFakePred, dFakeLabels);
			//GAILGradient(dGeneratorGrad, dPredictions, discFakePred, dCtrlBatchFloat, generator->batchSize_, numCtrls_, numButs_, 0.5f, 0.01f, 1.0f, 1.0f, 32.0f);
			Gradient(dGeneratorGrad, dPredictions, dCtrlBatchHalf, numCtrls_*generator->batchSize_);
			//PrintDataHalf(dGeneratorGrad, numCtrls_*generator->batchSize_, "thingy");
			//PrintDataHalf(dGeneratorGrad, 16, "dGeneratorGrad");
			if(IsnanHalf(generator->Backward(dGeneratorGrad), stateSize_*generator->batchSize_)){
				std::cout << " NaN in gradient\r\n";
				nan = true;
				continue;
			}
			generator->UpdateParams();
			//discriminator->UpdateParams();
		}
		if(!nan){
			generator->SaveModel(ckptFileName);
			generator->SaveOptimizerState(optFileName);
			//discriminator->SaveModel(ckptFileNameDisc);
			//discriminator->SaveOptimizerState(optFileNameDisc);
		}
	}
	checkCUDA(cudaFree(dStateBatchBytes));
	checkCUDA(cudaFree(dstateBatchHalf));
	checkCUDA(cudaFreeHost(hCtrlBatchFloat));
	checkCUDA(cudaFree(dCtrlBatchFloat));
	checkCUDA(cudaFree(dCtrlBatchHalf));
	//checkCUDA(cudaFree(dRealLabels));
	//checkCUDA(cudaFree(dFakeLabels));
	checkCUDA(cudaFree(dGeneratorGrad));
	delete generator;
	//delete discriminator;
}