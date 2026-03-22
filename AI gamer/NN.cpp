#include "NN.h"
#include "BatchNorm.h"
#include "CuCommon.cuh"
#include "ResizeLayer.h"
#include "ActionHead.h"
#include "PatchEmbedLayer.h"
#include "SwinUnetLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
NN::NN(cudnnHandle_t cudnnHandle, int w, int h, bool train) : cudnn_(cudnnHandle), batchSize_(80), gradAccumLength_(1){
	if(!train) batchSize_ = 1;
	int netWidth = w;
	int netHeight = h;
	std::ifstream ckptFile(ckptFileName, std::ios::binary);
	if(ckptFile.is_open()){
		std::cout<<"Checkpoint file found...\n";
		ckptFile.read(reinterpret_cast<char*>(&netWidth), sizeof(int));
		ckptFile.read(reinterpret_cast<char*>(&netHeight), sizeof(int));
		if(w>0&&w!=netWidth||h>0&&h!=netHeight) throw std::invalid_argument("Training data resolution does not match checkpoint resolution");
		std::cout<<"Checkpoint resolution: "<<netWidth<<"x"<<netHeight<<"\n";
	} else{
		std::cout<<"Checkpoint file not found\n";
		if(w<=0||h<=0){ throw std::invalid_argument("Invalid training data resolution"); }
	}
	inWidth_ = netWidth;
	inHeight_ = netHeight;
	stateSize_ = inWidth_*inHeight_*3;
	checkCLNN(InitCublas());
	std::cout<<"Initializing layers...\n";
	constexpr auto wd = 0.1f;
	constexpr auto patchSize = 16;
	constexpr auto embedH = 16;
	constexpr auto embedW = 16;
	auto embedSize = embedH*embedW;
	constexpr int baseHeads = 8;
	constexpr int blocksPerStage = 1;
	constexpr int numMergeStages = 4;
	constexpr int baseWindowSize = 8;
	constexpr float maxDropPathRate = 0.1f;
	constexpr int scaledHeight = 256;
	constexpr int scaledWidth = 256;
	auto patchRows = DivCeil(scaledHeight, patchSize);
	auto patchCols = DivCeil(scaledWidth, patchSize);
	constexpr bool enableViewerLayers = false;
	layers_.push_back(new ResizeLayer(batchSize_, 3, netHeight, netWidth, scaledHeight, scaledWidth, "Input Resize 256x256", train));
	if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchSize_*3*scaledHeight*scaledWidth, 3, scaledHeight, scaledWidth, 3, "Input Viewer", true, 1.0f, false));
	auto nTokens = patchRows*patchCols;
	auto embedDim = embedSize;
	layers_.push_back(new PatchEmbedLayer(cudnn_, batchSize_, 3, scaledHeight, scaledWidth, patchSize, embedDim, "PatchEmbedLayer", train, wd, gradAccumLength_, Xavier));
	layers_.push_back(new SwinUnetLayer(cudnn_, batchSize_, scaledHeight, scaledWidth, patchSize, embedH, embedW, blocksPerStage, numMergeStages, baseHeads, baseWindowSize, maxDropPathRate, "SwinUnet", train, wd, gradAccumLength_, Xavier));
	if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchSize_*nTokens*embedDim, nTokens, sqrt(embedDim), sqrt(embedDim), patchCols, "Encoders Output Viewer", true, 1.0f, false));
	layers_.push_back(new ActionHead(cudnn_, batchSize_, patchRows, patchCols, embedDim, "ActionHead", train, wd, gradAccumLength_));
	for(const auto& layer : layers_){
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetOptimizerStateSize());
	}
	CollectAdamWTasks();
	std::cout<<"Collected AdamW tasks: half="<<adamWHalfTasks_.size()<<", float="<<adamWFloatTasks_.size()<<"\n";
	std::cout<<"Done\n";
	if(ckptFile.is_open()){
		std::cout<<"Loading weights... ";
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->LoadParameters(ckptFile, buffer); }
		ckptFile.close();
		std::cout<<"Done\n";
		if(train){
			std::cout<<"Loading optimizer state... ";
			std::ifstream optFile(optFileName, std::ios::binary);
			if(optFile.is_open()){
				for(const auto& layer : layers_){ layer->LoadOptimizerState(optFile, buffer); }
				optFile.close();
				std::cout<<"Done\n";
			} else{ std::cerr<<"No optimizer state file: "<<optFileName<<"\n"; }
		}
		cudaFreeHost(buffer);
	}
}
NN::~NN(){
	if(dAdamWHalfTasks_){ cudaFree(dAdamWHalfTasks_); }
	if(dAdamWFloatTasks_){ cudaFree(dAdamWFloatTasks_); }
	for(const auto* layer : layers_){
		delete layer;
	}
	layers_.clear();
}
__half* NN::Forward(__half* data){
	for(const auto layer : layers_){
		//std::cout << "\n" << layer->layerName_ << " ";
		data = layer->Forward(data);
		//SummarizeHalfDevice(data, layer->outNCHW_, "data");
	}
	return data;
}
__half* NN::Backward(__half* grad){
	auto outGrad = grad;
	for(int i = layers_.size(); --i >= 0; ){
		//std::cout << "\n" << layers_[i]->layerName_ << " ";
		outGrad = layers_[i]->Backward(outGrad);
		//SummarizeHalfDevice(outGrad, layers_[i]->outNCHW_, "gradient");
	}
	return outGrad;
}
void NN::UpdateParams(const float lr){
	++accumStep_;
	if(accumStep_%gradAccumLength_>0) return;
	if(!adamWHalfTasks_.empty()){ AdamWHalfMulti(dAdamWHalfTasks_, static_cast<int>(adamWHalfTasks_.size()), totalAdamWHalfSize_, lr, adamWStep_); }
	if(!adamWFloatTasks_.empty()){ AdamWFloatMulti(dAdamWFloatTasks_, static_cast<int>(adamWFloatTasks_.size()), totalAdamWFloatSize_, lr, adamWStep_); }
	++adamWStep_;
}
void NN::SaveModel(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		file.write(reinterpret_cast<const char*>(&inWidth_), sizeof inWidth_);
		file.write(reinterpret_cast<const char*>(&inHeight_), sizeof inHeight_);
		for(const auto& layer : layers_){
			layer->SaveParameters(file, buffer);
		}
		cudaFreeHost(buffer);
		file.close();
	} else{
		std::cerr << "Unable to open file for saving checkpoint: " << filename << "\n";
	}
}
void NN::SaveOptimizerState(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){
			layer->SaveOptimizerState(file, buffer);
		}
		cudaFreeHost(buffer);
		file.close();
	} else{
		std::cerr << "Unable to open file for saving optimizer state: " << filename << "\n";
	}
}
void NN::SetTrain(const bool enable){
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SetTrain(enable);
	}
	CollectAdamWTasks();
}
void NN::CollectAdamWTasks(){
	adamWHalfTasks_.clear();
	adamWFloatTasks_.clear();
	totalAdamWHalfSize_ = 0;
	totalAdamWFloatSize_ = 0;
	for(const auto layer : layers_){
		layer->CollectAdamWTasks(adamWHalfTasks_, adamWFloatTasks_);
	}
	for(const auto& task : adamWHalfTasks_){ totalAdamWHalfSize_ += task.size; }
	for(const auto& task : adamWFloatTasks_){ totalAdamWFloatSize_ += task.size; }
	if(dAdamWHalfTasks_){ cudaFree(dAdamWHalfTasks_); dAdamWHalfTasks_ = nullptr; }
	if(dAdamWFloatTasks_){ cudaFree(dAdamWFloatTasks_); dAdamWFloatTasks_ = nullptr; }
	if(!adamWHalfTasks_.empty()){
		checkCUDA(cudaMalloc(&dAdamWHalfTasks_, adamWHalfTasks_.size()*sizeof(AdamWHalfTask)));
		checkCUDA(cudaMemcpy(dAdamWHalfTasks_, adamWHalfTasks_.data(), adamWHalfTasks_.size()*sizeof(AdamWHalfTask), cudaMemcpyHostToDevice));
	}
	if(!adamWFloatTasks_.empty()){
		checkCUDA(cudaMalloc(&dAdamWFloatTasks_, adamWFloatTasks_.size()*sizeof(AdamWFloatTask)));
		checkCUDA(cudaMemcpy(dAdamWFloatTasks_, adamWFloatTasks_.data(), adamWFloatTasks_.size()*sizeof(AdamWFloatTask), cudaMemcpyHostToDevice));
	}
}
