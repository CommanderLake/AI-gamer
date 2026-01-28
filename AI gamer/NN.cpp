#include "NN.h"
#include "BatchNorm.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "LayerNorm.h"
#include "MLPKBMHead.h"
#include "PatchEmbedLayer.h"
#include "PatchMergingLayer.h"
#include "ResizeLayer.h"
#include "SwinBlockLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
NN::NN(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int w, int h, bool train) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(80), gradAccumLength_(1){
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
	std::cout<<"Initializing layers...\n";
	constexpr auto wd = 0.01f;
	constexpr auto patchSize = 16;
	constexpr auto embedH = 16;
	constexpr auto embedW = 16;
	auto embedSize = embedH*embedW;
	auto ffDim = embedSize*4;
	constexpr int baseHeads = 8;
	constexpr int blocksPerStage = 2;
	constexpr int numMergeStages = 4;
	constexpr int baseWindowSize = 8;
	constexpr float maxDropPathRate = 0.1f;
	constexpr int scaledHeight = 256;
	constexpr int scaledWidth = 256;
	auto patchRows = DivCeil(scaledHeight, patchSize);
	auto patchCols = DivCeil(scaledWidth, patchSize);
	auto nTokens = patchRows*patchCols;
	constexpr bool enableViewerLayers = true;
	layers_.push_back(new ResizeLayer(batchSize_, 3, netHeight, netWidth, scaledHeight, scaledWidth, "Input Resize 256x256", train));
	if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchSize_*3*scaledHeight*scaledWidth, 3, scaledHeight, scaledWidth, 3, "Input Viewer", true, 1.0f, false));
	layers_.push_back(new PatchEmbedLayer(cudnn_, cublas_, batchSize_, 3, scaledHeight, scaledWidth, patchSize, embedSize, "PatchEmbed", train, wd, gradAccumLength_, Xavier));
	int embedDim = embedSize;
	for(int stage = 0; stage < numMergeStages; ++stage){
		if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchSize_*nTokens*embedSize, nTokens, embedH, embedW, patchCols, "Swin Block In Viewer", true, 1.0f, false));
		const int stageHeads = baseHeads << stage;
		const int windowHeight = std::min(baseWindowSize, patchRows);
		const int windowWidth = std::min(baseWindowSize, patchCols);
		const int shiftHeight = windowHeight > 1 ? windowHeight/2 : 0;
		const int shiftWidth = windowWidth > 1 ? windowWidth/2 : 0;
		for(int block = 0; block < blocksPerStage; ++block){
			const int blockIndex = stage*blocksPerStage + block;
			constexpr int totalBlocks = numMergeStages * blocksPerStage;
			const float dropPathRate = totalBlocks > 1 ? maxDropPathRate * (static_cast<float>(blockIndex) / static_cast<float>(totalBlocks - 1)) : 0.0f;
			auto name = "SwinBlock" + std::to_string(blockIndex);
			const bool useShift = block % 2 != 0;
			const int blockShiftHeight = useShift ? shiftHeight : 0;
			const int blockShiftWidth = useShift ? shiftWidth : 0;
			layers_.push_back(new SwinBlockLayer(cudnn_, cublas_, batchSize_, nTokens, embedDim, ffDim, stageHeads, patchRows, patchCols, windowHeight, windowWidth, blockShiftHeight, blockShiftWidth, dropPathRate, _strdup(name.c_str()), train, wd, gradAccumLength_, Xavier));
		}
		auto mergeName = "PatchMerge" + std::to_string(stage);
		layers_.push_back(new PatchMergingLayer(cudnn_, cublas_, batchSize_, nTokens, embedDim, patchRows, patchCols, _strdup(mergeName.c_str()), train, wd, gradAccumLength_, Xavier));
		patchRows /= 2;
		patchCols /= 2;
		nTokens = patchRows*patchCols;
		embedDim *= 2;
		ffDim = embedDim*4;
	}
	layers_.push_back(new LayerNorm(batchSize_*nTokens, embedDim, 1, 1, "Post-encoder norm", train));
	if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchSize_*nTokens*embedDim, nTokens, sqrt(embedDim), sqrt(embedDim), patchCols, "Encoders Output Viewer", true, 1.0f, false));
	layers_.push_back(new MLPKBMHead(cudnn_, cublas_, batchSize_, embedDim, "SpatialActionHead", train, wd, gradAccumLength_));
	for(const auto& layer : layers_){
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetOptimizerStateSize());
	}
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
	for(const auto layer : layers_){ layer->UpdateParameters(lr); }
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
}
