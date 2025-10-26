#include "NN.h"
#include "BatchNorm.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "CustomOutLayer.h"
#include "EncoderLayer.h"
#include "GlobalPoolLayer.h"
#include "LayerNorm.h"
#include "PatchEmbedLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
NN::NN(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int w, int h, bool train): cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(40), seqLength_(4), gradAccumLength_(1){
	if(!train) batchSize_ = 1;
	batchStateTotal_ = batchSize_*seqLength_;
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
	constexpr auto wd = 0.02f;
	constexpr auto patchSize = 20;
	constexpr auto embedH = 32;
	constexpr auto embedW = 16;
	constexpr auto embedSize = embedH*embedW;
	constexpr auto ffDim = embedSize*4;
	constexpr int numHeads = 8;
	constexpr int numEncoders = 8;
	const int patchRows = DivCeil(netHeight, patchSize);
	const int patchCols = DivCeil(netWidth, patchSize);
	const auto nTokens = patchRows*patchCols;
	constexpr bool enableViewerLayers = false;
	if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchStateTotal_*nTokens*embedSize, 3, netHeight, netWidth, 3, "Input Viewer", true, 1.0f, false));
	layers_.push_back(new PatchEmbedLayer(cudnn_, cublas_, batchStateTotal_, seqLength_, 3, netHeight, netWidth, patchSize, embedSize, "PatchEmbed", train, wd, gradAccumLength_, Xavier));
	if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchStateTotal_*nTokens*embedSize, nTokens, embedH, embedW, patchCols, "Patch Embedding Viewer", true, 1.0f, false));
	for(int i = 0; i < numEncoders; ++i){
		auto name = "Encoder" + std::to_string(i);
		layers_.push_back(new EncoderLayer(cudnn_, cublas_, batchStateTotal_, nTokens, embedSize, ffDim, numHeads, _strdup(name.c_str()), train, wd, gradAccumLength_));
		//if(enableViewerLayers){
			//if(i == 0 || i == numEncoders/2 || i == numEncoders-1) 
				//layers_.push_back(new ViewerLayer(nTokens, embedSqrt, embedSqrt, patchCols, name + " Output Viewer", 1.0f, false));
		//}
	}
	layers_.push_back(new LayerNorm(batchStateTotal_*nTokens, embedSize, 1, 1, "Post-encoder norm", train));
	if(enableViewerLayers) layers_.push_back(new ViewerLayer(batchStateTotal_*nTokens*embedSize, nTokens, embedH, embedW, patchCols, "Encoders Output Viewer", true, 1.0f, false));
	//layers_.push_back(new SpatialActionHead(cudnn_, cublas_, batchStateTotal_, seqLength_, patchRows, patchCols, embedDim, "SpatialActionHead", train, wd, gradAccumLength_));
	layers_.push_back(new GlobalPoolLayer(batchStateTotal_, nTokens, embedSize, "GPL", train));
	layers_.push_back(new CustomOutLayer(cudnn_, cublas_, batchStateTotal_, seqLength_, embedSize, "SpatialActionHead", train, wd, gradAccumLength_));
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
		file.write(reinterpret_cast<const char*>(&inWidth_), sizeof(inWidth_));
		file.write(reinterpret_cast<const char*>(&inHeight_), sizeof(inHeight_));
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