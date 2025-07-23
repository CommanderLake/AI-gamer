#include "NN.h"
#include "BatchNorm.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "CustomOutLayer.h"
#include "EncoderLayer.h"
#include "PatchEmbedLayer.h"
#include "ViewerLayer.h"
NN::NN(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int w, int h, bool train): cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(80), seqLength_(1), gradAccumLength_(1){
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
	constexpr auto wd = 0.000001f;
	auto outC = 768;
	constexpr auto patchSize = 20;
	layers_.push_back(new PatchEmbedLayer(cudnn_, cublas_, batchStateTotal_, 3, netHeight, netWidth, patchSize, outC, "PatchEmbed", train, wd, gradAccumLength_, Xavier));
	const auto numPatches = DivCeil(netHeight, patchSize)*DivCeil(netWidth, patchSize);
	layers_.push_back(new EncoderLayer(cudnn_, cublas_, batchStateTotal_, numPatches, outC, outC, 4, "Encoder0", train, wd, gradAccumLength_));
	//layers_.push_back(new ViewerLayer(numPatches, 24, 32, 16, "PatchEmbedLayer viewer"));
	layers_.push_back(new CustomOutLayer(cudnn_, cublas_, batchStateTotal_, seqLength_, outC*numPatches, "SplitOut", train, wd, gradAccumLength_));
	for(const auto& layer : layers_){
		maxBufferSize_ = max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = max(maxBufferSize_, layer->GetOptimizerStateSize());
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
void NN::SetDropout(const bool enable){
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SetDropout(enable);
	}
}