#include "NN.h"
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
	constexpr auto wd = 0.00001f;
	auto outC = 768;
	//layers_.push_back(new ViewerLayer(seqLength_*3, netHeight, netWidth, 6, "input viewer"));
	//layers_.push_back(new ConvLayer(cudnn_, batchStateTotal_, 3, outC, 5, 2, &netHeight, &netWidth, "Conv0A", train, wd, gradAccumLength_));
	//layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchStateTotal_, outC, netHeight, netWidth, "Conv0A_BatchNorm", train, wd, gradAccumLength_));
	//layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchStateTotal_, outC, netHeight, netWidth, "Conv0A_ReLU"));
	////layers_.push_back(new ViewerLayer(outC, netHeight, netWidth, 8, "Conv0A viewer"));
	//layers_.push_back(new PoolLayer(cudnn_, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, batchSize_, outC, &netHeight, &netWidth, 2, 2, "Conv0_MaxPool", train));
	//outC = 64;
	//layers_.push_back(new ConvLayer(cudnn_, batchStateTotal_, 32, outC, 4, 2, &netHeight, &netWidth, "Conv1A", train, wd, gradAccumLength_));
	//layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchStateTotal_, outC, netHeight, netWidth, "Conv1A_BatchNorm", train, wd, gradAccumLength_));
	//layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchStateTotal_, outC, netHeight, netWidth, "Conv1A_ReLU"));
	////layers_.push_back(new ViewerLayer(outC, netHeight, netWidth, 16, "Conv1A viewer"));
	//outC = 128;
	//layers_.push_back(new ConvLayer(cudnn_, batchStateTotal_, 64, outC, 4, 2, &netHeight, &netWidth, "Conv2A", train, wd, gradAccumLength_));
	//layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchStateTotal_, outC, netHeight, netWidth, "Conv2A_BatchNorm", train, wd, gradAccumLength_));
	//layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchStateTotal_, outC, netHeight, netWidth, "Conv2A_ReLU"));
	////layers_.push_back(new ViewerLayer(outC*seqLength_, netHeight, netWidth, 32, "Conv2A viewer"));
	//outC = 256;
	//layers_.push_back(new ConvLayer(cudnn_, batchStateTotal_, 128, outC, 3, 1, &netHeight, &netWidth, "Conv3A", train, wd, gradAccumLength_));
	//layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchStateTotal_, outC, netHeight, netWidth, "Conv3A_BatchNorm", train, wd, gradAccumLength_));
	//layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchStateTotal_, outC, netHeight, netWidth, "Conv3A_ReLU"));
	//layers_.push_back(new ViewerLayer(outC*seqLength_, netHeight, netWidth, 32, "Conv3A viewer"));
	constexpr auto patchSize = 40;
	layers_.push_back(new PatchEmbedLayer(cudnn_, cublas_, batchStateTotal_, 3, netHeight, netWidth, patchSize, outC, "PatchEmbed", train, wd, gradAccumLength_));
	const auto patchRows = DivCeil(netHeight, patchSize);
	const auto patchCols = DivCeil(netWidth, patchSize);
	const auto numPatches = patchRows * patchCols;
	layers_.push_back(new EncoderLayer(cudnn_, cublas_, batchSize_, numPatches, outC, outC, 4, "Encoder0", train, wd, gradAccumLength_));
	layers_.push_back(new CustomOutLayer(cudnn_, cublas_, batchSize_, seqLength_, outC*numPatches, "SplitOut", train, wd, gradAccumLength_));
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
		std::cout << "\n" << layer->layerName_ << " ";
		data = layer->Forward(data);
		PrintDataHalfDevice(data, 16, "data");
	}
	return data;
}
__half* NN::Backward(__half* grad){
	auto outGrad = grad;
	for(int i = layers_.size(); --i >= 0; ){
		std::cout << "\n" << layers_[i]->layerName_ << " ";
		outGrad = layers_[i]->Backward(outGrad);
		PrintDataHalfDevice(outGrad, 16, "gradient");
	}
	return outGrad;
}
void NN::UpdateParams(float lr){
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