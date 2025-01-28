#include "NN.h"
#include "Activate.h"
#include "ConvLayer.h"
#include "BatchNorm.h"
#include "CustomOutLayer.h"
#include "ResConvLayer.h"
#include "SpatialAttentionLayer.h"
NN::NN(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int w, int h, bool train, float lr): cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(20), seqLength_(1), inWidth_(0), inHeight_(0), learningRate_(lr), maxBufferSize_(0){
	if(!train) batchSize_ = 1;
	batchStateTotal_ = batchSize_*seqLength_;
	int netWidth = w;
	int netHeight = h;
	std::ifstream ckptFile(ckptFileName, std::ios::binary);
	if(ckptFile.is_open()){
		std::cout << "Checkpoint file found...\r\n";
		ckptFile.read(reinterpret_cast<char*>(&netWidth), sizeof(int));
		ckptFile.read(reinterpret_cast<char*>(&netHeight), sizeof(int));
		if(w > 0 && w != netWidth || h > 0 && h != netHeight) throw std::invalid_argument("Training data resolution does not match checkpoint resolution");
	}else{
		std::cout << "Checkpoint file not found\r\n";
		if(w <= 0 || h <= 0){
			throw std::invalid_argument("Invalid training data resolution");
		}
	}
	inWidth_ = netWidth;
	inHeight_ = netHeight;
	stateSize_ = inWidth_*inHeight_*3;
	std::cout << "Initializing layers... ";
	constexpr auto wd = 0.0000001f;
	auto outC = 32;
	layers_.push_back(new ConvLayer(cudnn_, batchStateTotal_, 3, outC, 3, 2, 1, &netHeight, &netWidth, "Conv0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchStateTotal_, outC, netHeight, netWidth, "Conv0 BatchNorm", train, wd));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchStateTotal_, outC, netHeight, netWidth, "Conv0 ReLU"));
	outC = 32;
	layers_.push_back(new ResConvLayer(cudnn_, batchStateTotal_, 32, outC, &netHeight, &netWidth, "ResConv0", train, wd));
	outC = 64;
	layers_.push_back(new ResConvLayer(cudnn_, batchStateTotal_, 32, outC, &netHeight, &netWidth, "ResConv1", train, wd));
	outC = 128;
	layers_.push_back(new ResConvLayer(cudnn_, batchStateTotal_, 64, outC, &netHeight, &netWidth, "ResConv2", train, wd));
	//layers_.push_back(new SpatialAttentionLayer(cudnn_, 32, 8, batchSize_, outC, netHeight, netWidth, "SpatAtt", train, wd));
	layers_.push_back(new CustomOutLayer(cudnn_, cublas_, batchSize_, outC*netHeight*netWidth, "SplitOut", train, wd));
	for(const auto& layer : layers_){
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetOptimizerStateSize());
	}
	std::cout << "Done.\r\n";
	if(ckptFile.is_open()){
		std::cout << "Loading weights... ";
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){
			layer->LoadParameters(ckptFile, buffer);
		}
		ckptFile.close();
		std::cout << "Done.\r\n";
		if(train){
			std::cout << "Loading optimizer state... ";
			std::ifstream optFile(optFileName, std::ios::binary);
			if(optFile.is_open()){
				for(const auto& layer : layers_){
					layer->LoadOptimizerState(optFile, buffer);
				}
				optFile.close();
				std::cout << "Done.\r\n";
			} else{
				std::cerr << "No optimizer state file: " << optFileName << "\r\n";
			}
		}
		cudaFreeHost(buffer);
	}
}
NN::~NN(){
	layers_.clear();
}
__half* NN::Forward(__half* data){
	for(const auto layer : layers_){
		//std::cout << "\r\n" << layer->layerName_ << " ";
		data = layer->Forward(data);
		//PrintDataHalf(data, 16, "data");
	}
	return data;
}
__half* NN::Backward(__half* grad){
	auto outGrad = grad;
	for(int i = layers_.size(); --i >= 0; ){
		//std::cout << "\r\n" << layers_[i]->layerName_ << " ";
		outGrad = layers_[i]->Backward(outGrad);
		//PrintDataHalf(outGrad, 16, "gradient");
	}
	return outGrad;
}
void NN::UpdateParams(){
	for(const auto layer : layers_){ layer->UpdateParameters(learningRate_); }
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
		std::cerr << "Unable to open file for saving checkpoint: " << filename << "\r\n";
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
		std::cerr << "Unable to open file for saving optimizer state: " << filename << "\r\n";
	}
}
void NN::SetTrain(bool enable){
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SetTrain(enable);
	}
}