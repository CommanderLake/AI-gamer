#include "NN.h"
#include "Activate.h"
#include "ConvLayer.h"
#include "FCLayer.h"
#include "BatchNorm.h"
#include "Dropout.h"
#include "LeakyReLU.h"
#include "ResConvLayer.h"
#include "Sigmoid.h"
#include "Swish.h"
NN::NN(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int w, int h, bool train): cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(40), seqLength_(1), inWidth_(0), inHeight_(0), learningRate_(0.00001f), maxBufferSize_(0){
	if(!train) batchSize_ = 1;
	std::ifstream ckptFile(ckptFileName, std::ios::binary);
	const bool fileOpen = ckptFile.is_open();
	int netWidth = w;
	int netHeight = h;
	if(fileOpen){
		std::cout << "Checkpoint file found...\r\n";
		ckptFile.read(reinterpret_cast<char*>(&netWidth), sizeof(int));
		ckptFile.read(reinterpret_cast<char*>(&netHeight), sizeof(int));
	}
	if(train){
		if(fileOpen && (w != netWidth || h != netHeight)){
			std::cerr << "Error: Checkpoint resolution does not match training data resolution.\r\n";
			ckptFile.close();
			return;
		}
	}
	inWidth_ = netWidth;
	inHeight_ = netHeight;
	std::cout << "Initializing layers... ";
	constexpr auto wd = 0.000005f;
	auto outC = 32;
	layers_.push_back(new ConvLayer(cudnn_, batchSize_, 3, outC, 2, 2, 0, &netHeight, &netWidth, "Conv0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchSize_, outC, netHeight, netWidth, "Conv0 BatchNorm", train, wd));
	//layers_.push_back(new Swish(batchSize_*outC*netHeight*netWidth, "Conv0 Swish"));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, netHeight, netWidth, "Conv0 ReLU"));
	outC = 64;
	layers_.push_back(new ResConvLayer(cudnn_, batchSize_, 32, outC, &netHeight, &netWidth, "ResConv0", train));
	outC = 128;
	layers_.push_back(new ResConvLayer(cudnn_, batchSize_, 64, outC, &netHeight, &netWidth, "ResConv1", train));
	outC = 256;
	layers_.push_back(new ResConvLayer(cudnn_, batchSize_, 128, outC, &netHeight, &netWidth, "ResConv2", train));
	outC = 1024;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256*netHeight*netWidth, outC, "FC0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "FC0 BatchNorm", train, wd));
	//layers_.push_back(new Swish(batchSize_*outC, "FC0 Swish"));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "FC0 ReLU"));
	outC = 512;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 1024, outC, "FC1", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "FC1 BatchNorm", train, wd));
	//layers_.push_back(new Swish(batchSize_*outC, "FC1 Swish"));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "FC1 ReLU"));
	outC = 256;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 512, outC, "FC2", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "FC2 BatchNorm", train, wd));
	//layers_.push_back(new Swish(batchSize_*outC, "FC2 Swish"));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "FC2 ReLU"));
	outC = numCtrls_;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256, outC, "FC Out", train, wd));
	layers_.push_back(new Sigmoid(numButs_, batchSize_, outC, "Sigmoid"));
	for(const auto& layer : layers_){
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetOptimizerStateSize());
	}
	std::cout << "Done.\r\n";
	if(fileOpen){
		std::cout << "Loading weights/bias... ";
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
NN::~NN(){}
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