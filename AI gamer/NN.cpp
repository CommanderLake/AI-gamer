#include "NN.h"
#include "ConvLayer.h"
#include "FCLayer.h"
#include "LeakyReLU.h"
#include "BatchNorm.h"
#include "Dropout.h"
#include "Sigmoid.h"
NN::NN(int w, int h, bool train): cudnn_(nullptr), cublas_(nullptr), batchSize_(64), seqLength_(1), inWidth_(0), inHeight_(0), learningRate_(0.00001f), maxBufferSize_(0){
	if(!train) batchSize_ = seqLength_ = 1;
	std::ifstream ckptFile(ckptFileName, std::ios::binary);
	const bool fileOpen = ckptFile.is_open();
	int modelWidth = w;
	int modelHeight = h;
	if(fileOpen){
		std::cout << "Checkpoint file found...\r\n";
		ckptFile.read(reinterpret_cast<char*>(&modelWidth), sizeof(int));
		ckptFile.read(reinterpret_cast<char*>(&modelHeight), sizeof(int));
	}
	if(train){
		if(fileOpen && (w != modelWidth || h != modelHeight)){
			std::cerr << "Error: Checkpoint resolution does not match training data resolution.\r\n";
			ckptFile.close();
			return;
		}
	}
	inWidth_ = modelWidth;
	inHeight_ = modelHeight;
	auto inputSize = modelWidth*modelHeight*3;
	cudnnCreate(&cudnn_);
	cublasCreate(&cublas_);
	cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);//S
	std::cout << "Initializing layers... ";
	constexpr auto wd = 0.000005f;
	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 3, 32, 4, 2, 0, &modelWidth, &modelHeight, &inputSize, "Conv0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv0 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv0 Leaky ReLU"));

	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 32, 64, 4, 2, 0, &modelWidth, &modelHeight, &inputSize, "Conv1", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv1 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv1 Leaky ReLU"));

	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 64, 128, 4, 2, 0, &modelWidth, &modelHeight, &inputSize, "Conv2", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv2 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv2 Leaky ReLU"));

	layers_.push_back(new ConvLayer(cudnn_, cublas_, batchSize_, 128, 256, 4, 2, 0, &modelWidth, &modelHeight, &inputSize, "Conv3", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_SPATIAL, batchSize_, "Conv3 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Conv3 Leaky ReLU"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, inputSize, 1024, "FC0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC0 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC0 Leaky ReLU"));
	layers_.push_back(new Dropout(cudnn_, layers_.back()->outDesc_, 0.2f, "Drop0"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 1024, 512, "FC1", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC1 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC1 Leaky ReLU"));
	layers_.push_back(new Dropout(cudnn_, layers_.back()->outDesc_, 0.2f, "Drop1"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 512, 256, "FC2", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC2 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC2 Leaky ReLU"));
	layers_.push_back(new Dropout(cudnn_, layers_.back()->outDesc_, 0.2f, "Drop2"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256, 128, "FC3", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "FC3 BatchNorm", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "FC3 Leaky ReLU"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 128, numCtrls_, "FC Out", train, wd));
	layers_.push_back(new Sigmoid(layers_.back()->outDesc_, numButs_, batchSize_, "Sigmoid"));
	checkCUDA(cudaMalloc(&gradient_, ctrlBatchSize_*sizeof(__half)));
	checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchFloat_), ctrlBatchSize_*sizeof(float)));
	checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&ctrlBatchHalf_), ctrlBatchSize_*sizeof(__half)));
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
NN::~NN(){
	cudnnDestroy(cudnn_);
	cublasDestroy(cublas_);
	if(gradient_) cudaFree(gradient_);
	if(ctrlBatchFloat_) cudaFreeHost(ctrlBatchFloat_);
	if(ctrlBatchHalf_) cudaFreeHost(ctrlBatchHalf_);
}
__half* NN::Forward(__half* data){
	for(const auto layer : layers_){
		data = layer->Forward(data);
	}
	return data;
}
__half* NN::Backward(__half* grad){
	auto outGrad = grad;
	for(int i = layers_.size(); --i >= 0; ){
		outGrad = layers_[i]->Backward(outGrad);
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