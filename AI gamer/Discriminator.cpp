#include "Discriminator.h"
#include "Activate.h"
#include "FCLayer.h"
#include "LeakyReLU.h"
#include "BatchNorm.h"
#include "Dropout.h"
#include "Sigmoid.h"
Discriminator::Discriminator(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inputSize, bool train) : cudnn_(cudnnHandle), cublas_(cublasHandle), gradient_(nullptr), inputSize_(inputSize), batchSize_(batchSize), learningRate_(0.000005f), maxBufferSize_(0){
	std::cout<<"Initializing Discriminator layers... ";
	constexpr auto wd = 0.0000001f;
	int outC = 128;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, inputSize, outC, "Disc FC0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Disc BN0", train, wd));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Disc ReLU0"));
	outC = 256;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 128, outC, "Disc FC1", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Disc BN1", train, wd));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Disc ReLU1"));
	outC = 512;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256, outC, "Disc FC2", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Disc BN2", train, wd));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Disc ReLU2"));
	outC = 256;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 512, outC, "Disc FC3", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Disc BN3", train, wd));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Disc ReLU3"));
	outC = 128;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 256, outC, "Disc FC4", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Disc BN4", train, wd));
	layers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Disc ReLU4"));
	outC = inputSize_;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 128, outC, "Disc FC Out", train, wd));
	layers_.push_back(new Sigmoid(numButs_, batchSize_, outC, "Disc Sigmoid Out"));
	checkCUDA(cudaMalloc(&gradient_, inputSize_*batchSize_*sizeof(__half)));
	for(const auto& layer : layers_){
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetOptimizerStateSize());
	}
	std::cout << "Done.\r\n";
}
Discriminator::~Discriminator(){
	if(gradient_) cudaFree(gradient_);
}
__half* Discriminator::Forward(__half* data){
	for(const auto& layer : layers_){ data = layer->Forward(data); }
	return data;
}
__half* Discriminator::Backward(const __half* predictions, const __half* targets){
	DiscriminatorGradient(gradient_, predictions, targets, inputSize_*batchSize_, numCtrls_, numButs_, 1.0f, 1.0f, 32.0f);
	auto outGrad = gradient_;
	for(int i = layers_.size(); --i>=0;){ outGrad = layers_[i]->Backward(outGrad); }
	return outGrad;
}
void Discriminator::UpdateParams(){ for(const auto& layer : layers_){ layer->UpdateParameters(learningRate_); } }
void Discriminator::SaveModel(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->SaveParameters(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for saving Discriminator: "<<filename<<"\r\n"; }
}
void Discriminator::SaveOptimizerState(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->SaveOptimizerState(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for saving optimizer state: "<<filename<<"\r\n"; }
}
void Discriminator::LoadModel(const std::string& filename){
	std::ifstream file(filename, std::ios::binary);
	if(file.is_open()){
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->LoadParameters(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for loading Discriminator: "<<filename<<"\r\n"; }
}
void Discriminator::LoadOptimizerState(const std::string& filename){
	std::ifstream file(filename, std::ios::binary);
	if(file.is_open()){
		unsigned char* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->LoadOptimizerState(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for loading optimizer state: "<<filename<<"\r\n"; }
}