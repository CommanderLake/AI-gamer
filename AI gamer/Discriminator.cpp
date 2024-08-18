#include "Discriminator.h"
#include "FCLayer.h"
#include "LeakyReLU.h"
#include "BatchNorm.h"
#include "Sigmoid.h"
Discriminator::Discriminator(int batchSize, int inputSize, int hiddenSize, int outputSize, bool train) : cudnn_(nullptr), cublas_(nullptr), gradient_(nullptr), inputSize_(inputSize), batchSize_(batchSize), learningRate_(0.0001f), maxBufferSize_(0){
	cudnnCreate(&cudnn_);
	cublasCreate(&cublas_);
	cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH); //S
	std::cout<<"Initializing Discriminator layers...\r\n";
	constexpr auto wd = 0.001f;
	// First fully connected layer
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, inputSize_, hiddenSize, "Disc FC0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Disc BN0", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Disc LeakyReLU0"));
	// Second fully connected layer
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, hiddenSize, hiddenSize, "Disc FC1", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Disc BN1", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Disc LeakyReLU1"));
	// Output layer
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, hiddenSize, outputSize, "Disc Output FC", train, wd));
	layers_.push_back(new Sigmoid(layers_.back()->outDesc_, outputSize, batchSize_, "Disc Output Sigmoid"));
	// Allocate memory for gradient
	checkCUDA(cudaMalloc(&gradient_, outputSize*sizeof(__half)));
	// Calculate the maximum buffer size needed for saving/loading parameters
	for(const auto& layer : layers_){
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetOptimizerStateSize());
	}
}
Discriminator::~Discriminator(){
	cudnnDestroy(cudnn_);
	cublasDestroy(cublas_);
	if(gradient_) cudaFree(gradient_);
}
__half* Discriminator::Forward(__half* data){
	for(const auto& layer : layers_){ data = layer->Forward(data); }
	return data;
}
__half* Discriminator::Backward(const __half* predictions, const __half* targets){
	BCEGradient(gradient_, predictions, targets, batchSize_, 1.0f);
	auto outGrad = gradient_;
	for(int i = layers_.size(); --i>=0;){ outGrad = layers_[i]->Backward(outGrad); }
	return outGrad;
}
void Discriminator::UpdateParams(){ for(const auto& layer : layers_){ layer->UpdateParameters(learningRate_); } }
void Discriminator::SaveModel(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		float* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->SaveParameters(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for saving Discriminator: "<<filename<<"\r\n"; }
}
void Discriminator::SaveOptimizerState(const std::string& filename){
	std::ofstream file(filename, std::ios::binary);
	if(file.is_open()){
		float* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->SaveOptimizerState(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for saving optimizer state: "<<filename<<"\r\n"; }
}
void Discriminator::LoadModel(const std::string& filename){
	std::ifstream file(filename, std::ios::binary);
	if(file.is_open()){
		float* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->LoadParameters(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for loading Discriminator: "<<filename<<"\r\n"; }
}
void Discriminator::LoadOptimizerState(const std::string& filename){
	std::ifstream file(filename, std::ios::binary);
	if(file.is_open()){
		float* buffer = nullptr;
		checkCUDA(cudaMallocHost(&buffer, maxBufferSize_));
		for(const auto& layer : layers_){ layer->LoadOptimizerState(file, buffer); }
		cudaFreeHost(buffer);
		file.close();
	} else{ std::cerr<<"Unable to open file for loading optimizer state: "<<filename<<"\r\n"; }
}