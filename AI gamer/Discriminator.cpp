#include "Discriminator.h"
#include "FCLayer.h"
#include "LeakyReLU.h"
#include "BatchNorm.h"
#include "Dropout.h"
#include "Sigmoid.h"
Discriminator::Discriminator(int batchSize, int inputSize, bool train) : cudnn_(nullptr), cublas_(nullptr), gradient_(nullptr), inputSize_(inputSize), batchSize_(batchSize), learningRate_(0.000005f), maxBufferSize_(0){
	cudnnCreate(&cudnn_);
	cublasCreate(&cublas_);
	cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH); //S
	std::cout<<"Initializing Discriminator layers... ";
	constexpr auto wd = 0.000005f;
	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, inputSize_, 128, "Disc FC0", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Disc BN0", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Disc LeakyReLU0"));
	layers_.push_back(new Dropout(cudnn_, layers_.back()->outDesc_, 0.3f, "Drop0"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 128, 64, "Disc FC1", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Disc BN1", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Disc LeakyReLU1"));
	layers_.push_back(new Dropout(cudnn_, layers_.back()->outDesc_, 0.3f, "Drop1"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 64, 32, "Disc FC2", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Disc BN2", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Disc LeakyReLU2"));
	layers_.push_back(new Dropout(cudnn_, layers_.back()->outDesc_, 0.3f, "Drop2"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 32, 16, "Disc FC3", train, wd));
	layers_.push_back(new BatchNorm(cudnn_, layers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Disc BN3", train, wd));
	layers_.push_back(new LeakyReLU(layers_.back()->outDesc_, "Disc LeakyReLU3"));

	layers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, 16, 1, "Disc Output FC", train, wd));
	layers_.push_back(new Sigmoid(layers_.back()->outDesc_, 1, batchSize_, "Disc Output Sigmoid"));
	checkCUDA(cudaMalloc(&gradient_, batchSize_*sizeof(__half)));
	for(const auto& layer : layers_){
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetParameterSize());
		maxBufferSize_ = std::max(maxBufferSize_, layer->GetOptimizerStateSize());
	}
	std::cout << "Done.\r\n";
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
	BCEGradient(gradient_, predictions, targets, batchSize_, 100.0f);
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