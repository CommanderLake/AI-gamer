#pragma once
#include <cudnn.h>
#include <cuda_fp16.h>
#include <fstream>
#include <vector>
class Layer{
public:
	struct AdamWHalfTask{
		__half* params;
		const __half* grads;
		__half* m;
		__half* v;
		int size;
	};
	struct AdamWFloatTask{
		float* params;
		const float* grads;
		float* m;
		float* v;
		int size;
	};
	virtual ~Layer() = default;
	virtual __half* Forward(__half* data){ return nullptr; }
	virtual __half* Backward(__half* grad){ return nullptr; }
	virtual void UpdateParameters(float learningRate){}
	virtual void SaveParameters(std::ofstream& file, unsigned char* buffer){}
	virtual void LoadParameters(std::ifstream& file, unsigned char* buffer){}
	virtual void SaveOptimizerState(std::ofstream& file, unsigned char* buffer){}
	virtual void LoadOptimizerState(std::ifstream& file, unsigned char* buffer){}
	virtual size_t GetParameterSize(){ return 0; }
	virtual size_t GetOptimizerStateSize(){ return 0; }
	virtual void SetTrain(bool enable){}
	virtual void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){}
	cudnnTensorDescriptor_t outDesc_;
	size_t outNCHW_ = 0;
	std::string layerName_ = "";
	bool train_;
	size_t weightCount_ = 0;
	__half* weights_ = nullptr;
};
