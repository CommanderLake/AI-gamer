#include "LeakyReLU.h"
#include "common.h"
LeakyReLU::LeakyReLU(cudnnTensorDescriptor_t outDesc, const char* layerName): slope_(1.0f/64.0f){
	layerName_ = layerName;
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outNCHW_ = n*c*h*w;
}
LeakyReLU::~LeakyReLU(){
}
__half* LeakyReLU::Forward(__half* data){
	data_ = data;
	LeakyRelu(data, outNCHW_, slope_);
	cudaDeviceSynchronize();
	if(const cudaError_t err = cudaGetLastError(); err != cudaSuccess){
		printf("CUDA error in Forward: %s\n", cudaGetErrorString(err));
	}
	return data;
}
__half* LeakyReLU::Backward(__half* grad){
	LeakyReluBackward(grad, data_, outNCHW_, slope_);
	cudaDeviceSynchronize();
	if(const cudaError_t err = cudaGetLastError(); err != cudaSuccess){
		printf("CUDA error in Backward: %s\n", cudaGetErrorString(err));
	}
	return grad;
}