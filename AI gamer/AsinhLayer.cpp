#include "AsinhLayer.h"
#include "CuCommon.cuh"
AsinhLayer::AsinhLayer(const int batchSize, const int channels, const int height, const int width, const float alpha, const char* layerName) : batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width), alpha_(alpha){
	layerName_ = layerName;
	outNCHW_ = batchSize_*outC_*outHeight_*outWidth_;
}
__half* AsinhLayer::Forward(__half* data){
	data_ = data;
	AsinhForward(data, data, outNCHW_, alpha_);
	return data;
}
__half* AsinhLayer::Backward(__half* grad){
	AsinhBackward(grad, data_, outNCHW_, alpha_);
	return grad;
}
void AsinhLayer::SetTrain(const bool enable){
	train_ = enable;
}