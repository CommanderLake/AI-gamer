#include "NvDisplayCap.h"
#include "NvFBCLibrary.h"
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include <NvFBC\nvFBC.h>
#include <NvFBC\nvFBCCuda.h>
extern "C" cudaError ARGBtoRGB(unsigned char* src, unsigned char* dst, int n);
extern "C" cudaError ARGBtoRGBplanar(unsigned char* src, unsigned char* dst, int n);
namespace{
	NvFBCLibrary* nvfbc = nullptr;
	NvFBCCuda* nvfbcCuda = nullptr;
	int magic[] = {0x0D7BC620, 0x4C17E142, 0x5E6B5997, 0x4B5A855B};
	NvFBCCreateParams createParams = {0};
	NVFBC_CUDA_SETUP_PARAMS fbcCudaSetupParams = {0};
	unsigned long maxBufferSize = -1;
	unsigned char* pDevBufBGRA = nullptr;
	unsigned char* pDevBufRGB = nullptr;
	unsigned char* pBufCPU = nullptr;
	ConvScale* convScale_ = nullptr;
}
void FreeHost(){
	if(pBufCPU){
		cudaFreeHost(pBufCPU);
		pBufCPU = nullptr;
	}
}
void FreeGPU(){
	if(pDevBufRGB){
		cudaFreeHost(pDevBufRGB);
		pDevBufRGB = nullptr;
	}
	if(pDevBufBGRA){
		cudaFreeHost(pDevBufBGRA);
		pDevBufBGRA = nullptr;
	}
}
void DisposeNvFBC(){
	if(convScale_){
		delete convScale_;
		convScale_ = nullptr;
	}
	FreeGPU();
	FreeHost();
	if(nvfbcCuda){
		nvfbcCuda->NvFBCCudaRelease();
		nvfbcCuda = nullptr;
	}
	if(nvfbc){
		nvfbc->close();
		delete nvfbc;
		nvfbc = nullptr;
	}
}
void AllocHost(){
	const auto result = cudaMallocHost(&pBufCPU, maxBufferSize);
	if(result!=CUDA_SUCCESS) throw std::exception("Unable to allocate CUDA host memory");
	memset(pBufCPU, 0, maxBufferSize);
}
void AllocGPU(){
	if(cudaMalloc(&pDevBufBGRA, maxBufferSize)!=CUDA_SUCCESS||cudaMalloc(&pDevBufRGB, maxBufferSize*0.75f)!=CUDA_SUCCESS){
		FreeGPU();
		throw std::exception("Unable to allocate CUDA device memory.");
	}
	cudaMemset(pDevBufBGRA, 0, maxBufferSize);
	cudaMemset(pDevBufRGB, 0, maxBufferSize);
}
int InitNvFBC(){
	if(nvfbc) return 0;
	nvfbc = new NvFBCLibrary();
	nvfbc->load();
	unsigned long maxw = 0, maxh = 0;
	nvfbcCuda = static_cast<NvFBCCuda*>(nvfbc->create(NVFBC_SHARED_CUDA, &maxw, &maxh));
	nvfbcCuda->NvFBCCudaGetMaxBufferSize(&maxBufferSize);
	fbcCudaSetupParams.dwVersion = NVFBC_CUDA_SETUP_PARAMS_VER;
	fbcCudaSetupParams.eFormat = NVFBC_TOCUDA_ARGB;
	const auto fbcRes = nvfbcCuda->NvFBCCudaSetup(&fbcCudaSetupParams);
	if(fbcRes!=NVFBC_SUCCESS){
		std::cerr<<"NVFBC CUDA setup failed, result: "<<NvFBCLibrary::NVFBCResultToString(fbcRes)<<"\n";
		throw std::runtime_error("NVFBC CUDA setup failed.");
	}
	AllocGPU();
	return 0;
}
unsigned char* GrabFrameUInt8(int* outWidth, int* outHeight, bool planar, bool toCPU){
	NVFBC_CUDA_GRAB_FRAME_PARAMS fbcCudaGrabParams = {0};
	NvFBCFrameGrabInfo frameGrabInfo;
	fbcCudaGrabParams.dwVersion = NVFBC_CUDA_GRAB_FRAME_PARAMS_VER;
	fbcCudaGrabParams.pCUDADeviceBuffer = reinterpret_cast<void*>(pDevBufBGRA);
	fbcCudaGrabParams.pNvFBCFrameGrabInfo = &frameGrabInfo;
	fbcCudaGrabParams.dwFlags = NVFBC_TOCUDA_WITH_HWCURSOR|NVFBC_TOCUDA_NOWAIT;
	const auto fbcRes = nvfbcCuda->NvFBCCudaGrabFrame(&fbcCudaGrabParams);
	if(fbcRes!=NVFBC_SUCCESS){ throw std::runtime_error("NVFBC unable to capture display, result:\n\n"+NvFBCLibrary::NVFBCResultToString(fbcRes)); }
	*outWidth = frameGrabInfo.dwWidth;
	*outHeight = frameGrabInfo.dwHeight;
	if(planar){ ARGBtoRGBplanar(pDevBufBGRA, pDevBufRGB, *outWidth**outHeight); } else{ ARGBtoRGB(pDevBufBGRA, pDevBufRGB, *outWidth**outHeight); }
	if(toCPU){
		if(!pBufCPU) AllocHost();
		cudaMemcpy(pBufCPU, pDevBufRGB, *outWidth**outHeight*3, cudaMemcpyDeviceToHost);
		return pBufCPU;
	}
	return pDevBufRGB;
}
unsigned char* GrabFrameScaleUInt8(cudnnHandle_t cudnnHandle, int* outWidth, int* outHeight, int downscaleFactor, bool planar, bool toCPU){
	int grabWidth, grabHeight;
	const auto bufUInt8 = GrabFrameUInt8(&grabWidth, &grabHeight, planar, false);
	if(convScale_==nullptr||convScale_->stride_!=downscaleFactor||convScale_->inWidth_!=grabWidth||convScale_->inHeight_!=grabHeight){
		if(convScale_) delete convScale_;
		convScale_ = new ConvScale(cudnnHandle, downscaleFactor, downscaleFactor, 0, 1, 3, &grabHeight, &grabWidth);
	}
	*outWidth = convScale_->outWidth_;
	*outHeight = convScale_->outHeight_;
	convScale_->ScaleUInt8InPlaceDevice(bufUInt8);
	if(toCPU){
		if(!pBufCPU) AllocHost();
		cudaMemcpy(pBufCPU, bufUInt8, *outWidth**outHeight*3, cudaMemcpyDeviceToHost);
		return pBufCPU;
	}
	return bufUInt8;
}
__half* GrabFrameScaleFP16(cudnnHandle_t cudnnHandle, int* outWidth, int* outHeight, int downscaleFactor, bool planar){
	int grabWidth, grabHeight;
	const auto bufUInt8 = GrabFrameUInt8(&grabWidth, &grabHeight, planar, false);
	if(convScale_==nullptr||convScale_->stride_!=downscaleFactor||convScale_->inWidth_!=grabWidth||convScale_->inHeight_!=grabHeight){
		if(convScale_) delete convScale_;
		convScale_ = new ConvScale(cudnnHandle, downscaleFactor, downscaleFactor, 0, 1, 3, &grabHeight, &grabWidth);
	}
	*outWidth = convScale_->outWidth_;
	*outHeight = convScale_->outHeight_;
	return convScale_->ScaleUInt8ToFP16Device(bufUInt8);
}