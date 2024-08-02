#pragma once
#include "ConvScale.h"
#include "NvFBCLibrary.h"
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <NvFBC\nvFBC.h>
#include <NvFBC\nvFBCCuda.h>
struct pixARGB;
struct pixRGB;
inline NvFBCLibrary* nvfbc = nullptr;
inline NvFBCCuda* nvfbcCuda = nullptr;
inline int magic[] = {0x0D7BC620, 0x4C17E142, 0x5E6B5997, 0x4B5A855B};
inline NvFBCCreateParams createParams = {0};
inline NVFBC_CUDA_SETUP_PARAMS fbcCudaSetupParams = {0};
inline unsigned long maxBufferSize = -1;
inline unsigned long long cpubuffersize = 0;
inline unsigned char* pDevBufBGRA = nullptr;
inline unsigned char* pDevBufRGB = nullptr;
inline void* pBufCPU = nullptr;
inline ConvScale* convScale_ = nullptr;
inline int InitNvFBC(){
	nvfbc = new NvFBCLibrary();
	nvfbc->load();
	unsigned long maxw = 0, maxh = 0;
	nvfbcCuda = static_cast<NvFBCCuda*>(nvfbc->create(NVFBC_SHARED_CUDA, &maxw, &maxh));
	nvfbcCuda->NvFBCCudaGetMaxBufferSize(&maxBufferSize);
	fbcCudaSetupParams.dwVersion = NVFBC_CUDA_SETUP_PARAMS_VER;
	fbcCudaSetupParams.eFormat = NVFBC_TOCUDA_ARGB;
	const auto fbcRes = nvfbcCuda->NvFBCCudaSetup(&fbcCudaSetupParams);
	if(fbcRes != NVFBC_SUCCESS){
		std::cerr << "NVFBC CUDA setup failed, result: " << NvFBCLibrary::NVFBCResultToString(fbcRes) << std::endl;
		throw std::runtime_error("NVFBC CUDA setup failed.");
	}
	return 0;
}
inline void FreeHost(){ if(pBufCPU){ cudaFreeHost(pBufCPU); } }
inline void FreeGPU(){
	if(pDevBufRGB){ cudaFreeHost(pDevBufRGB); }
	if(pDevBufBGRA){ cudaFreeHost(pDevBufBGRA); }
}
inline void DisposeNvFBC(){
	if(convScale_) delete convScale_;
	FreeHost();
	FreeGPU();
	nvfbcCuda->NvFBCCudaRelease();
	if(nvfbc){
		nvfbc->close();
		delete nvfbc;
		nvfbc = nullptr;
	}
}
inline void AllocHost(unsigned long long& buffersize){
	cpubuffersize = max(buffersize, maxBufferSize);
	const auto result = cudaMallocHost(&pBufCPU, cpubuffersize);
	if(result != CUDA_SUCCESS) throw std::exception("Unable to allocate CUDA host memory");
	memset(pBufCPU, 0, cpubuffersize);
	buffersize = cpubuffersize;
}
inline void AllocGPU(){
	nvfbcCuda->NvFBCCudaGetMaxBufferSize(&maxBufferSize);
	if(cudaMalloc(&pDevBufBGRA, maxBufferSize) != CUDA_SUCCESS || cudaMalloc(&pDevBufRGB, maxBufferSize) != CUDA_SUCCESS){
		FreeGPU();
		throw std::exception("Unable to allocate CUDA device memory.");
	}
	cudaMemset(pDevBufRGB, 0, maxBufferSize);
}
extern "C" cudaError ARGBtoRGB(unsigned char* src, unsigned char* dst, int n);
extern "C" cudaError ARGBtoRGBplanar(unsigned char* src, unsigned char* dst, int n);
inline unsigned char* GrabFrameInt8(int* outWidth, int* outHeight, bool planar, bool toCPU){
	NVFBC_CUDA_GRAB_FRAME_PARAMS fbcCudaGrabParams = {0};
	NvFBCFrameGrabInfo frameGrabInfo;
	fbcCudaGrabParams.dwVersion = NVFBC_CUDA_GRAB_FRAME_PARAMS_VER;
	fbcCudaGrabParams.pCUDADeviceBuffer = reinterpret_cast<void*>(pDevBufBGRA);
	fbcCudaGrabParams.pNvFBCFrameGrabInfo = &frameGrabInfo;
	fbcCudaGrabParams.dwFlags = NVFBC_TOCUDA_WITH_HWCURSOR | NVFBC_TOCUDA_NOWAIT;
	const auto fbcRes = nvfbcCuda->NvFBCCudaGrabFrame(&fbcCudaGrabParams);
	if(fbcRes == NVFBC_SUCCESS){
		*outWidth = frameGrabInfo.dwWidth;
		*outHeight = frameGrabInfo.dwHeight;
		if(planar) ARGBtoRGBplanar(pDevBufBGRA, pDevBufRGB, *outWidth**outHeight);
		else ARGBtoRGB(pDevBufBGRA, pDevBufRGB, *outWidth**outHeight);
		if(toCPU){
			cudaMemcpy(pBufCPU, pDevBufRGB, *outWidth**outHeight*3, cudaMemcpyDeviceToHost);
			return static_cast<unsigned char*>(pBufCPU);
		}
		return pDevBufRGB;
	}
	throw std::runtime_error("NVFBC unable to capture display, result:\n\n" + NvFBCLibrary::NVFBCResultToString(fbcRes));
}
inline unsigned char* GrabFrameScaleInt8(cudnnHandle_t cudnnHandle, int* outWidth, int* outHeight, int downscaleFactor, bool planar){
	int grabWidth, grabHeight;
	const auto frameInt8 = GrabFrameInt8(&grabWidth, &grabHeight, planar, false);
	if(convScale_ == nullptr || convScale_->scaleFactor_ != downscaleFactor ||
		convScale_->inWidth_ != grabWidth || convScale_->inHeight_ != grabHeight){
		if(convScale_ != nullptr){
			delete convScale_;
		}
		convScale_ = new ConvScale(cudnnHandle, downscaleFactor, grabWidth, grabHeight);
	}
	*outWidth = convScale_->outWidth_;
	*outHeight = convScale_->outHeight_;
	convScale_->ScaleInPlace(frameInt8);
	return frameInt8;
}
inline __half* GrabFrameScaleFP16(cudnnHandle_t cudnnHandle, int* outWidth, int* outHeight, int downscaleFactor, bool planar){
	int grabWidth, grabHeight;
	const auto frameInt8 = GrabFrameInt8(&grabWidth, &grabHeight, planar, false);
	if(convScale_ == nullptr || convScale_->scaleFactor_ != downscaleFactor ||
		convScale_->inWidth_ != grabWidth || convScale_->inHeight_ != grabHeight){
		if(convScale_ != nullptr){
			delete convScale_;
		}
		convScale_ = new ConvScale(cudnnHandle, downscaleFactor, grabWidth, grabHeight);
	}
	*outWidth = convScale_->outWidth_;
	*outHeight = convScale_->outHeight_;
	return convScale_->ScaleToHalf(frameInt8);
}