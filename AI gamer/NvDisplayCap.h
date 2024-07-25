#pragma once
#include <cuda.h>
#include <driver_types.h>
//#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <NvFBC\nvFBC.h>
#include <NvFBC\nvFBCCuda.h>
#include "NvFBCLibrary.h"
struct pixARGB;
struct pixRGB;
inline CUcontext cudaCtx = nullptr;
inline NvFBCLibrary* nvfbc = nullptr;
inline NvFBCCuda* nvfbcCuda = nullptr;
inline int magic[] = {0x0D7BC620, 0x4C17E142, 0x5E6B5997, 0x4B5A855B};
inline NvFBCCreateParams createParams = {0};
inline NVFBC_CUDA_SETUP_PARAMS fbcCudaSetupParams = {0};
inline NvFBCFrameGrabInfo frameGrabInfo;
inline unsigned long maxBufferSize = -1;
inline unsigned long long cpubuffersize = 0;
inline unsigned char* pDevBGRABuffer = nullptr;
inline unsigned char* pDevRGBBuffer = nullptr;
inline void* pCPUBuffer = nullptr;
inline int frameCunt = 0;
inline void FreeHost(){ if(pCPUBuffer){ cudaFreeHost(pCPUBuffer); } }
inline void FreeGPU(){
	if(pDevRGBBuffer){ cudaFreeHost(pDevRGBBuffer); }
	if(pDevBGRABuffer){ cudaFreeHost(pDevBGRABuffer); }
}
inline void DisposeNvFBC(){
	FreeHost();
	FreeGPU();
	nvfbcCuda->NvFBCCudaRelease();
	if(nvfbc){
		nvfbc->close();
		delete nvfbc;
		nvfbc = nullptr;
	}
}
inline int InitNvFBC(){
	nvfbc = new NvFBCLibrary();
	nvfbc->load();
	unsigned long maxw = 0, maxh = 0;
	nvfbcCuda = static_cast<NvFBCCuda*>(nvfbc->create(NVFBC_SHARED_CUDA, &maxw, &maxh));
	cuCtxPopCurrent(&cudaCtx);
	cuCtxPushCurrent(cudaCtx);
	nvfbcCuda->NvFBCCudaGetMaxBufferSize(&maxBufferSize);
	fbcCudaSetupParams.dwVersion = NVFBC_CUDA_SETUP_PARAMS_VER;
	fbcCudaSetupParams.eFormat = NVFBC_TOCUDA_ARGB;
	const auto fbcRes = nvfbcCuda->NvFBCCudaSetup(&fbcCudaSetupParams);
	if(fbcRes != NVFBC_SUCCESS) throw std::runtime_error("NVFBC CUDA setup failed, result:\n\n" + NvFBCLibrary::NVFBCResultToString(fbcRes));
	return 0;
}
inline void* AllocHost(unsigned long long& buffersize){
	cpubuffersize = buffersize ? max(buffersize, maxBufferSize) : maxBufferSize;
	const auto result = cudaMallocHost(&pCPUBuffer, cpubuffersize);
	if(result != CUDA_SUCCESS) throw std::exception("Unable to allocate CUDA host memory");
	memset(pCPUBuffer, 0, cpubuffersize);
	buffersize = cpubuffersize;
	return pCPUBuffer;
}
inline void AllocGPU(){
	nvfbcCuda->NvFBCCudaGetMaxBufferSize(&maxBufferSize);
	if(cudaMalloc(&pDevBGRABuffer, maxBufferSize) != CUDA_SUCCESS || cudaMalloc(&pDevRGBBuffer, maxBufferSize) != CUDA_SUCCESS){
		FreeGPU();
		throw std::exception("Unable to allocate CUDA GPU memory.");
	}
	cudaMemset(pDevRGBBuffer, 0, maxBufferSize);
}
extern "C" cudaError ARGBtoRGB(unsigned char* src, unsigned char* dst, int n);
extern "C" cudaError ARGBtoRGBplanar(unsigned char* src, unsigned char* dst, int n);
inline unsigned char* GrabFrame(int* cuntWidth, int* cuntHeight, bool planar, bool toCPU){
	NVFBC_CUDA_GRAB_FRAME_PARAMS fbcCudaGrabParams = {0};
	fbcCudaGrabParams.dwVersion = NVFBC_CUDA_GRAB_FRAME_PARAMS_VER;
	fbcCudaGrabParams.pCUDADeviceBuffer = reinterpret_cast<void*>(pDevBGRABuffer);
	fbcCudaGrabParams.pNvFBCFrameGrabInfo = &frameGrabInfo;
	fbcCudaGrabParams.dwFlags = NVFBC_TOCUDA_WITH_HWCURSOR | NVFBC_TOCUDA_NOWAIT;
	const auto fbcRes = nvfbcCuda->NvFBCCudaGrabFrame(&fbcCudaGrabParams);
	if(fbcRes == NVFBC_SUCCESS){
		*cuntWidth = frameGrabInfo.dwWidth;
		*cuntHeight = frameGrabInfo.dwHeight;
		if(planar) ARGBtoRGBplanar(pDevBGRABuffer, pDevRGBBuffer, *cuntWidth**cuntHeight);
		else ARGBtoRGB(pDevBGRABuffer, pDevRGBBuffer, *cuntWidth**cuntHeight);
		if(toCPU){
			cudaMemcpy(pCPUBuffer, pDevRGBBuffer, *cuntWidth**cuntHeight*3, cudaMemcpyDeviceToHost);
			return static_cast<unsigned char*>(pCPUBuffer);
		}
		return pDevRGBBuffer;
	}
	throw std::runtime_error("NVFBC unable to capture display, result:\n\n" + NvFBCLibrary::NVFBCResultToString(fbcRes));
}