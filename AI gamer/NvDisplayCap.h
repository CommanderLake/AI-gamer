#pragma once
#include <cuda.h>
#include <driver_types.h>
//#include <helper_cuda.h>
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
inline CUdeviceptr pDevGrabBuffer = 0;
inline CUdeviceptr pDevRGBBuffer = 0;
inline CUdeviceptr nv12Buffer = 0;
inline void* pCPUBuffer = nullptr;
inline int encBitrate = 1048576 * 8;
inline int frameCunt = 0;
extern "C" cudaError launch_CudaARGB2NV12Process(int w, int h, CUdeviceptr pARGBImage, CUdeviceptr pNV12Image);
inline void FreeHost(){ if(pCPUBuffer){ cuMemFreeHost(pCPUBuffer); } }
inline void FreeGPU(){
	if(pDevRGBBuffer){ cuMemFree_v2(pDevRGBBuffer); }
	if(pDevGrabBuffer){ cuMemFree_v2(pDevGrabBuffer); }
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
	const auto result = cuMemAllocHost_v2(&pCPUBuffer, cpubuffersize);
	if(result != CUDA_SUCCESS) throw std::exception("Unable to allocate CUDA host memory");
	memset(pCPUBuffer, 0, cpubuffersize);
	buffersize = cpubuffersize;
	return pCPUBuffer;
}
inline void AllocGPU(){
	nvfbcCuda->NvFBCCudaGetMaxBufferSize(&maxBufferSize);
	if(cuMemAlloc_v2(&pDevGrabBuffer, maxBufferSize) != CUDA_SUCCESS || cuMemAlloc_v2(&pDevRGBBuffer, maxBufferSize) != CUDA_SUCCESS || cuMemAlloc_v2(&nv12Buffer, 2ui64 * (maxBufferSize / 3)) !=
		CUDA_SUCCESS){
		FreeGPU();
		throw std::exception("Unable to allocate CUDA GPU memory.");
	}
	cuMemsetD8_v2(pDevRGBBuffer, 0, maxBufferSize);
}
extern "C" cudaError ARGBtoRGB(CUdeviceptr src, CUdeviceptr dst, int n);
extern "C" cudaError ARGBtoRGBplanar(CUdeviceptr src, CUdeviceptr dst, int n);
inline unsigned char* GrabFrame(int* cuntWidth, int* cuntHeight, bool planar){
	NVFBC_CUDA_GRAB_FRAME_PARAMS fbcCudaGrabParams = {0};
	fbcCudaGrabParams.dwVersion = NVFBC_CUDA_GRAB_FRAME_PARAMS_VER;
	fbcCudaGrabParams.pCUDADeviceBuffer = reinterpret_cast<void*>(pDevGrabBuffer);
	fbcCudaGrabParams.pNvFBCFrameGrabInfo = &frameGrabInfo;
	fbcCudaGrabParams.dwFlags = NVFBC_TOCUDA_WITH_HWCURSOR | NVFBC_TOCUDA_NOWAIT;
	const auto fbcRes = nvfbcCuda->NvFBCCudaGrabFrame(&fbcCudaGrabParams);
	if(fbcRes == NVFBC_SUCCESS){
		*cuntWidth = frameGrabInfo.dwWidth;
		*cuntHeight = frameGrabInfo.dwHeight;
		if(planar) ARGBtoRGBplanar(pDevGrabBuffer, pDevRGBBuffer, *cuntWidth * *cuntHeight);
		else ARGBtoRGB(pDevGrabBuffer, pDevRGBBuffer, *cuntWidth * *cuntHeight);
		cuMemcpyDtoH_v2(pCPUBuffer, pDevRGBBuffer, static_cast<long long>(*cuntWidth) * *cuntHeight * 3ui32);
		return static_cast<unsigned char*>(pCPUBuffer);
	}
	throw std::runtime_error("NVFBC unable to capture display, result:\n\n" + NvFBCLibrary::NVFBCResultToString(fbcRes));
}
inline void SetBitrate(const int bitrate){ encBitrate = bitrate; }