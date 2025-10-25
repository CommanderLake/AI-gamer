#pragma once
#include "ConvScale.h"
struct pixARGB;
struct pixRGB;
void FreeHost();
void FreeGPU();
void DisposeNvFBC();
void AllocHost();
void AllocGPU();
int InitNvFBC();
unsigned char* GrabFrameUInt8(int* outWidth, int* outHeight, bool planar, bool toCPU);
unsigned char* GrabFrameScaleUInt8(cudnnHandle_t cudnnHandle, int* outWidth, int* outHeight, int downscaleFactor, bool planar, bool toCPU);
__half* GrabFrameScaleFP16(cudnnHandle_t cudnnHandle, int* outWidth, int* outHeight, int downscaleFactor, bool planar);
void ARGBtoRGB(unsigned char* src, unsigned char* dst, size_t n);
void ARGBtoRGBplanar(const unsigned char* src, unsigned char* dst, size_t n);