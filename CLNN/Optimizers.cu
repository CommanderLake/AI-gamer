#define __CUDACC__
#include "CuCommon.h"
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#define EPSILON_OPT 1e-6f
#define BETA1_F 0.9f
#define BETA2_F 0.95f
#define CLIP 1.0f
__global__ void SGDHalfKernel(__half* params, const __half* grads, const int size, const float learningRate, const float weightDecay){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		params[idx] *= 1.0f - weightDecay;
		const float gradClipped = fmaxf(fminf(__half2float(grads[idx]), CLIP), -CLIP);
		params[idx] = __float2half(__half2float(params[idx]) - learningRate*gradClipped);
	}
}
void SGDHalf(__half* params, const __half* grads, const int size, const float learningRate, const float weightDecay){
	size_t blocks, tpb = 128;
	GetLaunchConfigGridStride(size, blocks, tpb);
	SGDHalfKernel<<<blocks, tpb>>>(params, grads, size, learningRate, weightDecay);
	checkCUDA(cudaGetLastError());
}
__global__ void SGDFloatKernel(float* params, const float* grads, const int size, const float learningRate, const float weightDecay){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		params[idx] *= 1.0f - weightDecay;
		const float gradClipped = fmaxf(fminf(grads[idx], CLIP), -CLIP);
		params[idx] -= learningRate*gradClipped;
	}
}
void SGDFloat(float* params, const float* grads, const int size, const float learningRate, const float weightDecay){
	size_t blocks, tpb = 128;
	GetLaunchConfigGridStride(size, blocks, tpb);
	SGDFloatKernel<<<blocks, tpb>>>(params, grads, size, learningRate, weightDecay);
	checkCUDA(cudaGetLastError());
}
__global__ void AdamwKernelFloat(float* __restrict__ params, const float* __restrict__ grads, float* __restrict__ m, float* __restrict__ v, const float lr, const int t, const float wd, const int n){
	__shared__ float sBiasCorrection2;
	__shared__ float sLrT;
	__shared__ float sBeta1Complement;
	__shared__ float sBeta3Complement;
	if(threadIdx.x == 0){
		const float biasCorrection1 = 1.0f - powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f - powf(BETA2_F, t);
		sLrT = lr/biasCorrection1;
		sBeta1Complement = 1.0f - BETA1_F;
		sBeta3Complement = 1.0f - BETA2_F;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	const float lrWeightDecay = lr*wd;
#pragma unroll 4
	for(int i = idx; i < n/4; i += stride){
		float4 params4 = reinterpret_cast<const float4*>(params)[i];
		const float4 grads4 = reinterpret_cast<const float4*>(grads)[i];
		const float4 m4 = reinterpret_cast<const float4*>(m)[i];
		const float4 v4 = reinterpret_cast<const float4*>(v)[i];
		// Clip gradients
		float4 gradClipped;
		gradClipped.x = fmaxf(fminf(grads4.x, CLIP), -CLIP);
		gradClipped.y = fmaxf(fminf(grads4.y, CLIP), -CLIP);
		gradClipped.z = fmaxf(fminf(grads4.z, CLIP), -CLIP);
		gradClipped.w = fmaxf(fminf(grads4.w, CLIP), -CLIP);
		// Update momentum
		float4 mNew;
		mNew.x = BETA1_F*m4.x + sBeta1Complement*gradClipped.x;
		mNew.y = BETA1_F*m4.y + sBeta1Complement*gradClipped.y;
		mNew.z = BETA1_F*m4.z + sBeta1Complement*gradClipped.z;
		mNew.w = BETA1_F*m4.w + sBeta1Complement*gradClipped.w;
		// Update velocity
		float4 vNew;
		vNew.x = BETA2_F*v4.x + sBeta3Complement*gradClipped.x*gradClipped.x;
		vNew.y = BETA2_F*v4.y + sBeta3Complement*gradClipped.y*gradClipped.y;
		vNew.z = BETA2_F*v4.z + sBeta3Complement*gradClipped.z*gradClipped.z;
		vNew.w = BETA2_F*v4.w + sBeta3Complement*gradClipped.w*gradClipped.w;
		// Compute updates
		float4 update;
		update.x = sLrT*mNew.x/(sqrtf(vNew.x/sBiasCorrection2) + EPSILON_OPT);
		update.y = sLrT*mNew.y/(sqrtf(vNew.y/sBiasCorrection2) + EPSILON_OPT);
		update.z = sLrT*mNew.z/(sqrtf(vNew.z/sBiasCorrection2) + EPSILON_OPT);
		update.w = sLrT*mNew.w/(sqrtf(vNew.w/sBiasCorrection2) + EPSILON_OPT);
		// Apply updates
		const float paramX = params4.x;
		const float paramY = params4.y;
		const float paramZ = params4.z;
		const float paramW = params4.w;
		params4.x = paramX - update.x - lrWeightDecay*paramX;
		params4.y = paramY - update.y - lrWeightDecay*paramY;
		params4.z = paramZ - update.z - lrWeightDecay*paramZ;
		params4.w = paramW - update.w - lrWeightDecay*paramW;
		// Store results
		reinterpret_cast<float4*>(params)[i] = params4;
		reinterpret_cast<float4*>(m)[i] = mNew;
		reinterpret_cast<float4*>(v)[i] = vNew;
	}
	// Handle remaining elements
	const int remainStart = n/4*4;
	for(int i = remainStart + idx; i < n; i += stride){
		const float grad = fmaxf(fminf(grads[i], CLIP), -CLIP);
		const float mVal = BETA1_F*m[i] + sBeta1Complement*grad;
		const float vVal = BETA2_F*v[i] + sBeta3Complement*grad*grad;
		const float param = params[i];
		const float update = sLrT*mVal/(sqrtf(vVal/sBiasCorrection2) + EPSILON_OPT);
		params[i] = param - update - lrWeightDecay*param;
		m[i] = mVal;
		v[i] = vVal;
	}
}
void AdamWFloat(float* params, const float* grads, float* m, float* v, const float learningRate, const int t, const float weightDecay, const int size){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	AdamwKernelFloat<<<blocks, tpb>>>(params, grads, m, v, learningRate, t, weightDecay, size);
	checkCUDA(cudaGetLastError());
}
__global__ void AdamwKernelHalf(__half* __restrict__ params, const __half* __restrict__ grads, __half* __restrict__ m, __half* __restrict__ v, const float lr, const int t, const float wd, const int n){
	// Shared memory for frequently used constants
	__shared__ float sBiasCorrection1;
	__shared__ float sBiasCorrection2;
	__shared__ float sLrT;
	__shared__ float sBeta1Complement;
	__shared__ float sBeta3Complement;
	__shared__ float sLrWeightDecay;
	// First thread in block computes constants
	if(threadIdx.x == 0){
		sBiasCorrection1 = 1.0f - powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f - powf(BETA2_F, t);
		sLrT = lr/sBiasCorrection1;
		sBeta1Complement = 1.0f - BETA1_F;
		sBeta3Complement = 1.0f - BETA2_F;
		sLrWeightDecay = lr*wd;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	// Process 4 half2 elements (8 total elements) at once
#pragma unroll 4
	for(int i = idx; i < n/8; i += stride){
		// Load 4 half2 pairs (8 elements)
		const auto paramsPtr = reinterpret_cast<__half2*>(params + 8*i);
		const auto gradsPtr = reinterpret_cast<const __half2*>(grads + 8*i);
		const auto mPtr = reinterpret_cast<__half2*>(m + 8*i);
		const auto vPtr = reinterpret_cast<__half2*>(v + 8*i);
		// Load 4 pairs of values
		__half2 paramsH2[4], gradsH2[4], mH2[4], vH2[4];
#pragma unroll
		for(int j = 0; j < 4; j++){
			paramsH2[j] = paramsPtr[j];
			gradsH2[j] = gradsPtr[j];
			mH2[j] = mPtr[j];
			vH2[j] = vPtr[j];
		}
		// Convert to float2 for computations
		float2 paramsF2[4], gradsF2[4], mF2[4], vF2[4];
#pragma unroll
		for(int j = 0; j < 4; j++){
			paramsF2[j] = __half22float2(paramsH2[j]);
			gradsF2[j] = __half22float2(gradsH2[j]);
			mF2[j] = __half22float2(mH2[j]);
			vF2[j] = __half22float2(vH2[j]);
		}
		// Process each pair
#pragma unroll
		for(int j = 0; j < 4; j++){
			// Clip gradients
			gradsF2[j].x = fmaxf(fminf(gradsF2[j].x, CLIP), -CLIP);
			gradsF2[j].y = fmaxf(fminf(gradsF2[j].y, CLIP), -CLIP);
			// Update momentum
			mF2[j].x = BETA1_F*mF2[j].x + sBeta1Complement*gradsF2[j].x;
			mF2[j].y = BETA1_F*mF2[j].y + sBeta1Complement*gradsF2[j].y;
			// Update velocity
			vF2[j].x = BETA2_F*vF2[j].x + sBeta3Complement*gradsF2[j].x*gradsF2[j].x;
			vF2[j].y = BETA2_F*vF2[j].y + sBeta3Complement*gradsF2[j].y*gradsF2[j].y;
			// Compute denominator
			float2 denom;
			denom.x = sqrtf(vF2[j].x/sBiasCorrection2) + EPSILON_OPT;
			denom.y = sqrtf(vF2[j].y/sBiasCorrection2) + EPSILON_OPT;
			// Update parameters
			const float updateX = sLrT*mF2[j].x/denom.x;
			const float updateY = sLrT*mF2[j].y/denom.y;
			const float paramX = paramsF2[j].x;
			const float paramY = paramsF2[j].y;
			paramsF2[j].x = paramX - updateX - sLrWeightDecay*paramX;
			paramsF2[j].y = paramY - updateY - sLrWeightDecay*paramY;
			// Convert back to half2
			paramsH2[j] = __float22half2_rn(paramsF2[j]);
			mH2[j] = __float22half2_rn(mF2[j]);
			vH2[j] = __float22half2_rn(vF2[j]);
		}
		// Store results
#pragma unroll
		for(int j = 0; j < 4; j++){
			paramsPtr[j] = paramsH2[j];
			mPtr[j] = mH2[j];
			vPtr[j] = vH2[j];
		}
	}
	// Handle remaining elements
	const int remainStart = n/8*8;
	for(int i = remainStart + idx; i < n; i += stride){
		const float grad = fmaxf(fminf(__half2float(grads[i]), CLIP), -CLIP);
		const float mVal = BETA1_F*__half2float(m[i]) + sBeta1Complement*grad;
		const float vVal = BETA2_F*__half2float(v[i]) + sBeta3Complement*grad*grad;
		const float param = __half2float(params[i]);
		const float denom = sqrtf(vVal/sBiasCorrection2) + EPSILON_OPT;
		const float update = sLrT*mVal/denom;
		params[i] = __float2half(param - update - sLrWeightDecay*param);
		m[i] = __float2half(mVal);
		v[i] = __float2half(vVal);
	}
}
void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, const float lr, const int t, const float weightDecay, const int size){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	AdamwKernelHalf<<<blocks, tpb>>>(params, grads, m, v, lr, t, weightDecay, size);
	checkCUDA(cudaGetLastError());
}

__global__ void AdamwKernelHalfMulti(const AdamWHalfTask* __restrict__ tasks, const int taskCount, const float lr, const int t){
	__shared__ float sBiasCorrection1;
	__shared__ float sBiasCorrection2;
	__shared__ float sLrT;
	__shared__ float sBeta1Complement;
	__shared__ float sBeta3Complement;
	if(threadIdx.x == 0){
		sBiasCorrection1 = 1.0f - powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f - powf(BETA2_F, t);
		sLrT = lr/sBiasCorrection1;
		sBeta1Complement = 1.0f - BETA1_F;
		sBeta3Complement = 1.0f - BETA2_F;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	for(int taskIdx = 0; taskIdx < taskCount; ++taskIdx){
		const AdamWHalfTask task = tasks[taskIdx];
		const int n = task.size;
		const float sLrWeightDecay = lr*task.weightDecay;
#pragma unroll 4
		for(int i = idx; i < n/8; i += stride){
			const auto paramsPtr = reinterpret_cast<__half2*>(task.params + 8*i);
			const auto gradsPtr = reinterpret_cast<const __half2*>(task.grads + 8*i);
			const auto mPtr = reinterpret_cast<__half2*>(task.m + 8*i);
			const auto vPtr = reinterpret_cast<__half2*>(task.v + 8*i);
			__half2 paramsH2[4], gradsH2[4], mH2[4], vH2[4];
#pragma unroll
			for(int j = 0; j < 4; j++){
				paramsH2[j] = paramsPtr[j];
				gradsH2[j] = gradsPtr[j];
				mH2[j] = mPtr[j];
				vH2[j] = vPtr[j];
			}
			float2 paramsF2[4], gradsF2[4], mF2[4], vF2[4];
#pragma unroll
			for(int j = 0; j < 4; j++){
				paramsF2[j] = __half22float2(paramsH2[j]);
				gradsF2[j] = __half22float2(gradsH2[j]);
				mF2[j] = __half22float2(mH2[j]);
				vF2[j] = __half22float2(vH2[j]);
			}
#pragma unroll
			for(int j = 0; j < 4; j++){
				gradsF2[j].x = fmaxf(fminf(gradsF2[j].x, CLIP), -CLIP);
				gradsF2[j].y = fmaxf(fminf(gradsF2[j].y, CLIP), -CLIP);
				mF2[j].x = BETA1_F*mF2[j].x + sBeta1Complement*gradsF2[j].x;
				mF2[j].y = BETA1_F*mF2[j].y + sBeta1Complement*gradsF2[j].y;
				vF2[j].x = BETA2_F*vF2[j].x + sBeta3Complement*gradsF2[j].x*gradsF2[j].x;
				vF2[j].y = BETA2_F*vF2[j].y + sBeta3Complement*gradsF2[j].y*gradsF2[j].y;
				float2 denom;
				denom.x = sqrtf(vF2[j].x/sBiasCorrection2) + EPSILON_OPT;
				denom.y = sqrtf(vF2[j].y/sBiasCorrection2) + EPSILON_OPT;
				const float updateX = sLrT*mF2[j].x/denom.x;
				const float updateY = sLrT*mF2[j].y/denom.y;
				const float paramX = paramsF2[j].x;
				const float paramY = paramsF2[j].y;
				paramsF2[j].x = paramX - updateX - sLrWeightDecay*paramX;
				paramsF2[j].y = paramY - updateY - sLrWeightDecay*paramY;
				paramsH2[j] = __float22half2_rn(paramsF2[j]);
				mH2[j] = __float22half2_rn(mF2[j]);
				vH2[j] = __float22half2_rn(vF2[j]);
			}
#pragma unroll
			for(int j = 0; j < 4; j++){
				paramsPtr[j] = paramsH2[j];
				mPtr[j] = mH2[j];
				vPtr[j] = vH2[j];
			}
		}
		const int remainStart = n/8*8;
		for(int i = remainStart + idx; i < n; i += stride){
			const float grad = fmaxf(fminf(__half2float(task.grads[i]), CLIP), -CLIP);
			const float mVal = BETA1_F*__half2float(task.m[i]) + sBeta1Complement*grad;
			const float vVal = BETA2_F*__half2float(task.v[i]) + sBeta3Complement*grad*grad;
			const float param = __half2float(task.params[i]);
			const float denom = sqrtf(vVal/sBiasCorrection2) + EPSILON_OPT;
			const float update = sLrT*mVal/denom;
			task.params[i] = __float2half(param - update - sLrWeightDecay*param);
			task.m[i] = __float2half(mVal);
			task.v[i] = __float2half(vVal);
		}
	}
}
__global__ void AdamwKernelFloatMulti(const AdamWFloatTask* __restrict__ tasks, const int taskCount, const float lr, const int t){
	__shared__ float sBiasCorrection2;
	__shared__ float sLrT;
	__shared__ float sBeta1Complement;
	__shared__ float sBeta3Complement;
	if(threadIdx.x == 0){
		const float biasCorrection1 = 1.0f - powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f - powf(BETA2_F, t);
		sLrT = lr/biasCorrection1;
		sBeta1Complement = 1.0f - BETA1_F;
		sBeta3Complement = 1.0f - BETA2_F;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	for(int taskIdx = 0; taskIdx < taskCount; ++taskIdx){
		const AdamWFloatTask task = tasks[taskIdx];
		const int n = task.size;
		const float lrWeightDecay = lr*task.weightDecay;
#pragma unroll 4
		for(int i = idx; i < n/4; i += stride){
			float4 params4 = reinterpret_cast<const float4*>(task.params)[i];
			const float4 grads4 = reinterpret_cast<const float4*>(task.grads)[i];
			const float4 m4 = reinterpret_cast<const float4*>(task.m)[i];
			const float4 v4 = reinterpret_cast<const float4*>(task.v)[i];
			float4 gradClipped;
			gradClipped.x = fmaxf(fminf(grads4.x, CLIP), -CLIP);
			gradClipped.y = fmaxf(fminf(grads4.y, CLIP), -CLIP);
			gradClipped.z = fmaxf(fminf(grads4.z, CLIP), -CLIP);
			gradClipped.w = fmaxf(fminf(grads4.w, CLIP), -CLIP);
			float4 mNew;
			mNew.x = BETA1_F*m4.x + sBeta1Complement*gradClipped.x;
			mNew.y = BETA1_F*m4.y + sBeta1Complement*gradClipped.y;
			mNew.z = BETA1_F*m4.z + sBeta1Complement*gradClipped.z;
			mNew.w = BETA1_F*m4.w + sBeta1Complement*gradClipped.w;
			float4 vNew;
			vNew.x = BETA2_F*v4.x + sBeta3Complement*gradClipped.x*gradClipped.x;
			vNew.y = BETA2_F*v4.y + sBeta3Complement*gradClipped.y*gradClipped.y;
			vNew.z = BETA2_F*v4.z + sBeta3Complement*gradClipped.z*gradClipped.z;
			vNew.w = BETA2_F*v4.w + sBeta3Complement*gradClipped.w*gradClipped.w;
			float4 update;
			update.x = sLrT*mNew.x/(sqrtf(vNew.x/sBiasCorrection2) + EPSILON_OPT);
			update.y = sLrT*mNew.y/(sqrtf(vNew.y/sBiasCorrection2) + EPSILON_OPT);
			update.z = sLrT*mNew.z/(sqrtf(vNew.z/sBiasCorrection2) + EPSILON_OPT);
			update.w = sLrT*mNew.w/(sqrtf(vNew.w/sBiasCorrection2) + EPSILON_OPT);
			const float paramX = params4.x;
			const float paramY = params4.y;
			const float paramZ = params4.z;
			const float paramW = params4.w;
			params4.x = paramX - update.x - lrWeightDecay*paramX;
			params4.y = paramY - update.y - lrWeightDecay*paramY;
			params4.z = paramZ - update.z - lrWeightDecay*paramZ;
			params4.w = paramW - update.w - lrWeightDecay*paramW;
			reinterpret_cast<float4*>(task.params)[i] = params4;
			reinterpret_cast<float4*>(task.m)[i] = mNew;
			reinterpret_cast<float4*>(task.v)[i] = vNew;
		}
		const int remainStart = n/4*4;
		for(int i = remainStart + idx; i < n; i += stride){
			const float grad = fmaxf(fminf(task.grads[i], CLIP), -CLIP);
			const float mVal = BETA1_F*task.m[i] + sBeta1Complement*grad;
			const float vVal = BETA2_F*task.v[i] + sBeta3Complement*grad*grad;
			const float param = task.params[i];
			const float update = sLrT*mVal/(sqrtf(vVal/sBiasCorrection2) + EPSILON_OPT);
			task.params[i] = param - update - lrWeightDecay*param;
			task.m[i] = mVal;
			task.v[i] = vVal;
		}
	}
}
void AdamWHalfMulti(const AdamWHalfTask* tasks, const int taskCount, const int totalSize, const float lr, const int t){
	if(taskCount <= 0 || totalSize <= 0) return;
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(totalSize, blocks, tpb);
	AdamwKernelHalfMulti<<<blocks, tpb>>>(tasks, taskCount, lr, t);
	checkCUDA(cudaGetLastError());
}
void AdamWFloatMulti(const AdamWFloatTask* tasks, const int taskCount, const int totalSize, const float lr, const int t){
	if(taskCount <= 0 || totalSize <= 0) return;
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(totalSize, blocks, tpb);
	AdamwKernelFloatMulti<<<blocks, tpb>>>(tasks, taskCount, lr, t);
	checkCUDA(cudaGetLastError());
}
