#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "common.h"
#include <cuda.h>
#include <curand.h>
struct pixARGB{
	unsigned char B;
	unsigned char G;
	unsigned char R;
	unsigned char A;
};
struct pixRGB{
	unsigned char B;
	unsigned char G;
	unsigned char R;
};
curandGenerator_t generator_;
int GS, BS, RPB, CPB, TPG, maxTPB, smemPB;
bool inited = false;
extern "C" void InitCUDA(){
	if(inited) return;
	const CUresult cudaRes = cuInit(0);
	if(cudaRes != CUDA_SUCCESS){
		const char* pStr = nullptr;
		cuGetErrorString(cudaRes, &pStr);
		throw std::runtime_error("CUDA Init failed, error string:\n\n" + std::string(pStr));
	}
	inited = true;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int major;
	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
	int minor;
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
	const auto TPM = ConvertSmVer2Cores(major, minor);
	const auto MP = prop.multiProcessorCount;
	const auto warps = prop.warpSize;
	maxTPB = prop.maxThreadsPerBlock;
	smemPB = prop.sharedMemPerBlock;
	GS = warps*MP;
	BS = TPM;
	int TPB = maxTPB;
	TPB = TPB / warps*warps;
	TPG = warps;
	while(TPG*2 <= TPB && TPG < warps){ TPG *= 2; }
	const int groups = TPB / TPG;
	RPB = sqrt(groups);
	CPB = groups / RPB;
	while(RPB*CPB < groups){ if(RPB < CPB){ RPB++; } else{ CPB++; } }
	curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator_, static_cast<unsigned long long>(time(nullptr)));
}
__device__ __host__ int DivCeil(const int a, const int b){ return a % b != 0 ? a / b + 1 : a / b; }
static void GetLaunchConfig(int n, int& blocks, int& tpb){
	if(tpb == 0) tpb = BS;
	blocks = min(DivCeil(n, tpb*8), GS);
}
__global__ void cuARGBtoRGB(const pixARGB* src, pixRGB* dst, int n){
	const auto stride = blockDim.x*gridDim.x;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride){
		dst[i].R = src[i].R;
		dst[i].G = src[i].G;
		dst[i].B = src[i].B;
	}
}
extern "C" cudaError ARGBtoRGB(unsigned char* src, unsigned char* dst, int n){
	int blocks, tpb = 256;
	GetLaunchConfig(n, blocks, tpb);
	cuARGBtoRGB<<<blocks, tpb>>>(reinterpret_cast<pixARGB*>(src), reinterpret_cast<pixRGB*>(dst), n);
	return cudaGetLastError();
}
__global__ void cuARGBtoRGBplanar(const unsigned char* src, unsigned char* dst, int n){
	const auto stride = blockDim.x*gridDim.x;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride){
		const int srcIdx = i*4;
		dst[i] = src[srcIdx + 2];
		dst[i + n] = src[srcIdx + 1];
		dst[i + 2*n] = src[srcIdx];
	}
}
extern "C" cudaError ARGBtoRGBplanar(unsigned char* src, unsigned char* dst, int n){
	int blocks, tpb = 256;
	GetLaunchConfig(n, blocks, tpb);
	cuARGBtoRGBplanar<<<blocks, tpb>>>(src, dst, n);
	return cudaGetLastError();
}
__device__ float d_loss;
__global__ void mseLossKernel(const __half* predictions, const float* targets, int size){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float diff = 0.0f;
	if(idx < size){
		diff = __half2float(predictions[idx]) - targets[idx];
		diff *= diff;
	}
	sdata[tid] = diff;
	__syncthreads();
	for(int i = blockDim.x / 2; i > 0; i >>= 1){
		if(tid < i){ sdata[tid] += sdata[tid + i]; }
		__syncthreads();
	}
	if(tid == 0){ atomicAdd(&d_loss, sdata[0]); }
}
extern "C" float MseLoss(const __half* dPredictions, const float* dTargets, int size){
	constexpr auto zero = 0.0f;
	cudaMemcpyToSymbol(d_loss, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	int gridSize = DivCeil(size, BS);
	mseLossKernel<<<gridSize, BS, BS*sizeof(float)>>>(dPredictions, dTargets, size);
	float h_loss;
	cudaMemcpyFromSymbol(&h_loss, d_loss, sizeof(float));
	return h_loss / size;
}
__device__ float dLossKeys;
__device__ float dLossMouse;
__global__ void mseLoss2Kernel(const __half* predictions, const float* targets, const int size, const int numKeys, const int numCtrls){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float sumKeys = 0.0f;
	float sumMouse = 0.0f;
	while(idx < size){
		const float pred = __half2float(predictions[idx]);
		const float target = targets[idx];
		const bool isKey = idx % numCtrls < numKeys;
		const float diff = isKey ? pred >= 0.5f != target >= 0.5f : (pred - target)*(pred - target);
		sumKeys += diff*isKey;
		sumMouse += diff*!isKey;
		idx += gridDim.x*blockDim.x;
	}
	sdata[tid] = sumKeys;
	sdata[tid + blockDim.x] = sumMouse;
	__syncthreads();
	for(int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
			sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
		}
		__syncthreads();
	}
	if(tid == 0){
		atomicAdd(&dLossKeys, sdata[0]);
		atomicAdd(&dLossMouse, sdata[blockDim.x]);
	}
}
extern "C" void MseLoss2(const __half* dPredictions, const float* dTargets, const int numButs, const int numCtrls, const int batchSize, float* butLoss, float* axesLoss){
	constexpr auto zero = 0.0f;
	const auto size = numCtrls*batchSize;
	cudaMemcpyToSymbol(dLossKeys, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dLossMouse, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	int gridSize = DivCeil(size, BS);
	mseLoss2Kernel<<<gridSize, BS, 2*BS*sizeof(float)>>>(dPredictions, dTargets, size, numButs, numCtrls);
	cudaMemcpyFromSymbol(butLoss, dLossKeys, sizeof(float));
	cudaMemcpyFromSymbol(axesLoss, dLossMouse, sizeof(float));
	*butLoss /= numButs*batchSize;
	*axesLoss /= (numCtrls - numButs)*batchSize;
}
extern "C" void BlockShiftHalf(__half* hPtr, const int shiftBy, const int blocksToShift){
	auto blockSize = shiftBy;
	if(blockSize < 0) blockSize = -blockSize;
	if(shiftBy > 0){ for(int i = blocksToShift; 0 < i; --i){ cudaMemcpy(hPtr + i*blockSize, hPtr + (i - 1)*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice); } } else{
		for(int i = 0; i < blocksToShift; ++i){ cudaMemcpy(hPtr + (i - 1)*blockSize, hPtr + i*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice); }
	}
}
__global__ void ConvertByteToHalfNormKernel(const unsigned char* input, __half* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = __float2half(static_cast<float>(input[idx]) / 255.0f); }
}
__global__ void ConvertByteToHalfKernel(const unsigned char* input, __half* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = __float2half(input[idx]); }
}
extern "C" void ConvertByteToHalf(const unsigned char* input, __half* output, const size_t size, bool normalize){
	int blocks, tpb = 256;
	GetLaunchConfig(size, blocks, tpb);
	if(normalize) ConvertByteToHalfNormKernel<<<blocks, tpb>>>(input, output, size);
	else ConvertByteToHalfKernel<<<blocks, tpb>>>(input, output, size);
}
__global__ void ConvertHalfToByteNormKernel(const __half* input, unsigned char* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = static_cast<unsigned char>(__half2float(input[idx])*255.0f); }
}
__global__ void ConvertHalfToByteKernel(const __half* input, unsigned char* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = static_cast<unsigned char>(__half2float(input[idx])); }
}
extern "C" void ConvertHalfToByte(const __half* input, unsigned char* output, const size_t size, const bool normalize){
	int blocks, tpb = 256;
	GetLaunchConfig(size, blocks, tpb);
	if(normalize) ConvertHalfToByteNormKernel<<<blocks, tpb>>>(input, output, size);
	else ConvertHalfToByteKernel<<<blocks, tpb>>>(input, output, size);
}
__global__ void ConvertFloatToHalfKernel(const float* input, __half* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = __float2half(input[idx]); }
}
extern "C" void ConvertFloatToHalf(const float* input, __half* output, const size_t size){
	int blocks, tpb = 256;
	GetLaunchConfig(size, blocks, tpb);
	ConvertFloatToHalfKernel<<<blocks, tpb>>>(input, output, size);
}
__global__ void ConvertHalfToFloatKernel(const __half* input, float* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = __half2float(input[idx]); }
}
extern "C" void ConvertHalfToFloat(const __half* input, float* output, const size_t size){
	int blocks, tpb = 256;
	GetLaunchConfig(size, blocks, tpb);
	ConvertHalfToFloatKernel<<<blocks, tpb>>>(input, output, size);
}
__global__ void ConvertFloatToHalfScaleKernel(__half* halfWeights, const float* weights, const size_t size, const float scale){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ halfWeights[idx] = __float2half(weights[idx]*scale); }
}
extern "C" void ConvertFloatToHalfScale(__half* halfWeights, const float* weights, const size_t size, const float scale){
	int blocks, tpb = 256;
	GetLaunchConfig(size, blocks, tpb);
	ConvertFloatToHalfScaleKernel<<<blocks, tpb>>>(halfWeights, weights, size, scale);
}
extern "C" void WeightInit(__half* weightHalf, const int numWeights, const int fanIn, const int fanOut, const WeightInitMethod method){
	if(method == Orthogonal){ OrthogonalInit(weightHalf, fanIn, fanOut); } else{
		float* weightFloat;
		checkCUDA(cudaMalloc(&weightFloat, numWeights*sizeof(float)));
		const float factor = method == Xavier ? 1.0f : 2.0f;
		curandGenerateNormal(generator_, weightFloat, numWeights, 0.0f, 1.0f);
		ConvertFloatToHalfScale(weightHalf, weightFloat, numWeights, sqrtf(factor / fanIn));
		cudaFree(weightFloat);
	}
}
#define BETA1_F 0.9f
#define BETA2_F 0.999f
#define EPSILON_F 1e-7f
#define CLIP 1.0f
__global__ void SGDHalfKernel(__half* params, const __half* grads, const int size, const float learningRate, const float weightDecay){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		params[idx] *= 1.0f - weightDecay;
		const float gradClipped = fmaxf(fminf(__half2float(grads[idx]), CLIP), -CLIP);
		params[idx] = __float2half(__half2float(params[idx]) - learningRate*gradClipped);
	}
}
extern "C" void SGDHalf(__half* params, const __half* grads, const int size, const float learningRate, const float weightDecay){
	int blocks, tpb = 128;
	GetLaunchConfig(size, blocks, tpb);
	SGDHalfKernel<<<blocks, tpb>>>(params, grads, size, learningRate, weightDecay);
}
__global__ void SGDFloatKernel(float* params, const float* grads, const int size, const float learningRate, const float weightDecay){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		params[idx] *= 1.0f - weightDecay;
		const float gradClipped = fmaxf(fminf(grads[idx], CLIP), -CLIP);
		params[idx] -= learningRate*gradClipped;
	}
}
extern "C" void SGDFloat(float* params, const float* grads, const int size, const float learningRate, const float weightDecay){
	int blocks, tpb = 128;
	GetLaunchConfig(size, blocks, tpb);
	SGDFloatKernel<<<blocks, tpb>>>(params, grads, size, learningRate, weightDecay);
}
__global__ void AdamwKernelFloat(float* __restrict__ params, const float* __restrict__ grads, float* __restrict__ m, float* __restrict__ v, const float lr, const int t, const float wd, const int n){
	__shared__ float sBiasCorrection2;
	__shared__ float sLrT;
	__shared__ float sBeta1Complement;
	__shared__ float sBeta3Complement;
	if(threadIdx.x == 0){
		const float biasCorrection1 = 1.0f - powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f - powf(BETA2_F, t);
		sLrT = lr / biasCorrection1;
		sBeta1Complement = 1.0f - BETA1_F;
		sBeta3Complement = 1.0f - BETA2_F;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
#pragma unroll 4
	for(int i = idx; i < n / 4; i += stride){
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
		// Apply weight decay
		params4.x *= 1.0f - wd;
		params4.y *= 1.0f - wd;
		params4.z *= 1.0f - wd;
		params4.w *= 1.0f - wd;
		// Compute updates
		float4 update;
		update.x = sLrT*mNew.x / (sqrtf(vNew.x / sBiasCorrection2) + EPSILON_F);
		update.y = sLrT*mNew.y / (sqrtf(vNew.y / sBiasCorrection2) + EPSILON_F);
		update.z = sLrT*mNew.z / (sqrtf(vNew.z / sBiasCorrection2) + EPSILON_F);
		update.w = sLrT*mNew.w / (sqrtf(vNew.w / sBiasCorrection2) + EPSILON_F);
		// Apply updates
		params4.x -= update.x;
		params4.y -= update.y;
		params4.z -= update.z;
		params4.w -= update.w;
		// Store results
		reinterpret_cast<float4*>(params)[i] = params4;
		reinterpret_cast<float4*>(m)[i] = mNew;
		reinterpret_cast<float4*>(v)[i] = vNew;
	}
	// Handle remaining elements
	const int remainStart = n / 4*4;
	for(int i = remainStart + idx; i < n; i += stride){
		const float grad = fmaxf(fminf(grads[i], CLIP), -CLIP);
		const float mVal = BETA1_F*m[i] + sBeta1Complement*grad;
		const float vVal = BETA2_F*v[i] + sBeta3Complement*grad*grad;
		params[i] = (params[i] - sLrT*mVal / (sqrtf(vVal / sBiasCorrection2) + EPSILON_F))*(1.0f - wd);
		m[i] = mVal;
		v[i] = vVal;
	}
}
extern "C" void AdamWFloat(float* params, const float* grads, float* m, float* v, const float learningRate, const int t, const float weightDecay, const int size){
	int blocks, tpb = 16;
	GetLaunchConfig(size, blocks, tpb);
	AdamwKernelFloat<<<blocks, tpb>>>(params, grads, m, v, learningRate, t, weightDecay, size);
}
__global__ void AdamwKernelHalf(__half* __restrict__ params, const __half* __restrict__ grads, __half* __restrict__ m, __half* __restrict__ v, const float lr, const int t, const float wd, const int n){
	// Shared memory for frequently used constants
	__shared__ float sBiasCorrection1;
	__shared__ float sBiasCorrection2;
	__shared__ float sLrT;
	__shared__ float sBeta1Complement;
	__shared__ float sBeta3Complement;
	__shared__ float sWeightDecay;
	// First thread in block computes constants
	if(threadIdx.x == 0){
		sBiasCorrection1 = 1.0f - powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f - powf(BETA2_F, t);
		sLrT = lr / sBiasCorrection1;
		sBeta1Complement = 1.0f - BETA1_F;
		sBeta3Complement = 1.0f - BETA2_F;
		sWeightDecay = 1.0f - wd;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	// Process 4 half2 elements (8 total elements) at once
#pragma unroll 4
	for(int i = idx; i < n / 8; i += stride){
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
			// Apply weight decay
			paramsF2[j].x *= sWeightDecay;
			paramsF2[j].y *= sWeightDecay;
			// Compute denominator
			float2 denom;
			denom.x = sqrtf(vF2[j].x / sBiasCorrection2) + EPSILON_F;
			denom.y = sqrtf(vF2[j].y / sBiasCorrection2) + EPSILON_F;
			// Update parameters
			paramsF2[j].x -= sLrT*mF2[j].x / denom.x;
			paramsF2[j].y -= sLrT*mF2[j].y / denom.y;
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
	const int remainStart = n / 8*8;
	for(int i = remainStart + idx; i < n; i += stride){
		const float grad = fmaxf(fminf(__half2float(grads[i]), CLIP), -CLIP);
		const float mVal = BETA1_F*__half2float(m[i]) + sBeta1Complement*grad;
		const float vVal = BETA2_F*__half2float(v[i]) + sBeta3Complement*grad*grad;
		const float param = __half2float(params[i])*sWeightDecay;
		const float denom = sqrtf(vVal / sBiasCorrection2) + EPSILON_F;
		params[i] = __float2half(param - sLrT*mVal / denom);
		m[i] = __float2half(mVal);
		v[i] = __float2half(vVal);
	}
}
extern "C" void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, const float lr, const int t, const float weightDecay, const int size){
	int blocks, tpb = 16;
	GetLaunchConfig(size, blocks, tpb);
	AdamwKernelHalf<<<blocks, tpb>>>(params, grads, m, v, lr, t, weightDecay, size);
}
__global__ void GradientKernel(__half* grads, const __half* predictions, const __half* targets, const float clip, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ grads[idx] = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx])))); }
}
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size){
	auto gridSize = DivCeil(size, BS);
	GradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size);
}
__global__ void SplitGradKernel(__half* gradients, const __half* predictions, const float* targets, const float clip, const int numCtrls, const int numButs, const int batchSize, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		const auto diff = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx]) - targets[idx])));
		if(ctrlId < numButs){
			const auto gradIdx = batchId*numButs + ctrlId;
			gradients[gradIdx] = diff;
		} else{
			const auto gradIdx = numButs*batchSize + batchId*(numCtrls - numButs) + (ctrlId - numButs);
			gradients[gradIdx] = diff;
		}
	}
}
extern "C" void SplitGradient(__half* dGradient, const __half* dPredictions, const float* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = DivCeil(size, BS);
	SplitGradKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, numCtrls, numButs, batchSize, size);
}
__global__ void MergeOutputsKernel(__half* predOut, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		if(ctrlId < numButs){ predOut[idx] = buttonData[batchId*numButs + ctrlId]; } else{ predOut[idx] = axisData[batchId*(numCtrls - numButs) + (ctrlId - numButs)]; }
	}
}
extern "C" void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, const int numCtrls, const int numButs, const int size){
	auto gridSize = DivCeil(size, BS);
	MergeOutputsKernel<<<gridSize, BS>>>(predOut, buttonData, axisData, size, numCtrls, numButs);
}
__global__ void GetPredictionKernel(const __half* predBatch, float* prediction, const int numCtrls, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < numCtrls){ prediction[idx] = __half2float(predBatch[idx + size - numCtrls]); }
}
extern "C" void GetPrediction(const __half* predBatch, float* prediction, const int numCtrls, const int batchSize){
	float* devPtr = nullptr;
	cudaHostGetDevicePointer(&devPtr, prediction, 0);
	GetPredictionKernel<<<1, numCtrls>>>(predBatch, devPtr, numCtrls, batchSize*numCtrls);
	cudaDeviceSynchronize();
}
__global__ void BCEGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float y = __half2float(targets[idx]);
		const float pClamped = fminf(fmaxf(__half2float(predictions[idx]), EPSILON_F), 1.0f - EPSILON_F);
		float gradient = 0.0f;
		if(y == 1.0f){
			gradient = (pClamped - 1.0f) / pClamped;
			gradients[idx] = __float2half(gradient*scale);
		} else{
			gradient = pClamped / (1.0f - pClamped);
			gradients[idx] = __float2half(gradient*scale);
		}
	}
}
extern "C" void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale){
	auto gridSize = DivCeil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
}
__global__ void LeakyReluKernel(const __half* __restrict__ dataIn, __half* __restrict__ dataOut, const int size, const float negativeSlope){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const auto val = __half2float(dataIn[idx]);
		dataOut[idx] = __float2half(val < 0.0f ? val*negativeSlope : 0.0f);
	}
}
extern "C" void LeakyReluForward(const __half* dataIn, __half* dataOut, const int size, const float negativeSlope, cudaStream_t stream){
	int blocks, tpb = 128;
	GetLaunchConfig(size, blocks, tpb);
	LeakyReluKernel<<<blocks, tpb, 0, stream>>>(dataIn, dataOut, size, negativeSlope);
}
__global__ void LeakyReluBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ dataIn, const int size, const float negativeSlope){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ grad[idx] *= __float2half(__half2float(dataIn[idx]) < 0.0f ? negativeSlope : 1.0f); }
}
extern "C" void LeakyReluBackward(__half* grad, const __half* dataIn, const int size, const float negativeSlope, cudaStream_t stream){
	int blocks, tpb = 128;
	GetLaunchConfig(size, blocks, tpb);
	LeakyReluBackwardKernel<<<blocks, tpb, 0, stream>>>(grad, dataIn, size, negativeSlope);
}
__global__ void SwishKernel(const __half* __restrict__ dataIn, __half* __restrict__ outData, const int size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const auto val = __half2float(dataIn[idx]);
		outData[idx] = __float2half(val / (1.0f + exp(-val)));
	}
}
extern "C" void SwishForward(const __half* dataIn, __half* outData, const int size, cudaStream_t stream){
	int blocks, tpb = 128;
	GetLaunchConfig(size, blocks, tpb);
	SwishKernel<<<blocks, tpb, 0, stream>>>(dataIn, outData, size);
}
__global__ void SwishBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ dataIn, const int size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const auto val = __half2float(dataIn[idx]);
		const auto sig = val / (1.0f + val);
		grad[idx] = __float2half(__half2float(grad[idx])*(val + sig*(1.0f - val)));
	}
}
extern "C" void SwishBackward(__half* grad, const __half* dataIn, const int size, cudaStream_t stream){
	int blocks, tpb = 128;
	GetLaunchConfig(size, blocks, tpb);
	SwishBackwardKernel<<<blocks, tpb, 0, stream>>>(grad, dataIn, size);
}
__global__ void SigmoidForwardKernel(const __half* __restrict__ dataIn, __half* __restrict__ dataOut, const int numCtrls, const int numButs, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(dataIn[idx]);
	dataOut[idx] = __float2half(1.0f / (1.0f + expf(-val)));
}
extern "C" void SigmoidForward(const __half* dataIn, __half* dataOut, const int numCtrls, const int numButs, const int size, cudaStream_t cudaStream){
	auto gridSize = DivCeil(size, BS);
	SigmoidForwardKernel<<<gridSize, BS, 0, cudaStream>>>(dataIn, dataOut, numCtrls, numButs, size);
}
__global__ void SigmoidBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ dataIn, const int numCtrls, const int numButs, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(dataIn[idx]);
	grad[idx] = __float2half(__half2float(grad[idx])*val*(1.0f - val));
}
extern "C" void SigmoidBackward(__half* grad, const __half* dataIn, const int numCtrls, const int numButs, const int size, cudaStream_t cudaStream){
	auto gridSize = DivCeil(size, BS);
	SigmoidBackwardKernel<<<gridSize, BS, 0, cudaStream>>>(grad, dataIn, numCtrls, numButs, size);
}
constexpr float SQRT_2_PI = 0.7978845608028654f;
constexpr float GELU_COEF_A = 0.044715f;
__global__ void GELUForwardKernel(const __half* __restrict__ dataIn, __half* __restrict__ dataOut, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	for(int i = idx; i < size; i += stride){
		const float x = __half2float(dataIn[i]);
		const float cdf = 0.5f*(1.0f + tanhf(SQRT_2_PI*(x + GELU_COEF_A*x*x*x)));
		const float result = x*cdf;
		dataOut[i] = __float2half(result);
	}
}
extern "C" void GELUForward(const __half* dataIn, __half* dataOut, const int size, cudaStream_t stream){
	int blocks, threads = 512;
	GetLaunchConfig(size, blocks, threads);
	GELUForwardKernel<<<blocks, threads, 0, stream>>>(dataIn, dataOut, size);
}
__global__ void GELUBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ dataIn, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	for(int i = idx; i < size; i += stride){
		const float x = __half2float(dataIn[i]);
		const float xCu = x*x*x;
		const float cdf = 0.5f*(1.0f + tanhf(SQRT_2_PI*(x + GELU_COEF_A*xCu)));
		const float pdf = SQRT_2_PI*(1.0f + 3.0f*GELU_COEF_A*x*x)*(1.0f - tanhf(SQRT_2_PI*(x + GELU_COEF_A*xCu))*tanhf(SQRT_2_PI*(x + GELU_COEF_A*xCu)))*0.5f;
		const float result = __half2float(grad[i])*(cdf + x*pdf);
		grad[i] = __float2half(result);
	}
}
extern "C" void GELUBackward(__half* grad, const __half* dataIn, const int size, cudaStream_t stream){
	int blocks, threads = 512;
	GetLaunchConfig(size, blocks, threads);
	GELUBackwardKernel<<<blocks, threads, 0, stream>>>(grad, dataIn, size);
}
__global__ void ComputeMeanVarianceKernel(const __half* __restrict__ dataIn, float* mean, float* variance, const int N, const int C, const int HW){
	extern __shared__ float sdata[];
	float *sMean = sdata, *sM2 = sMean + blockDim.x, *sCount = sM2 + blockDim.x;
	const int tid = threadIdx.x, cid = blockIdx.x;
	if(cid >= C) return;
	float tMean = 0.0f, tM2 = 0.0f, count = 0.0f;
	for(int n = 0; n < N; ++n){
		for(int i = tid; i < HW; i += blockDim.x){
			const int idx = n*C*HW + cid*HW + i;
			const float val = __half2float(dataIn[idx]);
			++count;
			const float delta = val - tMean;
			tMean += delta / count;
			tM2 += delta*(val - tMean);
		}
	}
	sMean[tid] = tMean;
	sM2[tid] = tM2;
	sCount[tid] = count;
	__syncthreads();
	for(int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tid < s){
			const float m1 = sMean[tid], m2 = sMean[tid + s], c1 = sCount[tid], c2 = sCount[tid + s];
			const float delta = m2 - m1, nCount = c1 + c2;
			sMean[tid] = (m1*c1 + m2*c2) / nCount;
			sM2[tid] += sM2[tid + s] + delta*delta*c1*c2 / nCount;
			sCount[tid] = nCount;
		}
		__syncthreads();
	}
	if(tid == 0){
		mean[cid] = sMean[0];
		variance[cid] = sM2[0] / sCount[0];
	}
}
__global__ void LayerNormForwardKernel(__half* __restrict__ output, const __half* __restrict__ data, const float* gamma, const float* beta, const float* mean, const float* variance, int N, int C, int HW){
	extern __shared__ float sParams[];
	float *sMean = sParams, *sVar = sMean + C, *sGamma = sVar + C, *sBeta = sGamma + C;
	for(int i = threadIdx.x; i < C; i += blockDim.x){
		sMean[i] = mean[i];
		sVar[i] = variance[i];
		sGamma[i] = gamma[i];
		sBeta[i] = beta[i];
	}
	__syncthreads();
	const int total = N*C*HW;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < total; i += blockDim.x*gridDim.x){
		const int c = i / HW % C;
		const float x = __half2float(data[i]);
		const float norm = (x - sMean[c]) / sqrtf(sVar[c] + EPSILON_F);
		output[i] = __float2half(norm*sGamma[c] + sBeta[c]);
	}
}
extern "C" void LayerNormForward(__half* __restrict__ dataOut, const __half* __restrict__ dataIn, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW){
	int gridSize = DivCeil(C, BS);
	int sharedMemSize = 3*BS*sizeof(float);
	ComputeMeanVarianceKernel<<<gridSize, BS, sharedMemSize>>>(dataIn, mean, variance, N, C, HW);
	gridSize = DivCeil(N*C*HW, BS);
	sharedMemSize = 4*C*sizeof(float);
	LayerNormForwardKernel<<<gridSize, BS, sharedMemSize>>>(dataOut, dataIn, gamma, beta, mean, variance, N, C, HW);
	const cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){ printf("Kernel execution failed: %s\n", cudaGetErrorString(err)); }
}
__global__ void LayerNormBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW){
	extern __shared__ float sdata[];
	float *sGradGamma = sdata, *sGradBeta = sGradGamma + blockDim.x;
	const int tid = threadIdx.x, cid = blockIdx.x;
	if(cid >= C) return;
	float tGradGamma = 0.0f, tGradBeta = 0.0f;
	const float invStd = rsqrtf(variance[cid] + EPSILON_F);
	for(int n = 0; n < N; ++n){
		for(int i = tid; i < HW; i += blockDim.x){
			const int idx = n*C*HW + cid*HW + i;
			const float x = __half2float(data[idx]);
			const float dy = __half2float(grad[idx]);
			const float xHat = (x - mean[cid])*invStd;
			tGradGamma += xHat*dy;
			tGradBeta += dy;
		}
	}
	sGradGamma[tid] = tGradGamma;
	sGradBeta[tid] = tGradBeta;
	__syncthreads();
	for(int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tid < s){
			sGradGamma[tid] += sGradGamma[tid + s];
			sGradBeta[tid] += sGradBeta[tid + s];
		}
		__syncthreads();
	}
	if(tid == 0){
		atomicAdd(&gradGamma[cid], sGradGamma[0]);
		atomicAdd(&gradBeta[cid], sGradBeta[0]);
	}
	__syncthreads();
	const float gGamma = gradGamma[cid], gBeta = gradBeta[cid];
	for(int n = 0; n < N; ++n){
		for(int i = tid; i < HW; i += blockDim.x){
			const int idx = n*C*HW + cid*HW + i;
			const float x = __half2float(data[idx]);
			const float dy = __half2float(grad[idx]);
			const float xHat = (x - mean[cid])*invStd;
			const float gradInput = gamma[cid]*invStd*(dy - (xHat*gGamma + gBeta) / (N*HW));
			grad[idx] = __float2half(gradInput);
		}
	}
}
extern "C" void LayerNormBackward(__half* __restrict__ grad, const __half* __restrict__ dataIn, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW){
	cudaMemset(gradGamma, 0, C*sizeof(float));
	cudaMemset(gradBeta, 0, C*sizeof(float));
	int sharedMemSize = 2*BS*sizeof(float);
	LayerNormBackwardKernel<<<C, BS, sharedMemSize>>>(grad, dataIn, gamma, gradGamma, gradBeta, mean, variance, N, C, HW);
	const cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){ printf("Kernel execution failed: %s\n", cudaGetErrorString(err)); }
}
__device__ int deviceResult;
__global__ void isNaNKernel(const __half* __restrict__ data, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){ atomicExch(&deviceResult, 1); }
}
extern "C" bool IsnanHalf(const __half* __restrict__ data, int size){
	int hResult = 0;
	cudaMemcpyToSymbol(deviceResult, &hResult, sizeof(int));
	auto gridSize = DivCeil(size, BS);
	isNaNKernel<<<gridSize, BS>>>(data, size);
	cudaMemcpyFromSymbol(&hResult, deviceResult, sizeof(int));
	return hResult != 0;
}
__global__ void ComputeAttentionKernel(const __half* __restrict__ queryMap, const __half* __restrict__ keyMap, __half* __restrict__ attentionScores, int inC, int attC, int inH, int inW, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size) return;
	int i = idx;
	const int w = i % inW;
	i /= inW;
	const int h = i % inH;
	i /= inH;
	const int cPrime = i % attC;
	i /= attC;
	const int n = i;
	float sum = 0.f;
	const int qBase = ((n*attC + cPrime)*inH + h)*inW + w;
	for(int c = 0; c < inC; ++c){
		const int kBase = ((n*inC + c)*inH + h)*inW + w;
		const float qVal = __half2float(queryMap[qBase]);
		const float kVal = __half2float(keyMap[kBase]);
		sum += qVal*kVal;
	}
	sum *= 1.0f / sqrtf(static_cast<float>(inC));
	attentionScores[idx] = __float2half(sum);
}
extern "C" void ComputeAttention(const __half* __restrict__ queryMap, const __half* __restrict__ keyMap, __half* __restrict__ attentionScores, int inC, int attC, int inH, int inW){
	const auto size = inC*attC*inH*inW;
	auto gridSize = DivCeil(size, BS);
	ComputeAttentionKernel<<<gridSize, BS>>>(queryMap, keyMap, attentionScores, inC, attC, inH, inW, size);
}
__global__ void ApplyAttentionKernel(const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, __half* __restrict__ output, int inC, int attC, int inH, int inW, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size) return;
	int i = idx;
	const int w = i % inW;
	i /= inW;
	const int h = i % inH;
	i /= inH;
	const int c = i % inC;
	i /= inC;
	const int n = i;
	const int valIdx = ((n*inC + c)*inH + h)*inW + w;
	const float val = __half2float(valueMap[valIdx]);
	float sum = 0.f;
	for(int ac = 0; ac < attC; ++ac){
		const int attIdx = ((n*attC + ac)*inH + h)*inW + w;
		sum += val*__half2float(attentionScores[attIdx]);
	}
	output[idx] = __float2half(sum);
}
extern "C" void ApplyAttention(const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, __half* __restrict__ output, int inC, int attC, int inH, int inW){
	const auto size = inC*attC*inH*inW;
	auto gridSize = DivCeil(size, BS);
	ApplyAttentionKernel<<<gridSize, BS>>>(valueMap, attentionScores, output, inC, attC, inH, inW, size);
}
__global__ void ApplyAttentionBackwardKernel(const __half* __restrict__ gradIn, const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, __half* __restrict__ gradValue, __half* __restrict__ gradAttention, int inC, int attC, int inH, int inW, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size) return;
	int i = idx;
	const int w = i % inW;
	i /= inW;
	const int h = i % inH;
	i /= inH;
	//int c = i%inC;
	i /= inC;
	const int n = i;
	const float dOut = __half2float(gradIn[idx]);
	const float val = __half2float(valueMap[idx]);
	float gradValSum = 0.f;
	for(int ac = 0; ac < attC; ++ac){
		const int attIdx = ((n*attC + ac)*inH + h)*inW + w;
		gradValSum += dOut*__half2float(attentionScores[attIdx]);
	}
	gradValue[idx] = __float2half(gradValSum);
	for(int ac = 0; ac < attC; ++ac){
		const int attIdx = ((n*attC + ac)*inH + h)*inW + w;
		const float increment = dOut*val;
		atomicAdd(&gradAttention[attIdx], __float2half(increment));
	}
}
extern "C" void ApplyAttentionBackward(const __half* __restrict__ gradIn, const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, __half* __restrict__ gradValue, __half* __restrict__ gradAttention, int inC, int attC, int inH, int inW){
	const auto size = inC*attC*inH*inW;
	auto gridSize = DivCeil(size, BS);
	ApplyAttentionBackwardKernel<<<gridSize, BS>>>(gradIn, valueMap, attentionScores, gradValue, gradAttention, inC, attC, inH, inW, size);
}
__global__ void ComputeQueryKeyGradKernel(const __half* __restrict__ gradAttention, const __half* __restrict__ queryMap, const __half* __restrict__ keyMap, __half* __restrict__ gradQuery, __half* __restrict__ gradKey, int inC, int attC, int inH, int inW, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size) return;
	const int w = idx % inW;
	int tmp = idx / inW;
	const int h = tmp % inH;
	tmp /= inH;
	const int cPrime = tmp % attC;
	tmp /= attC;
	const int c = tmp % inC;
	const int n = tmp / inC;
	const int attIdx = ((n*attC + cPrime)*inH + h)*inW + w;
	float gAttVal = __half2float(gradAttention[attIdx]);
	const float scale = 1.0f / sqrtf(static_cast<float>(inC));
	gAttVal *= scale;
	const float qVal = __half2float(queryMap[attIdx]);
	const int keyIdx = ((n*inC + c)*inH + h)*inW + w;
	const float kVal = __half2float(keyMap[keyIdx]);
	const float dQuery = gAttVal*kVal;
	const float dKey = gAttVal*qVal;
	atomicAdd(&gradQuery[attIdx], dQuery);
	atomicAdd(&gradKey[keyIdx], dKey);
}
extern "C" void ComputeQueryKeyGrad(const __half* __restrict__ gradAttention, const __half* __restrict__ queryMap, const __half* __restrict__ keyMap, __half* __restrict__ gradQuery, __half* __restrict__ gradKey, int N, int inC, int attC, int inH, int inW){
	const auto size = N*inC*attC*inH*inW;
	auto gridSize = DivCeil(size, BS);
	ComputeQueryKeyGradKernel<<<gridSize, BS>>>(gradAttention, queryMap, keyMap, gradQuery, gradKey, inC, attC, inH, inW, size);
}
__global__ void SpatialSoftmaxKernelHalf(const __half* __restrict__ inData, __half* __restrict__ outData, int N, int C, int H, int W){
	const auto blockSize = blockDim.x;
	const int nc = blockIdx.x;
	if(nc >= N*C) return;
	const int n = nc / C;
	const int c = nc % C;
	const int HW = H*W;
	__shared__ float sMax;
	__shared__ float sSumExp;
	const int baseOffset = (n*C + c)*HW;
	float threadMax = -FLT_MAX;
	for(int tid = threadIdx.x; tid < HW; tid += blockSize){
		const float val = __half2float(inData[baseOffset + tid]);
		if(val > threadMax){ threadMax = val; }
	}
	__shared__ float buf[1024];
	buf[threadIdx.x] = threadMax;
	__syncthreads();
	for(int offset = blockSize / 2; offset > 0; offset >>= 1){
		if(threadIdx.x < offset){
			const float other = buf[threadIdx.x + offset];
			if(other > buf[threadIdx.x]){ buf[threadIdx.x] = other; }
		}
		__syncthreads();
	}
	if(threadIdx.x == 0){ sMax = buf[0]; }
	__syncthreads();
	float threadSumExp = 0.0f;
	for(int tid = threadIdx.x; tid < HW; tid += blockSize){
		const float val = __half2float(inData[baseOffset + tid]);
		const float expVal = expf(val - sMax);
		threadSumExp += expVal;
	}
	buf[threadIdx.x] = threadSumExp;
	__syncthreads();
	for(int offset = blockSize / 2; offset > 0; offset >>= 1){
		if(threadIdx.x < offset){ buf[threadIdx.x] += buf[threadIdx.x + offset]; }
		__syncthreads();
	}
	if(threadIdx.x == 0){ sSumExp = buf[0]; }
	__syncthreads();
	for(int tid = threadIdx.x; tid < HW; tid += blockSize){
		const float val = __half2float(inData[baseOffset + tid]);
		const float expVal = expf(val - sMax);
		const float softmaxVal = expVal / sSumExp;
		outData[baseOffset + tid] = __float2half(softmaxVal);
	}
}
extern "C" void SpatialSoftmaxHalf(const __half* inData, __half* outData, int N, int C, int H, int W){
	auto gridSize = DivCeil(N*C, 1);
	SpatialSoftmaxKernelHalf<<<gridSize, BS, 1040>>>(inData, outData, N, C, H, W);
}
__global__ void SpatialSoftmaxBackwardHalfKernel(const __half* __restrict__ outData, const __half* __restrict__ graIn, __half* __restrict__ gradOut, int N, int C, int H, int W){
	const int blockSize = blockDim.x;
	const int nc = blockIdx.x;
	if(nc >= N*C) return;
	const int n = nc / C;
	const int c = nc % C;
	const int hw = H*W;
	__shared__ float sSumG;
	__shared__ float buf[1024];
	const int baseOffset = (n*C + c)*hw;
	float threadSum = 0.0f;
	for(int tid = threadIdx.x; tid < hw; tid += blockSize){
		const float y = __half2float(outData[baseOffset + tid]);
		const float dO = __half2float(graIn[baseOffset + tid]);
		threadSum += y*dO;
	}
	buf[threadIdx.x] = threadSum;
	__syncthreads();
	for(int offset = blockSize / 2; offset > 0; offset >>= 1){
		if(threadIdx.x < offset){ buf[threadIdx.x] += buf[threadIdx.x + offset]; }
		__syncthreads();
	}
	if(threadIdx.x == 0){ sSumG = buf[0]; }
	__syncthreads();
	for(int tid = threadIdx.x; tid < hw; tid += blockSize){
		const float y = __half2float(outData[baseOffset + tid]);
		const float dO = __half2float(graIn[baseOffset + tid]);
		const float dI = y*(dO - sSumG);
		gradOut[baseOffset + tid] = __float2half(dI);
	}
}
extern "C" void SpatialSoftmaxBackwardHalf(const __half* outData, const __half* gradIn, __half* gradOut, int N, int C, int H, int W){
	int gridSize = DivCeil(N*C, 1);
	SpatialSoftmaxBackwardHalfKernel<<<gridSize, BS, 1032>>>(outData, gradIn, gradOut, N, C, H, W);
}
__global__ void FeatureMapMosaicKernel(const __half* __restrict__ input, unsigned char* __restrict__ output, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW){
	const int c = blockIdx.x*blockDim.x + threadIdx.x; // Channel index
	const int y = blockIdx.y*blockDim.y + threadIdx.y; // Y-coordinate in feature map
	const int x = blockIdx.z*blockDim.z + threadIdx.z; // X-coordinate in feature map
	if(c >= inC || y >= H || x >= W) return;
	// Calculate tile position using provided grid dimensions
	const int tileX = c % gridW;
	const int tileY = c / gridW;
	// Compute destination position in the mosaic
	const int outX = tileX*tileW + x;
	const int outY = tileY*tileH + y;
	// Convert FP16 to 8-bit unsigned char
	const __half value = input[c*H*W + y*W + x];
	const float fVal = __half2float(value);
	const unsigned char pixel = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, fVal*255.0f)));
	// Store to output
	output[outY*mosaicW + outX] = pixel;
}
extern "C" void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, cudaStream_t stream){
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((inC + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, (W + blockDim.z - 1) / blockDim.z);
	FeatureMapMosaicKernel<<<gridDim, blockDim, 0, stream>>>(dInput, dOutput, H, W, inC, mosaicW, tileW, tileH, gridW);
}
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda::wmma;
// Optimized WMMA Attention Kernel for Volta
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, float* __restrict__ AttentionWeights, int B, int T, int D, int H){
	// Block indices
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int row_block = blockIdx.x;
	// Early exit
	if(row_block * 16 >= T) return;
	// Warp and lane IDs
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	const int num_warps = blockDim.x / 32;
	// Calculate base offsets
	const int batch_head_offset = (batch * H + head) * T * D;
	// Shared memory layout
	extern __shared__ char shared_mem_bytes[];
	__half* Q_shared = (__half*)shared_mem_bytes;
	__half* K_shared = Q_shared + 16 * D;
	__half* V_shared = K_shared + 16 * D;
	float* scores_shared = (float*)(V_shared + 16 * D);
	float* row_max = scores_shared + 16 * T;
	float* row_sum = row_max + 16;
	// WMMA fragments
	fragment<matrix_a, 16, 16, 16, __half, row_major> q_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> k_frag;
	fragment<matrix_b, 16, 16, 16, __half, row_major> v_frag;
	fragment<accumulator, 16, 16, 16, float> scores_frag;
	fragment<accumulator, 16, 16, 16, float> out_frag;
	// Initialize row max and sum
	if(threadIdx.x < 16){
		row_max[threadIdx.x] = -FLT_MAX;
		row_sum[threadIdx.x] = 0.0f;
	}
	// Load Q tile for this row block
#pragma unroll
	for(int d = threadIdx.x; d < D * 16; d += blockDim.x){
		const int row = d / D;
		const int col = d % D;
		if(row_block * 16 + row < T){ Q_shared[row * D + col] = Q[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ Q_shared[row * D + col] = __float2half(0.0f); }
	}
	__syncthreads();
	// Step 1: Compute Q*K^T scores
	// Distribute column blocks across warps
	for(int col_block_base = 0; col_block_base < (T + 15) / 16; col_block_base += num_warps){
		const int col_block = col_block_base + warp_id;
		if(col_block < (T + 15) / 16){
			// Initialize scores accumulator
			fill_fragment(scores_frag, 0.0f);
			// Load K tile into shared memory (transposed for col_major)
#pragma unroll
			for(int d = lane_id; d < D * 16; d += 32){
				const int row = d / D;
				const int col = d % D;
				if(col_block * 16 + row < T){ K_shared[col * 16 + row] = K[batch_head_offset + (col_block * 16 + row) * D + col]; } else{ K_shared[col * 16 + row] = __float2half(0.0f); }
			}
			__syncwarp();
			// Compute Q*K^T using WMMA
#pragma unroll
			for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
				load_matrix_sync(q_frag, Q_shared + d_block * 16, D);
				load_matrix_sync(k_frag, K_shared + d_block * 16 * 16, 16);
				mma_sync(scores_frag, q_frag, k_frag, scores_frag);
			}
			// Scale scores and store
#pragma unroll
			for(int i = 0; i < scores_frag.num_elements; i++){ scores_frag.x[i] *= rsqrtf(static_cast<float>(D)); }
			// Store scores to shared memory
			store_matrix_sync(scores_shared + warp_id * 16 * 16, scores_frag, 16, mem_row_major);
		}
	}
	__syncthreads();
	// Find row-wise max
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block * 16 + row < T && col < T){
			const float score = scores_shared[row * T + col];
			atomicMax(reinterpret_cast<int*>(&row_max[row]), __float_as_int(score));
		}
	}
	__syncthreads();
	// Convert atomicMax result back to float
	if(threadIdx.x < 16){ row_max[threadIdx.x] = __int_as_float(static_cast<int>(row_max[threadIdx.x])); }
	__syncthreads();
	// Compute exp and sum
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block * 16 + row < T && col < T){
			const float val = expf(scores_shared[row * T + col] - row_max[row]);
			scores_shared[row * T + col] = val;
			atomicAdd(&row_sum[row], val);
		}
	}
	__syncthreads();
	// Normalize
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block * 16 + row < T && col < T){
			const float normalized = scores_shared[row * T + col] / row_sum[row];
			scores_shared[row * T + col] = normalized;
			// Store attention weights
			AttentionWeights[batch * H * T * T + head * T * T + (row_block * 16 + row) * T + col] = normalized;
		}
	}
	__syncthreads();
	// Step 3: Compute attention output
	// Distribute D blocks across warps
	for(int d_block_base = 0; d_block_base < (D + 15) / 16; d_block_base += num_warps){
		const int d_block = d_block_base + warp_id;
		if(d_block < (D + 15) / 16){
			fill_fragment(out_frag, 0.0f);
#pragma unroll
			for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
				// Load attention scores as matrix A
				fragment<matrix_a, 16, 16, 16, __half, row_major> att_frag;
				// Convert float scores to half
				const float* scores_ptr = scores_shared + col_block * 16;
#pragma unroll
				for(int i = 0; i < att_frag.num_elements; i++){
					const int idx = (i / 16) * T + (i % 16);
					att_frag.x[i] = __float2half(scores_ptr[idx]);
				}
				// Load V tile
#pragma unroll
				for(int d = lane_id; d < 16 * 16; d += 32){
					const int row = d / 16;
					const int col = d % 16;
					if(col_block * 16 + row < T && d_block * 16 + col < D){ V_shared[row * 16 + col] = V[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col]; } else{ V_shared[row * 16 + col] = __float2half(0.0f); }
				}
				__syncwarp();
				load_matrix_sync(v_frag, V_shared, 16);
				mma_sync(out_frag, att_frag, v_frag, out_frag);
			}
			// Store output
#pragma unroll
			for(int i = 0; i < out_frag.num_elements; i++){
				const int row = i / 16;
				const int col = i % 16;
				if(row_block * 16 + row < T && d_block * 16 + col < D){ Out[batch_head_offset + (row_block * 16 + row) * D + d_block * 16 + col] = __float2half(out_frag.x[i]); }
			}
		}
	}
}
extern "C" void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int B, int T, int D, int H){
	// Use 128 threads (4 warps) for better parallelism
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	// Calculate shared memory size
	size_t shared_size = sizeof(__half) * (16 * D * 3) + // Q, K, V tiles
		sizeof(float) * (16 * T) + // scores
		sizeof(float) * 32; // row_max + row_sum
	// Check shared memory limit for Volta (96KB max)
	cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	WmmaAttentionKernel<<<grid, block, shared_size>>>(Q, K, V, Out, AttentionWeights, B, T, D, H);
}
// Optimized WMMA Attention Backward Kernel for Volta
__global__ void WmmaAttentionBackwardKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ AttentionWeights, __half* __restrict__ dQ, __half* __restrict__ dK, __half* __restrict__ dV, int B, int T, int D, int H){
	// Block indices
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int row_block = blockIdx.x;
	// Early exit
	if(row_block * 16 >= T) return;
	// Warp and lane IDs
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	const int num_warps = blockDim.x / 32;
	// Calculate base offsets
	const int batch_head_offset = (batch * H + head) * T * D;
	const int att_offset = (batch * H + head) * T * T;
	// Shared memory layout optimized for bank conflict avoidance
	extern __shared__ char shared_mem_bytes[];
	__half* dOut_shared = reinterpret_cast<__half*>(shared_mem_bytes);
	__half* V_shared = dOut_shared + 16 * ((D + 7) & ~7); // Pad to avoid bank conflicts
	__half* K_shared = V_shared + 16 * ((D + 7) & ~7);
	__half* Q_shared = K_shared + 16 * ((D + 7) & ~7);
	float* att_shared = reinterpret_cast<float*>(Q_shared + 16 * ((D + 7) & ~7));
	float* dAtt_shared = att_shared + 16 * ((T + 3) & ~3); // Pad for alignment
	float* rowsum_shared = dAtt_shared + 16 * ((T + 3) & ~3);
	__half* workspace_half = reinterpret_cast<__half*>(rowsum_shared + 16);
	float* workspace_float = reinterpret_cast<float*>(workspace_half + 512);
	// WMMA fragments
	fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
	fragment<accumulator, 16, 16, 16, float> c_frag;
	const int D_padded = (D + 7) & ~7;
	const int T_padded = (T + 3) & ~3;
	// Initialize rowsum
	if(threadIdx.x < 16){ rowsum_shared[threadIdx.x] = 0.0f; }
	// Load dOut tile with coalesced access
#pragma unroll
	for(int d = threadIdx.x; d < 16 * D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(row_block * 16 + row < T && col < D){ dOut_shared[row * D_padded + col] = dOut[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ dOut_shared[row * D_padded + col] = __float2half(0.0f); }
	}
	// Load attention weights with coalesced access
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(row_block * 16 + row < T && col < T){ att_shared[row * T_padded + col] = AttentionWeights[att_offset + (row_block * 16 + row) * T + col]; } else{ att_shared[row * T_padded + col] = 0.0f; }
	}
	__syncthreads();
	// Step 1: Compute dV = A^T * dOut
	// Use shared memory accumulation to reduce atomics
	for(int col_block = warp_id; col_block < (T + 15) / 16; col_block += num_warps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(c_frag, 0.0f);
			// Load attention weights for WMMA (transposed view)
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[lane_id * 16 + i] = __float2half(col_block * 16 + lane_id < T ? att_shared[i * T_padded + col_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			// Load fragments and compute
			load_matrix_sync(a_frag, workspace_half, 16);
			load_matrix_sync(b_frag, dOut_shared + d_block * 16, D_padded);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			// Store to float workspace first
			store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
			__syncwarp();
			// Coalesced write to global memory - convert float to half
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dV[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
			}
		}
	}
	__syncthreads();
	// Step 2: Compute dA = dOut * V^T
	for(int col_block = warp_id; col_block < (T + 15) / 16; col_block += num_warps){
		fill_fragment(c_frag, 0.0f);
		// Load V tile with coalesced access
#pragma unroll
		for(int d = lane_id; d < D * 16; d += 32){
			int row = d / D;
			int col = d % D;
			if(col_block * 16 + row < T && col < D){ V_shared[row * D_padded + col] = V[batch_head_offset + (col_block * 16 + row) * D + col]; } else{ V_shared[row * D_padded + col] = __float2half(0.0f); }
		}
		__syncwarp();
		// Compute dOut * V^T using multiple WMMA operations
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			// Prepare V transpose in workspace
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				workspace_half[col * 16 + row] = V_shared[row * D_padded + d_block * 16 + col];
			}
			__syncwarp();
			load_matrix_sync(a_frag, dOut_shared + d_block * 16, D_padded);
			load_matrix_sync(b_frag, workspace_half, 16);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
		// Store result to float workspace
		store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
		__syncwarp();
		// Copy to dAtt_shared (already float)
#pragma unroll
		for(int i = lane_id; i < 16 * 16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(col_block * 16 + col < T){ dAtt_shared[row * T_padded + col_block * 16 + col] = workspace_float[row * 16 + col]; }
		}
	}
	__syncthreads();
	// Step 3: Softmax backward - compute row sums
#pragma unroll
	for(int row = 0; row < 16; row++){
		float sum = 0.0f;
		for(int col = threadIdx.x; col < T; col += blockDim.x){ if(row_block * 16 + row < T){ sum += dAtt_shared[row * T_padded + col] * att_shared[row * T_padded + col]; } }
		// Warp reduction
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ sum += __shfl_down_sync(0xffffffff, sum, offset); }
		if(lane_id == 0 && threadIdx.x / 32 == 0){ rowsum_shared[row] = sum; }
	}
	__syncthreads();
	// Apply softmax backward
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(row_block * 16 + row < T && col < T){
			float a = att_shared[row * T_padded + col];
			float da = dAtt_shared[row * T_padded + col];
			dAtt_shared[row * T_padded + col] = a * (da - rowsum_shared[row]);
		}
	}
	__syncthreads();
	// Load Q tile
#pragma unroll
	for(int d = threadIdx.x; d < 16 * D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(row_block * 16 + row < T && col < D){ Q_shared[row * D_padded + col] = Q[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ Q_shared[row * D_padded + col] = __float2half(0.0f); }
	}
	__syncthreads();
	// Step 4: Compute dQ = dS * K / sqrt(D)
	const float scale = rsqrtf((float)D);
	for(int d_block = warp_id; d_block < (D + 15) / 16; d_block += num_warps){
		fill_fragment(c_frag, 0.0f);
		for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
			// Load K tile
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block * 16 + row < T && d_block * 16 + col < D){ workspace_half[row * 16 + col] = K[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col]; } else{ workspace_half[row * 16 + col] = __float2half(0.0f); }
			}
			__syncwarp();
			// Load dS fragment
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[256 + i * 16 + lane_id] = __float2half(col_block * 16 + lane_id < T ? dAtt_shared[i * T_padded + col_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(a_frag, workspace_half + 256, 16);
			load_matrix_sync(b_frag, workspace_half, 16);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
		// Scale and store to float workspace
#pragma unroll
		for(int i = 0; i < c_frag.num_elements; i++){ c_frag.x[i] *= scale; }
		store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
		__syncwarp();
		// Write to global dQ - convert float to half
#pragma unroll
		for(int i = lane_id; i < 16 * 16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(row_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dQ[batch_head_offset + (row_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
		}
	}
	// Step 5: Compute dK = dS^T * Q / sqrt(D)
	for(int k_row_block = warp_id; k_row_block < (T + 15) / 16; k_row_block += num_warps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(c_frag, 0.0f);
			// Prepare dS^T in workspace
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[lane_id * 16 + i] = __float2half(k_row_block * 16 + lane_id < T ? dAtt_shared[i * T_padded + k_row_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(a_frag, workspace_half, 16);
			load_matrix_sync(b_frag, Q_shared + d_block * 16, D_padded);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			// Scale
#pragma unroll
			for(int i = 0; i < c_frag.num_elements; i++){ c_frag.x[i] *= scale; }
			store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
			__syncwarp();
			// Write to global dK - convert float to half
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(k_row_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dK[batch_head_offset + (k_row_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
			}
		}
	}
}
extern "C" void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* AttentionWeights, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H){
	// Initialize gradients to zero
	cudaMemset(dQ, 0, B * H * T * D * sizeof(__half));
	cudaMemset(dK, 0, B * H * T * D * sizeof(__half));
	cudaMemset(dV, 0, B * H * T * D * sizeof(__half));
	// Use 128 threads (4 warps) for Volta
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	// Calculate shared memory size with padding
	int D_padded = (D + 7) & ~7;
	int T_padded = (T + 3) & ~3;
	size_t shared_size = sizeof(__half) * (16 * D_padded * 4) + // dOut, V, K, Q tiles
		sizeof(float) * (16 * T_padded * 2) + // att_shared, dAtt_shared  
		sizeof(float) * 16 + // rowsum
		sizeof(__half) * 512 + // workspace_half
		sizeof(float) * 256; // workspace_float for WMMA accumulator storage
	// Configure for Volta's 96KB shared memory limit
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	// Add L1 cache configuration for Volta
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
	WmmaAttentionBackwardKernel<<<grid, block, shared_size>>>(Q, K, V, dOut, AttentionWeights, dQ, dK, dV, B, T, D, H);
}