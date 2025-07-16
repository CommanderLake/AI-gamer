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
	TPB = TPB/warps*warps;
	TPG = warps;
	while(TPG*2 <= TPB && TPG < warps){ TPG *= 2; }
	const int groups = TPB/TPG;
	RPB = sqrt(groups);
	CPB = groups/RPB;
	while(RPB*CPB < groups){ if(RPB < CPB){ RPB++; } else{ CPB++; } }
	curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator_, static_cast<unsigned long long>(time(nullptr)));
}
__device__ __host__ int DivCeil(const int a, const int b){ return a % b != 0 ? a/b + 1 : a/b; }
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
	for(int i = blockDim.x/2; i > 0; i >>= 1){
		if(tid < i){
			sdata[tid] += sdata[tid + i];
		}
		__syncthreads();
	}
	if(tid == 0){
		atomicAdd(&d_loss, sdata[0]);
	}
}
extern "C" float MseLoss(const __half* dPredictions, const float* dTargets, int size){
	constexpr auto zero = 0.0f;
	cudaMemcpyToSymbol(d_loss, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	int gridSize = DivCeil(size, BS);
	mseLossKernel<<<gridSize, BS, BS*sizeof(float)>>>(dPredictions, dTargets, size);
	float h_loss;
	cudaMemcpyFromSymbol(&h_loss, d_loss, sizeof(float));
	return h_loss/size;
}
__device__ float dLossKeys;
__device__ float dLossMouse;
__global__ void mseLoss2Kernel(const __half* predictions, const float* targets, const int size, const int numKeys, const int numCtrls){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	float sumKeys = 0.0f;
	float sumMouse = 0.0f;
	while(idx<size){
		const float pred = __half2float(predictions[idx]);
		const float target = targets[idx];
		const bool isKey = idx%numCtrls<numKeys;
		const float diff = isKey ? pred>=0.5f!=target>=0.5f : (pred-target)*(pred-target);
		sumKeys += diff*isKey;
		sumMouse += diff*!isKey;
		idx += gridDim.x*blockDim.x;
	}
	sdata[tid] = sumKeys;
	sdata[tid+blockDim.x] = sumMouse;
	__syncthreads();
	for(int s = blockDim.x/2; s>0; s >>= 1){
		if(tid<s){
			sdata[tid] += sdata[tid+s];
			sdata[tid+blockDim.x] += sdata[tid+blockDim.x+s];
		}
		__syncthreads();
	}
	if(tid==0){
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
	*axesLoss /= (numCtrls-numButs)*batchSize;
}
extern "C" void BlockShiftHalf(__half* hPtr, const int shiftBy, const int blocksToShift){
	auto blockSize = shiftBy;
	if(blockSize < 0) blockSize = -blockSize;
	if(shiftBy > 0){
		for(int i = blocksToShift; 0<i; --i){
			cudaMemcpy(hPtr + i*blockSize, hPtr + (i - 1)*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice);
		}
	} else{
		for(int i = 0; i<blocksToShift; ++i){
			cudaMemcpy(hPtr + (i - 1)*blockSize, hPtr + i*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice);
		}
	}
}
__global__ void ConvertByteToHalfNormKernel(const unsigned char* input, __half* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		output[idx] = __float2half(static_cast<float>(input[idx])/255.0f);
	}
}
__global__ void ConvertByteToHalfKernel(const unsigned char* input, __half* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		output[idx] = __float2half(input[idx]);
	}
}
extern "C" void ConvertByteToHalf(const unsigned char* input, __half* output, const size_t size, bool normalize){
	int blocks, tpb = 256;
	GetLaunchConfig(size, blocks, tpb);
	if(normalize) ConvertByteToHalfNormKernel<<<blocks, tpb>>>(input, output, size);
	else ConvertByteToHalfKernel<<<blocks, tpb>>>(input, output, size);
}

__global__ void ConvertHalfToByteNormKernel(const __half* input, unsigned char* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		output[idx] = static_cast<unsigned char>(__half2float(input[idx])*255.0f);
	}
}
__global__ void ConvertHalfToByteKernel(const __half* input, unsigned char* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		output[idx] = static_cast<unsigned char>(__half2float(input[idx]));
	}
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
	for(int idx = blockIdx.x*blockDim.x+threadIdx.x; idx<size; idx += stride){ halfWeights[idx] = __float2half(weights[idx]*scale); }
}
extern "C" void ConvertFloatToHalfScale(__half* halfWeights, const float* weights, const size_t size, const float scale){
	int blocks, tpb = 256;
	GetLaunchConfig(size, blocks, tpb);
	ConvertFloatToHalfScaleKernel<<<blocks, tpb>>>(halfWeights, weights, size, scale);
}
extern "C" void WeightInit(__half* weightHalf, const int numWeights, const int fanIn, const int fanOut, const WeightInitMethod method){
	if(method==Orthogonal){ OrthogonalInit(weightHalf, fanIn, fanOut); } else{
		float* weightFloat;
		checkCUDA(cudaMalloc(&weightFloat, numWeights*sizeof(float)));
		const float factor = method==Xavier ? 1.0f : 2.0f;
		curandGenerateNormal(generator_, weightFloat, numWeights, 0.0f, 1.0f);
		ConvertFloatToHalfScale(weightHalf, weightFloat, numWeights, sqrtf(factor/fanIn));
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
	if(threadIdx.x==0){
		const float biasCorrection1 = 1.0f-powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f-powf(BETA2_F, t);
		sLrT = lr/biasCorrection1;
		sBeta1Complement = 1.0f-BETA1_F;
		sBeta3Complement = 1.0f-BETA2_F;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
#pragma unroll 4
	for(int i = idx; i<n/4; i += stride){
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
		mNew.x = BETA1_F*m4.x+sBeta1Complement*gradClipped.x;
		mNew.y = BETA1_F*m4.y+sBeta1Complement*gradClipped.y;
		mNew.z = BETA1_F*m4.z+sBeta1Complement*gradClipped.z;
		mNew.w = BETA1_F*m4.w+sBeta1Complement*gradClipped.w;
		// Update velocity
		float4 vNew;
		vNew.x = BETA2_F*v4.x+sBeta3Complement*gradClipped.x*gradClipped.x;
		vNew.y = BETA2_F*v4.y+sBeta3Complement*gradClipped.y*gradClipped.y;
		vNew.z = BETA2_F*v4.z+sBeta3Complement*gradClipped.z*gradClipped.z;
		vNew.w = BETA2_F*v4.w+sBeta3Complement*gradClipped.w*gradClipped.w;
		// Apply weight decay
		params4.x *= 1.0f-wd;
		params4.y *= 1.0f-wd;
		params4.z *= 1.0f-wd;
		params4.w *= 1.0f-wd;
		// Compute updates
		float4 update;
		update.x = sLrT*mNew.x/(sqrtf(vNew.x/sBiasCorrection2)+EPSILON_F);
		update.y = sLrT*mNew.y/(sqrtf(vNew.y/sBiasCorrection2)+EPSILON_F);
		update.z = sLrT*mNew.z/(sqrtf(vNew.z/sBiasCorrection2)+EPSILON_F);
		update.w = sLrT*mNew.w/(sqrtf(vNew.w/sBiasCorrection2)+EPSILON_F);
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
	const int remainStart = n/4*4;
	for(int i = remainStart+idx; i<n; i += stride){
		const float grad = fmaxf(fminf(grads[i], CLIP), -CLIP);
		const float mVal = BETA1_F*m[i]+sBeta1Complement*grad;
		const float vVal = BETA2_F*v[i]+sBeta3Complement*grad*grad;
		params[i] = (params[i]-sLrT*mVal/(sqrtf(vVal/sBiasCorrection2)+EPSILON_F))*(1.0f-wd);
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
	if(threadIdx.x==0){
		sBiasCorrection1 = 1.0f-powf(BETA1_F, t);
		sBiasCorrection2 = 1.0f-powf(BETA2_F, t);
		sLrT = lr/sBiasCorrection1;
		sBeta1Complement = 1.0f-BETA1_F;
		sBeta3Complement = 1.0f-BETA2_F;
		sWeightDecay = 1.0f-wd;
	}
	__syncthreads();
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	// Process 4 half2 elements (8 total elements) at once
#pragma unroll 4
	for(int i = idx; i<n/8; i += stride){
		// Load 4 half2 pairs (8 elements)
		const auto paramsPtr = reinterpret_cast<__half2*>(params+8*i);
		const auto gradsPtr = reinterpret_cast<const __half2*>(grads+8*i);
		const auto mPtr = reinterpret_cast<__half2*>(m+8*i);
		const auto vPtr = reinterpret_cast<__half2*>(v+8*i);
		// Load 4 pairs of values
		__half2 paramsH2[4], gradsH2[4], mH2[4], vH2[4];
#pragma unroll
		for(int j = 0; j<4; j++){
			paramsH2[j] = paramsPtr[j];
			gradsH2[j] = gradsPtr[j];
			mH2[j] = mPtr[j];
			vH2[j] = vPtr[j];
		}
		// Convert to float2 for computations
		float2 paramsF2[4], gradsF2[4], mF2[4], vF2[4];
#pragma unroll
		for(int j = 0; j<4; j++){
			paramsF2[j] = __half22float2(paramsH2[j]);
			gradsF2[j] = __half22float2(gradsH2[j]);
			mF2[j] = __half22float2(mH2[j]);
			vF2[j] = __half22float2(vH2[j]);
		}
		// Process each pair
#pragma unroll
		for(int j = 0; j<4; j++){
			// Clip gradients
			gradsF2[j].x = fmaxf(fminf(gradsF2[j].x, CLIP), -CLIP);
			gradsF2[j].y = fmaxf(fminf(gradsF2[j].y, CLIP), -CLIP);
			// Update momentum
			mF2[j].x = BETA1_F*mF2[j].x+sBeta1Complement*gradsF2[j].x;
			mF2[j].y = BETA1_F*mF2[j].y+sBeta1Complement*gradsF2[j].y;
			// Update velocity
			vF2[j].x = BETA2_F*vF2[j].x+sBeta3Complement*gradsF2[j].x*gradsF2[j].x;
			vF2[j].y = BETA2_F*vF2[j].y+sBeta3Complement*gradsF2[j].y*gradsF2[j].y;
			// Apply weight decay
			paramsF2[j].x *= sWeightDecay;
			paramsF2[j].y *= sWeightDecay;
			// Compute denominator
			float2 denom;
			denom.x = sqrtf(vF2[j].x/sBiasCorrection2)+EPSILON_F;
			denom.y = sqrtf(vF2[j].y/sBiasCorrection2)+EPSILON_F;
			// Update parameters
			paramsF2[j].x -= sLrT*mF2[j].x/denom.x;
			paramsF2[j].y -= sLrT*mF2[j].y/denom.y;
			// Convert back to half2
			paramsH2[j] = __float22half2_rn(paramsF2[j]);
			mH2[j] = __float22half2_rn(mF2[j]);
			vH2[j] = __float22half2_rn(vF2[j]);
		}
		// Store results
#pragma unroll
		for(int j = 0; j<4; j++){
			paramsPtr[j] = paramsH2[j];
			mPtr[j] = mH2[j];
			vPtr[j] = vH2[j];
		}
	}
	// Handle remaining elements
	const int remainStart = n/8*8;
	for(int i = remainStart+idx; i<n; i += stride){
		const float grad = fmaxf(fminf(__half2float(grads[i]), CLIP), -CLIP);
		const float mVal = BETA1_F*__half2float(m[i])+sBeta1Complement*grad;
		const float vVal = BETA2_F*__half2float(v[i])+sBeta3Complement*grad*grad;
		const float param = __half2float(params[i])*sWeightDecay;
		const float denom = sqrtf(vVal/sBiasCorrection2)+EPSILON_F;
		params[i] = __float2half(param-sLrT*mVal/denom);
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
	if(idx < size){
		grads[idx] = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx]))));
	}
}
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size){
	auto gridSize = DivCeil(size, BS);
	GradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size);
}
__global__ void SplitGradKernel(__half* gradients, const __half* predictions, const float* targets, const float clip, const int numCtrls, const int numButs, const int batchSize, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx/numCtrls;
		const int ctrlId = idx%numCtrls;
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
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		const int batchId = idx/numCtrls;
		const int ctrlId = idx%numCtrls;
		if(ctrlId<numButs){
			predOut[idx] = buttonData[batchId*numButs+ctrlId];
		} else{
			predOut[idx] = axisData[batchId*(numCtrls - numButs)+(ctrlId-numButs)];
		}
	}
}
extern "C" void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, const int numCtrls, const int numButs, const int size){
	auto gridSize = DivCeil(size, BS);
	MergeOutputsKernel<<<gridSize, BS>>>(predOut, buttonData, axisData, size, numCtrls, numButs);
}
__global__ void GetPredictionKernel(const __half* predBatch, float* prediction, const int numCtrls, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < numCtrls){
		prediction[idx] = __half2float(predBatch[idx + size - numCtrls]);
	}
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
			gradient = (pClamped - 1.0f)/pClamped;
			gradients[idx] = __float2half(gradient*scale);
		} else{
			gradient = pClamped/(1.0f - pClamped);
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
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		grad[idx] *= __float2half(__half2float(dataIn[idx]) < 0.0f ? negativeSlope : 1.0f);
	}
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
		outData[idx] = __float2half(val/(1.0f + exp(-val)));
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
		const auto sig = val/(1.0f + val);
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
	dataOut[idx] = __float2half(1.0f/(1.0f + expf(-val)));
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
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	for(int i = idx; i<size; i += stride){
		const float x= __half2float(dataIn[i]);
		const float cdf = 0.5f*(1.0f+tanhf(SQRT_2_PI*(x+GELU_COEF_A*x*x*x)));
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
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int stride = blockDim.x*gridDim.x;
	for(int i = idx; i<size; i += stride){
		const float x = __half2float(dataIn[i]);
		const float xCu = x*x*x;
		const float cdf = 0.5f*(1.0f+tanhf(SQRT_2_PI*(x+GELU_COEF_A*xCu)));
		const float pdf = SQRT_2_PI*(1.0f+3.0f*GELU_COEF_A*x*x)*(1.0f-tanhf(SQRT_2_PI*(x+GELU_COEF_A*xCu))*tanhf(SQRT_2_PI*(x+GELU_COEF_A*xCu)))*0.5f;
		const float result = __half2float(grad[i])*(cdf+x*pdf);
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
	float* sMean = sdata, *sM2 = sMean + blockDim.x, *sCount = sM2 + blockDim.x;
	const int tid = threadIdx.x, cid = blockIdx.x;
	if(cid>=C) return;
	float tMean = 0.0f, tM2 = 0.0f, count = 0.0f;
	for(int n = 0; n<N; ++n){
		for(int i = tid; i<HW; i += blockDim.x){
			const int idx = n*C*HW+cid*HW+i;
			const float val = __half2float(dataIn[idx]);
			++count;
			const float delta = val - tMean;
			tMean += delta / count;
			tM2 += delta*(val - tMean);
		}
	}
	sMean[tid] = tMean; sM2[tid] = tM2; sCount[tid] = count;
	__syncthreads();
	for(int s = blockDim.x/2; s>0; s >>= 1){
		if(tid<s){
			const float m1 = sMean[tid], m2 = sMean[tid+s], c1 = sCount[tid], c2 = sCount[tid+s];
			const float delta = m2-m1, nCount = c1+c2;
			sMean[tid] = (m1*c1+m2*c2)/nCount;
			sM2[tid] += sM2[tid+s]+delta*delta*c1*c2/nCount;
			sCount[tid] = nCount;
		}
		__syncthreads();
	}
	if(tid==0){
		mean[cid] = sMean[0];
		variance[cid] = sM2[0]/sCount[0];
	}
}

__global__ void LayerNormForwardKernel(__half* __restrict__ output, const __half* __restrict__ data, const float* gamma, const float* beta, const float* mean, const float* variance, int N, int C, int HW){
	extern __shared__ float sParams[];
	float* sMean = sParams, *sVar = sMean+C, *sGamma = sVar+C, *sBeta = sGamma+C;
	for(int i = threadIdx.x; i<C; i += blockDim.x){
		sMean[i] = mean[i]; sVar[i] = variance[i];
		sGamma[i] = gamma[i]; sBeta[i] = beta[i];
	}
	__syncthreads();
	const int total = N*C*HW;
	for(int i = blockIdx.x*blockDim.x+threadIdx.x; i<total; i += blockDim.x*gridDim.x){
		const int c = i/HW%C;
		const float x = __half2float(data[i]);
		const float norm = (x-sMean[c])/sqrtf(sVar[c]+EPSILON_F);
		output[i] = __float2half(norm*sGamma[c]+sBeta[c]);
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
	if(err!=cudaSuccess){ printf("Kernel execution failed: %s\n", cudaGetErrorString(err)); }
}

__global__ void LayerNormBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW){
	extern __shared__ float sdata[];
	float* sGradGamma = sdata, *sGradBeta = sGradGamma+blockDim.x;
	const int tid = threadIdx.x, cid = blockIdx.x;
	if(cid>=C) return;
	float tGradGamma = 0.0f, tGradBeta = 0.0f;
	const float invStd = rsqrtf(variance[cid]+EPSILON_F);
	for(int n = 0; n<N; ++n){
		for(int i = tid; i<HW; i += blockDim.x){
			const int idx = n*C*HW+cid*HW+i;
			const float x = __half2float(data[idx]);
			const float dy = __half2float(grad[idx]);
			const float xHat = (x-mean[cid])*invStd;
			tGradGamma += xHat*dy;
			tGradBeta += dy;
		}
	}
	sGradGamma[tid] = tGradGamma; sGradBeta[tid] = tGradBeta;
	__syncthreads();
	for(int s = blockDim.x/2; s>0; s >>= 1){
		if(tid<s){
			sGradGamma[tid] += sGradGamma[tid+s];
			sGradBeta[tid] += sGradBeta[tid+s];
		}
		__syncthreads();
	}
	if(tid==0){
		atomicAdd(&gradGamma[cid], sGradGamma[0]);
		atomicAdd(&gradBeta[cid], sGradBeta[0]);
	}
	__syncthreads();
	const float gGamma = gradGamma[cid], gBeta = gradBeta[cid];
	for(int n = 0; n<N; ++n){
		for(int i = tid; i<HW; i += blockDim.x){
			const int idx = n*C*HW+cid*HW+i;
			const float x = __half2float(data[idx]);
			const float dy = __half2float(grad[idx]);
			const float xHat = (x-mean[cid])*invStd;
			const float gradInput = gamma[cid]*invStd*(dy-(xHat*gGamma+gBeta)/(N*HW));
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
	if(err!=cudaSuccess){ printf("Kernel execution failed: %s\n", cudaGetErrorString(err)); }
}

__device__ int deviceResult;
__global__ void isNaNKernel(const __half* __restrict__ data, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){
		atomicExch(&deviceResult, 1);
	}
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
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= size) return;
	int i = idx;
	const int w = i%inW;
	i /= inW;
	const int h = i%inH;
	i /= inH;
	const int cPrime = i%attC;
	i /= attC;
	const int n = i;
	float sum = 0.f;
	const int qBase = ((n*attC+cPrime)*inH+h)*inW+w;
	for(int c = 0; c<inC; ++c){
		const int kBase = ((n*inC+c)*inH+h)*inW+w;
		const float qVal = __half2float(queryMap[qBase]);
		const float kVal = __half2float(keyMap[kBase]);
		sum += qVal*kVal;
	}
	sum *= 1.0f/sqrtf(static_cast<float>(inC));
	attentionScores[idx] = __float2half(sum);
}
extern "C" void ComputeAttention(const __half* __restrict__ queryMap, const __half* __restrict__ keyMap, __half* __restrict__ attentionScores, int inC, int attC, int inH, int inW){
	const auto size = inC*attC*inH*inW;
	auto gridSize = DivCeil(size, BS);
	ComputeAttentionKernel<<<gridSize, BS>>>(queryMap, keyMap, attentionScores, inC, attC, inH, inW, size);
}
__global__ void ApplyAttentionKernel(const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, __half* __restrict__ output, int inC, int attC, int inH, int inW, int size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= size) return;
	int i = idx;
	const int w = i%inW;
	i /= inW;
	const int h = i%inH;
	i /= inH;
	const int c = i%inC;
	i /= inC;
	const int n = i;
	const int valIdx = ((n*inC+c)*inH+h)*inW+w;
	const float val = __half2float(valueMap[valIdx]);
	float sum = 0.f;
	for(int ac = 0; ac<attC; ++ac){
		const int attIdx = ((n*attC+ac)*inH+h)*inW+w;
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
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= size) return;
	int i = idx;
	const int w = i%inW;
	i /= inW;
	const int h = i%inH;
	i /= inH;
	//int c = i%inC;
	i /= inC;
	const int n = i;
	const float dOut = __half2float(gradIn[idx]);
	const float val = __half2float(valueMap[idx]);
	float gradValSum = 0.f;
	for(int ac = 0; ac<attC; ++ac){
		const int attIdx = ((n*attC+ac)*inH+h)*inW+w;
		gradValSum += dOut*__half2float(attentionScores[attIdx]);
	}
	gradValue[idx] = __float2half(gradValSum);
	for(int ac = 0; ac<attC; ++ac){
		const int attIdx = ((n*attC+ac)*inH+h)*inW+w;
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
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx>=size) return;
	const int w = idx%inW;
	int tmp = idx/inW;
	const int h = tmp%inH;
	tmp /= inH;
	const int cPrime = tmp%attC;
	tmp /= attC;
	const int c = tmp%inC;
	const int n = tmp/inC;
	const int attIdx = ((n*attC+cPrime)*inH+h)*inW+w;
	float gAttVal = __half2float(gradAttention[attIdx]);
	const float scale = 1.0f/sqrtf(static_cast<float>(inC));
	gAttVal *= scale;
	const float qVal = __half2float(queryMap[attIdx]);
	const int keyIdx = ((n*inC+c)*inH+h)*inW+w;
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
	if(nc>=N*C) return;
	const int n = nc/C;
	const int c = nc%C;
	const int HW = H*W;
	__shared__ float sMax;
	__shared__ float sSumExp;
	const int baseOffset = (n*C+c)*HW;
	float threadMax = -FLT_MAX;
	for(int tid = threadIdx.x; tid<HW; tid += blockSize){
		const float val = __half2float(inData[baseOffset+tid]);
		if(val>threadMax){ threadMax = val; }
	}
	__shared__ float buf[1024];
	buf[threadIdx.x] = threadMax;
	__syncthreads();
	for(int offset = blockSize/2; offset>0; offset >>= 1){
		if(threadIdx.x<offset){
			const float other = buf[threadIdx.x+offset];
			if(other>buf[threadIdx.x]){ buf[threadIdx.x] = other; }
		}
		__syncthreads();
	}
	if(threadIdx.x==0){ sMax = buf[0]; }
	__syncthreads();
	float threadSumExp = 0.0f;
	for(int tid = threadIdx.x; tid<HW; tid += blockSize){
		const float val = __half2float(inData[baseOffset+tid]);
		const float expVal = expf(val-sMax);
		threadSumExp += expVal;
	}
	buf[threadIdx.x] = threadSumExp;
	__syncthreads();
	for(int offset = blockSize/2; offset>0; offset >>= 1){
		if(threadIdx.x<offset){ buf[threadIdx.x] += buf[threadIdx.x+offset]; }
		__syncthreads();
	}
	if(threadIdx.x==0){ sSumExp = buf[0]; }
	__syncthreads();
	for(int tid = threadIdx.x; tid<HW; tid += blockSize){
		const float val = __half2float(inData[baseOffset+tid]);
		const float expVal = expf(val-sMax);
		const float softmaxVal = expVal/sSumExp;
		outData[baseOffset+tid] = __float2half(softmaxVal);
	}
}
extern "C" void SpatialSoftmaxHalf(const __half* inData, __half* outData, int N, int C, int H, int W){
	auto gridSize = DivCeil(N*C, 1);
	SpatialSoftmaxKernelHalf<<<gridSize, BS, 1040>>>(inData, outData, N, C, H, W);
}
__global__ void SpatialSoftmaxBackwardHalfKernel(const __half* __restrict__ outData, const __half* __restrict__ graIn, __half* __restrict__ gradOut, int N, int C, int H, int W){
	const int blockSize = blockDim.x;
	const int nc = blockIdx.x;
	if(nc>=N*C) return;
	const int n = nc/C;
	const int c = nc%C;
	const int hw = H*W;
	__shared__ float sSumG;
	__shared__ float buf[1024];
	const int baseOffset = (n*C+c)*hw;
	float threadSum = 0.0f;
	for(int tid = threadIdx.x; tid<hw; tid += blockSize){
		const float y = __half2float(outData[baseOffset+tid]);
		const float dO = __half2float(graIn[baseOffset+tid]);
		threadSum += y*dO;
	}
	buf[threadIdx.x] = threadSum;
	__syncthreads();
	for(int offset = blockSize/2; offset>0; offset >>= 1){
		if(threadIdx.x<offset){ buf[threadIdx.x] += buf[threadIdx.x+offset]; }
		__syncthreads();
	}
	if(threadIdx.x==0){ sSumG = buf[0]; }
	__syncthreads();
	for(int tid = threadIdx.x; tid<hw; tid += blockSize){
		const float y = __half2float(outData[baseOffset+tid]);
		const float dO = __half2float(graIn[baseOffset+tid]);
		const float dI = y*(dO-sSumG);
		gradOut[baseOffset+tid] = __float2half(dI);
	}
}
extern "C" void SpatialSoftmaxBackwardHalf(const __half* outData, const __half* gradIn, __half* gradOut, int N, int C, int H, int W){
	int gridSize = DivCeil(N*C, 1);
	SpatialSoftmaxBackwardHalfKernel<<<gridSize, BS, 1032>>>(outData, gradIn, gradOut, N, C, H, W);
}
__global__ void FeatureMapMosaicKernel(const __half* __restrict__ input, unsigned char* __restrict__ output, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW){
	const int c = blockIdx.x*blockDim.x+threadIdx.x; // Channel index
	const int y = blockIdx.y*blockDim.y+threadIdx.y; // Y-coordinate in feature map
	const int x = blockIdx.z*blockDim.z+threadIdx.z; // X-coordinate in feature map
	if(c>=inC||y>=H||x>=W) return;
	// Calculate tile position using provided grid dimensions
	const int tileX = c%gridW;
	const int tileY = c/gridW;
	// Compute destination position in the mosaic
	const int outX = tileX*tileW+x;
	const int outY = tileY*tileH+y;
	// Convert FP16 to 8-bit unsigned char
	const __half value = input[c*H*W+y*W+x];
	const float fVal = __half2float(value);
	const unsigned char pixel = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, fVal*255.0f)));
	// Store to output
	output[outY*mosaicW+outX] = pixel;
}
extern "C" void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, cudaStream_t stream){
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((inC+blockDim.x-1)/blockDim.x, (H+blockDim.y-1)/blockDim.y, (W+blockDim.z-1)/blockDim.z);
	FeatureMapMosaicKernel<<<gridDim, blockDim, 0, stream>>>(dInput, dOutput, H, W, inC, mosaicW, tileW, tileH, gridW);
}
#include <mma.h>
__global__ void WmmaAttentionKernel(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ Out, int B, int T, int D, int H){
	const int head = blockIdx.z;
	const int b = blockIdx.y;
	const int t = blockIdx.x*16 + threadIdx.y;
	if(t>=T) return;
	extern __shared__ half shared[];
	half* tileQ = shared;
	half* tileK = shared + D*16;
	half* tileV = shared + 2*D*16;
	const int hOffset = head*D;
	const half* Qptr = Q + (b*T + t)*D + hOffset;
	const half* Kptr = K + hOffset;
	const half* Vptr = V + hOffset;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> fragA;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> fragB;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> fragC;
	fill_fragment(fragC, __float2half(0.f));
	for(int d = 0; d<D; d += 16){
		load_matrix_sync(fragA, Qptr+d, D);
		load_matrix_sync(fragB, Kptr+d*T, T);
		mma_sync(fragC, fragA, fragB, fragC);
	}
	half attention[16];
	for(int i = 0; i<fragC.num_elements; i++){ attention[i] = fragC.x[i]; }
	__syncthreads();
	for(int i = 0; i<T; i += 16){
		load_matrix_sync(fragB, Vptr + i*D, D);
		mma_sync(fragC, fragA, fragB, fragC);
	}
	store_matrix_sync(Out + (b*T + t)*D + hOffset, fragC, D, nvcuda::wmma::mem_row_major);
}
extern "C" void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, int B, int T, int D, int H){
	dim3 block(32, 16);
	dim3 grid((T+15)/16, B, H);
	size_t smem = 3*D*16*sizeof(__half);
	WmmaAttentionKernel<<<grid, block, smem>>>(Q, K, V, Out, B, T, D, H);
}