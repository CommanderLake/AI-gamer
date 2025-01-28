#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "common.h"
#include <cuda.h>
#include <curand.h>
__device__ __host__ int div_ceil(int a, int b){ return a % b != 0 ? a/b + 1 : a/b; }
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
curandGenerator_t gen;
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
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}
__global__ void cuARGBtoRGB(const pixARGB* src, pixRGB* dst, int n){
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x){
		dst[i].R = src[i].R;
		dst[i].G = src[i].G;
		dst[i].B = src[i].B;
	}
}
extern "C" cudaError ARGBtoRGB(unsigned char* src, unsigned char* dst, int n){
	cuARGBtoRGB<<<GS, BS>>>(reinterpret_cast<pixARGB*>(src), reinterpret_cast<pixRGB*>(dst), n);
	return cudaGetLastError();
}
__global__ void cuARGBtoRGBplanar(const unsigned char* src, unsigned char* dst, int n){
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x){
		const int srcIdx = i*4;      
		dst[i] = src[srcIdx + 2];   
		dst[i + n] = src[srcIdx + 1];   
		dst[i + 2*n] = src[srcIdx];   
	}
}
extern "C" cudaError ARGBtoRGBplanar(unsigned char* src, unsigned char* dst, int n){
	cuARGBtoRGBplanar<<<GS, BS>>>(src, dst, n);
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
	int gridSize = div_ceil(size, BS);
	mseLossKernel<<<gridSize, BS, BS*sizeof(float)>>>(dPredictions, dTargets, size);
	float h_loss;
	cudaMemcpyFromSymbol(&h_loss, d_loss, sizeof(float));
	return h_loss/size;
}
__device__ float dLossKeys;
__device__ float dLossMouse;
__global__ void mseLoss2Kernel(const __half* predictions, const float* targets, int size, int numKeys, int numCtrls){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	float sumKeys = 0.0f;
	float sumMouse = 0.0f;
	while(idx<size){
		float diff = __half2float(predictions[idx])-targets[idx];
		diff *= diff;
		if(idx%numCtrls<numKeys){ sumKeys += diff; } else{ sumMouse += diff; }
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
extern "C" void MseLoss2(const __half* dPredictions, const float* dTargets, int numButs, int numCtrls, int batchSize, float* butLoss, float* axesLoss){
	constexpr auto zero = 0.0f;
	const auto size = numCtrls*batchSize;
	cudaMemcpyToSymbol(dLossKeys, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dLossMouse, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	int gridSize = div_ceil(size, BS);
	mseLoss2Kernel<<<gridSize, BS, 2*BS*sizeof(float)>>>(dPredictions, dTargets, size, numButs, numCtrls);
	cudaMemcpyFromSymbol(butLoss, dLossKeys, sizeof(float));
	cudaMemcpyFromSymbol(axesLoss, dLossMouse, sizeof(float));
	*butLoss /= numButs*batchSize;
	*axesLoss /= (numCtrls-numButs)*batchSize;
}
__global__ void convertAndNormalizeKernel(__half* output, const unsigned char* input, const size_t size){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		output[idx] = __float2half(static_cast<float>(input[idx])/255.0f);
	}
}
extern "C" void ConvertAndNormalize(__half* output, const unsigned char* input, const size_t size){
	convertAndNormalizeKernel<<<GS, BS>>>(output, input, size);
}
__global__ void UnConvertAndUnNormalizeKernel(unsigned char* output, const __half* input, const size_t size){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		output[idx] = static_cast<unsigned char>(__half2float(input[idx])*255.0f);
	}
}
extern "C" void UnConvertAndUnNormalize(unsigned char* output, const __half* input, const size_t size){
	UnConvertAndUnNormalizeKernel<<<BS, BS>>>(output, input, size);
}
__global__ void convertFloatToHalfKernel(float* src, __half* dst, size_t n){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n){ dst[i] = __float2half(src[i]); }
}
extern "C" void ConvertFloatToHalf(float* src, __half* dst, size_t n){
	auto gridSize = div_ceil(n, BS);
	convertFloatToHalfKernel<<<gridSize, BS>>>(src, dst, n);
}
__global__ void convertHalfToFloatKernel(__half* src, float* dst, size_t n){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n){ dst[i] = __half2float(src[i]); }
}
extern "C" void ConvertHalfToFloat(__half* src, float* dst, size_t n){
	auto gridSize = div_ceil(n, BS);
	convertHalfToFloatKernel<<<gridSize, BS>>>(src, dst, n);
}
__global__ void HeInitKernel(__half* halfWeights, const float* weights, int n, float scale){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n){ halfWeights[i] = __float2half(weights[i]*scale); }
}
extern "C" void HeInit(__half* weightHalf, int numWeights, float fanIn){
	float* weightFloat;
	cudaMalloc(&weightFloat, numWeights*sizeof(float));
	curandGenerateNormal(gen, weightFloat, numWeights, 0.0f, 1.0f);
	auto gridSize = div_ceil(numWeights, BS);
	HeInitKernel<<<gridSize, BS>>>(weightHalf, weightFloat, numWeights, sqrtf(2.0f/fanIn));
	cudaFree(weightFloat);
}
#define BETA1_F 0.9f
#define BETA2_F 0.99f
#define BETA3_F 0.999f
#define EPSILON_F 1e-7f
#define CLIP 1.0f
__global__ void sgdHalfKernel(__half* params, const __half* grads, const int size, const float learningRate, const float weightDecay){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		params[idx] *= 1.0f - weightDecay;
		const float gradClipped = fmaxf(fminf(__half2float(grads[idx]), CLIP), -CLIP);
		params[idx] = __float2half(__half2float(params[idx]) - learningRate*gradClipped);
	}
}
extern "C" void SGDHalf(__half* params, const __half* grads, const int size, const float learningRate, const float weightDecay){
	auto gridSize = div_ceil(size, BS);
	sgdHalfKernel<<<gridSize, BS>>>(params, grads, size, learningRate, weightDecay);
}
__global__ void sgdFloatKernel(float* params, const float* grads, const int size, const float learningRate, const float weightDecay){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		params[idx] *= 1.0f - weightDecay;
		const float gradClipped = fmaxf(fminf(grads[idx], CLIP), -CLIP);
		params[idx] -= learningRate*gradClipped;
	}
}
extern "C" void SGDFloat(float* params, const float* grads, const int size, const float learningRate, const float weightDecay){
	auto gridSize = div_ceil(size, BS);
	sgdFloatKernel<<<gridSize, BS>>>(params, grads, size, learningRate, weightDecay);
}
__global__ void AdamwKernelFloat(float* __restrict__ params, const float* __restrict__ grads, float* __restrict__ m, float* __restrict__ v, const float lr, const int t, const float wd, const int n){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<n){
		const float grad = fmaxf(fminf(grads[idx], CLIP), -CLIP);
		m[idx] = BETA1_F*m[idx]+(1.0f-BETA1_F)*grad;
		v[idx] = BETA3_F*v[idx]+(1.0f-BETA3_F)*grad*grad;
		params[idx] *= 1.0f - wd;
		params[idx] -= lr*static_cast<float>(m[idx]/(1.0 - pow(BETA1_F, t))/(sqrt(v[idx]/(1.0 - pow(BETA3_F, t))) + EPSILON_F));
	}
}
extern "C" void AdamWFloat(float* params, const float* grads, float* m, float* v, const float learningRate, const int t, const float weightDecay, const int size){
	auto gridSize = div_ceil(size, BS);
	AdamwKernelFloat<<<gridSize, BS>>>(params, grads, m, v, learningRate, t, weightDecay, size);
}
__global__ void AdamwKernelHalf(__half* __restrict__ params, const __half* __restrict__ grads, __half* __restrict__ m, __half* __restrict__ v, const float lr, const int t, const float wd, const int n){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<n){
		const auto grad = fmaxf(fminf(__half2float(grads[idx]), CLIP), -CLIP);
		const auto mF = BETA1_F*__half2float(m[idx])+(1.0f-BETA1_F)*grad;
		const auto vF = BETA3_F*__half2float(v[idx])+(1.0f-BETA3_F)*grad*grad;
		const auto param = __half2float(params[idx])*(1.0f - wd);
		params[idx] = __float2half(param - lr*static_cast<float>(mF/(1.0 - pow(BETA1_F, t))/(sqrt(vF/(1.0 - pow(BETA3_F, t))) + EPSILON_F)));
		m[idx] = __float2half(mF);
		v[idx] = __float2half(vF);
	}
}
extern "C" void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, const float lr, const int t, const float weightDecay, const int size){
	auto gridSize = div_ceil(size, BS);
	AdamwKernelHalf<<<gridSize, BS>>>(params, grads, m, v, lr, t, weightDecay, size);
}
__global__ void AdanKernelFloat(float* __restrict__ params, const float* __restrict__ grads, float* __restrict__ m, float* __restrict__ v, float* __restrict__ n, float* __restrict__ velocity, const float lr, const float wd, const int size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		float grad = grads[idx];
		grad += params[idx]*wd;
		m[idx] = BETA1_F*m[idx]+(1-BETA1_F)*grad;
		v[idx] = BETA2_F*v[idx]+(1-BETA2_F)*grad*grad;
		n[idx] = BETA3_F*n[idx]+(1-BETA3_F)*grad*grad*grad;
		velocity[idx] = BETA1_F*velocity[idx]+lr*grad;
		const float mHat = m[idx]/(1-powf(BETA1_F, idx+1));
		const float vHat = v[idx]/(1-powf(BETA2_F, idx+1));
		const float nHat = n[idx]/(1-powf(BETA3_F, idx+1));
		const float update = mHat/(sqrtf(vHat)+EPSILON_F)+nHat*velocity[idx];
		params[idx] = params[idx]-update;
	}
}
extern "C" void AdanFloat(float* params, const float* grads, float* m, float* v, float* n, float* velocity, const float learningRate, const float weightDecay, const int size){
	auto gridSize = div_ceil(size, BS);
	AdanKernelFloat<<<gridSize, BS>>>(params, grads, m, v, n, velocity, learningRate, weightDecay, size);
}
__global__ void AdanKernelHalf(__half* __restrict__ params, const __half* __restrict__ grads, __half* __restrict__ m, __half* __restrict__ v, __half* __restrict__ n, __half* __restrict__ velocity, const float lr, const int t, const float wd, const int size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		float grad = __half2float(grads[idx]);
		const float param = __half2float(params[idx]);
		grad += param*wd;
		const float mF = BETA1_F*__half2float(m[idx])+(1.0f-BETA1_F)*grad;
		const float vF = BETA2_F*__half2float(v[idx])+(1.0f-BETA2_F)*grad*grad;
		const float nF = BETA3_F*__half2float(n[idx])+(1.0f-BETA3_F)*grad*grad*grad;
		const float velocityF = BETA1_F*__half2float(velocity[idx])+lr*grad;
		const float mHat = mF/(1.0f-powf(BETA1_F, t));
		const float vHat = vF/(1.0f-powf(BETA2_F, t));
		const float nHat = nF/(1.0f-powf(BETA3_F, t));
		const float update = mHat/(sqrtf(vHat)+EPSILON_F)+nHat*velocityF;
		params[idx] = __float2half(param-update);
		m[idx] = __float2half(mF);
		v[idx] = __float2half(vF);
		n[idx] = __float2half(nF);
		velocity[idx] = __float2half(velocityF);
	}
}
extern "C" void AdanHalf(__half* params, const __half* grads, __half* m, __half* v, __half* n, __half* velocity, const float learningRate, const int t, const float weightDecay, const int size){
	auto gridSize = div_ceil(size, BS);
	AdanKernelHalf<<<gridSize, BS>>>(params, grads, m, v, n, velocity, learningRate, t, weightDecay, size);
}
__global__ void gradientKernel(__half* grads, const __half* predictions, const __half* targets, const float clip, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		grads[idx] = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx]))));
	}
}
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size){
	auto gridSize = div_ceil(size, BS);
	gradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size);
}
__global__ void SplitGradKernel(__half* grads, const __half* predictions, const __half* targets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx/numCtrls;
		const int ctrlId = idx%numCtrls;
		const auto diff = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx]))));
		if(ctrlId < numButs){
			const auto gradIdx = batchId*numButs + ctrlId;
			grads[gradIdx] = diff;
		} else{
			const auto gradIdx = numButs*batchSize + batchId*(numCtrls - numButs) + (ctrlId - numButs);
			grads[gradIdx] = diff;
		}
	}
}
extern "C" void SplitGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = div_ceil(size, BS);
	SplitGradKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size, numCtrls, numButs, batchSize);
}
__global__ void MergeOutputsKernel(__half* outData, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		const int batchId = idx/numCtrls;
		const int outputId = idx%numCtrls;
		if(outputId<numButs){
			outData[idx] = buttonData[batchId*numButs+outputId];
		} else{
			outData[idx] = axisData[batchId*(numCtrls - numButs)+(outputId-numButs)];
		}
	}
}
extern "C" void MergeOutputs(__half* outData, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	auto gridSize = div_ceil(size, BS);
	MergeOutputsKernel<<<gridSize, BS>>>(outData, buttonData, axisData, size, numCtrls, numButs);
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
	auto gridSize = div_ceil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
}
__global__ void DiscriminatorGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const int numCtrls, const int numButs, const float binaryScale, const float continuousScale, const float clip){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		const float pred = __half2float(predictions[idx]);
		const float target = __half2float(targets[idx]);
		float gradient;
		if(idx%numCtrls < numButs){
			gradient = pred - target;
			gradient *= binaryScale;
			gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, gradient)));
		} else{
			gradient = 2.0f * (pred - target);
			gradient *= continuousScale;
			gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, gradient)));
		}
	}
}
extern "C" void DiscriminatorGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const int numCtrls, const int numButs, const float binaryScale, const float continuousScale, const float clip){
	auto gridSize = div_ceil(size, BS);
	DiscriminatorGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, numCtrls, numButs, binaryScale, continuousScale, clip);
}
__global__ void GAILGradientKernel(__half* gradients, const __half* predictions, const __half* discOutput, const float* expertActions, const int size, const int numCtrls, const int numButs, const float lambda, const float entropyCoeff, const float butScale, const float axiScale, const float clip){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float pred = __half2float(predictions[idx]);
		const float expert = expertActions[idx];
		float imitationGrad, adversarialGrad, entropyGrad;
		float scale;
		if(idx%numCtrls < numButs){   
			imitationGrad = -expert / (pred + 1e-6f) + (1.0f - expert) / (1.0f - pred + 1e-6f);
			adversarialGrad = -1.0f;
			entropyGrad = -logf(pred + 1e-6f) + logf(1.0f - pred + 1e-6f);
			scale = butScale;
			const float combinedGrad = imitationGrad + lambda*adversarialGrad*__half2float(discOutput[idx]) + entropyCoeff*entropyGrad;
			gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, combinedGrad*scale)));
		} else{   
			imitationGrad = 2.0f * (pred - expert);
			adversarialGrad = -1.0f;
			entropyGrad = 0.0f;
			scale = axiScale;
			const float combinedGrad = imitationGrad + lambda*adversarialGrad*__half2float(discOutput[idx]) + entropyCoeff*entropyGrad;
			gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, combinedGrad*scale)));
		}
	}
}
extern "C" void GAILGradient(__half* gradients, const __half* predictions, const __half* discOutput, const float* expertActions, const int batchSize, const int numCtrls, const int numButs, const float lambda, const float entropyCoeff, const float butScale, const float axiScale, const float clip){
	const auto size = numCtrls*batchSize;
	auto gridSize = div_ceil(size, BS);
	GAILGradientKernel<<<gridSize, BS>>>(gradients, predictions, discOutput, expertActions, size, numCtrls, numButs, lambda, entropyCoeff, butScale, axiScale, clip);
}
__global__ void biasGradientsKernel(const __half* gradInput, __half* gradBias, int c, int batchSize){
	extern __shared__ float sharedGrad[];
	const int channelIdx = blockIdx.x*blockDim.x + threadIdx.x;
	if(channelIdx < c){
		float sum = 0.0f;
		for(int i = 0; i < batchSize; i++){ sum += __half2float(gradInput[i*c + channelIdx]); }
		sharedGrad[threadIdx.x] = sum;
	} else{ sharedGrad[threadIdx.x] = 0.0f; }
	__syncthreads();
	for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(threadIdx.x < stride){ sharedGrad[threadIdx.x] += sharedGrad[threadIdx.x + stride]; }
		__syncthreads();
	}
	if(threadIdx.x == 0){
		for(int i = 0; i < blockDim.x && blockIdx.x*blockDim.x + i < c; i++){ gradBias[blockIdx.x*blockDim.x + i] = __float2half(sharedGrad[i]); }
	}
}
extern "C" void BiasGradient(const __half* gradInput, __half* gradBias, const int c, const int batchSize, cudaStream_t cudaStream){
	auto gridSize = div_ceil(c, BS);
	biasGradientsKernel<<<gridSize, BS, BS*sizeof(float), cudaStream>>>(gradInput, gradBias, c, batchSize);
}
__global__ void LeakyReluKernel(__half* __restrict__ data, const int size, const __half negativeSlope){
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < size; i += blockDim.x*gridDim.x) if(data[i] < __half(0.0f)) data[i] *= negativeSlope;
}
extern "C" void LeakyReluForward(__half* data, const int size, const float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	LeakyReluKernel<<<min(gridSize, GS), BS>>>(data, size, negativeSlope);
}
__global__ void LeakyReluBackwardKernel(__half* __restrict__ gradient, const __half* __restrict__ inData, const int size, const __half negativeSlope){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		gradient[idx] *= inData[idx] < __half(0.0f) ? negativeSlope : __half(1.0f);
	}
}
extern "C" void LeakyReluBackward(__half* grad, const __half* data, const int size, const float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	LeakyReluBackwardKernel<<<min(gridSize, GS), BS>>>(grad, data, size, __float2half(negativeSlope));
}
__global__ void SwishKernel(const __half* __restrict__ inData, __half* __restrict__ outData, const int size){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		const auto x = __half2float(inData[idx]);
		outData[idx] = __float2half(x/(1.0f + exp(-x)));
	}
}
extern "C" void SwishForward(const __half* inData, __half* outData, const int size){
	auto gridSize = div_ceil(size, BS);
	SwishKernel<<<min(gridSize, GS), BS>>>(inData, outData, size);
}
__global__ void SwishBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ data, const int size){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		const auto swish = __half2float(data[idx]);
		const auto sig = swish/(1.0f + swish);
		grad[idx] = __float2half(__half2float(grad[idx])*(swish + sig*(1.0f - swish)));
	}
}
extern "C" void SwishBackward(__half* grad, const __half* data, const int size){
	auto gridSize = div_ceil(size, BS);
	SwishBackwardKernel<<<min(gridSize, GS), BS>>>(grad, data, size);
}
__global__ void SigmoidForwardKernel(__half* data, const int numCtrls, const int numButs, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(data[idx]);
	data[idx] = __float2half(1.0f/(1.0f + expf(-val)));
}
extern "C" void SigmoidForward(__half* data, const int numCtrls, const int numButs, const int size, cudaStream_t cudaStream){
	auto gridSize = div_ceil(size, BS);
	SigmoidForwardKernel<<<gridSize, BS, 0, cudaStream>>>(data, numCtrls, numButs, size);
}
__global__ void SigmoidBackwardKernel(__half* grad, const __half* data, const int numCtrls, const int numButs, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(data[idx]);
	grad[idx] = __float2half(__half2float(grad[idx])*val*(1.0f - val));
}
extern "C" void SigmoidBackward(__half* grad, const __half* data, const int numCtrls, const int numButs, const int size, cudaStream_t cudaStream){
	auto gridSize = div_ceil(size, BS);
	SigmoidBackwardKernel<<<gridSize, BS, 0, cudaStream>>>(grad, data, numCtrls, numButs, size);
}
__global__ void ComputeMeanVarianceKernel(const __half* data, float* mean, float* variance, const int N, const int C, const int HW){
	extern __shared__ float sdata[];
	float* sMean = sdata, *sM2 = sMean + blockDim.x, *sCount = sM2 + blockDim.x;
	const int tid = threadIdx.x, cid = blockIdx.x;
	if(cid>=C) return;
	float tMean = 0.0f, tM2 = 0.0f, count = 0.0f;
	for(int n = 0; n<N; ++n){
		for(int i = tid; i<HW; i += blockDim.x){
			const int idx = n*C*HW+cid*HW+i;
			const float val = __half2float(data[idx]);
			++count;
			const float delta = val - tMean;
			tMean += delta / count;
			tM2 += delta * (val - tMean);
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

__global__ void layerNormForwardKernel(__half* output, const __half* data, const float* gamma, const float* beta, const float* mean, const float* variance, int N, int C, int HW){
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

extern "C" void LayerNormForward(__half* output, const __half* data, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW){
	int gridSize = div_ceil(C, BS), sharedMemSize = 3*BS*sizeof(float);
	ComputeMeanVarianceKernel<<<gridSize, BS, sharedMemSize>>>(data, mean, variance, N, C, HW);
	gridSize = div_ceil(N*C*HW, BS); sharedMemSize = 4*C*sizeof(float);
	layerNormForwardKernel<<<gridSize, BS, sharedMemSize>>>(output, data, gamma, beta, mean, variance, N, C, HW);
	const cudaError_t err = cudaGetLastError();
	if(err!=cudaSuccess){ printf("Kernel execution failed: %s\n", cudaGetErrorString(err)); }
}

__global__ void layerNormBackwardKernel(__half* grad, const __half* data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW){
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

extern "C" void LayerNormBackward(__half* grad, const __half* data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW){
	cudaMemset(gradGamma, 0, C*sizeof(float));
	cudaMemset(gradBeta, 0, C*sizeof(float));
	int sharedMemSize = 2*BS*sizeof(float);
	layerNormBackwardKernel<<<C, BS, sharedMemSize>>>(grad, data, gamma, gradGamma, gradBeta, mean, variance, N, C, HW);
	const cudaError_t err = cudaGetLastError();
	if(err!=cudaSuccess){ printf("Kernel execution failed: %s\n", cudaGetErrorString(err)); }
}
__device__ int deviceResult;
__global__ void isNaNKernel(const __half* data, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){
		atomicExch(&deviceResult, 1);
	}
}
extern "C" bool IsnanHalf(const __half* data, int size){
	int hResult = 0;
	cudaMemcpyToSymbol(deviceResult, &hResult, sizeof(int));
	auto gridSize = div_ceil(size, BS);
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
	auto gridSize = div_ceil(size, BS);
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
	auto gridSize = div_ceil(size, BS);
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
	auto gridSize = div_ceil(size, BS);
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
	auto gridSize = div_ceil(size, BS);
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
	auto gridSize = div_ceil(N*C, 1);
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
	int gridSize = div_ceil(N*C, 1);
	SpatialSoftmaxBackwardHalfKernel<<<gridSize, BS, 1032>>>(outData, gradIn, gradOut, N, C, H, W);
}