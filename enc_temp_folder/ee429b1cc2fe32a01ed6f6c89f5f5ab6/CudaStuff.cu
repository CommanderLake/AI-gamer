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
extern "C" void InitCUDA(){
	const CUresult cudaRes = cuInit(0);
	if(cudaRes != CUDA_SUCCESS){
		const char* pStr = nullptr;
		cuGetErrorString(cudaRes, &pStr);
		throw std::runtime_error("CUDA Init failed, error string:\n\n" + std::string(pStr));
	}
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
		const int srcIdx = i*4; // Each pixARGB has 4 bytes
		dst[i] = src[srcIdx + 2]; // R plane
		dst[i + n] = src[srcIdx + 1]; // G plane
		dst[i + 2*n] = src[srcIdx]; // B plane
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
	// Perform reduction within the block
	while(idx<size){
		float diff = __half2float(predictions[idx])-targets[idx];
		diff *= diff;
		if(idx%numCtrls<numKeys){ sumKeys += diff; } else{ sumMouse += diff; }
		idx += gridDim.x*blockDim.x;
	}
	sdata[tid] = sumKeys;
	sdata[tid+blockDim.x] = sumMouse;
	__syncthreads();
	// Reduce within the block
	for(int s = blockDim.x/2; s>0; s >>= 1){
		if(tid<s){
			sdata[tid] += sdata[tid+s];
			sdata[tid+blockDim.x] += sdata[tid+blockDim.x+s];
		}
		__syncthreads();
	}
	// Write result for this block to global memory
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
#define CLIP 128.0f
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
__global__ void gradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		gradients[idx] = predictions[idx] - targets[idx];
	}
}
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size){
	auto gridSize = div_ceil(size, BS);
	gradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size);
}
__global__ void BCEGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float y = __half2float(targets[idx]);
		constexpr float epsilon = 1e-7f;
		const float pClamped = fminf(fmaxf(__half2float(predictions[idx]), epsilon), 1.0f - epsilon);
		float gradient = 0.0f;
		if(y == 1.0f){
			gradient = (pClamped - 1.0f)/pClamped;
		} else{
			gradient = pClamped/(1.0f - pClamped);
		}
		gradients[idx] = __float2half(gradient*scale);
	}
}
extern "C" void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale){
	auto gridSize = div_ceil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
}
__global__ void DiscriminatorGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const int numCtrls, const int numButs, const float binaryScale, const float continuousScale, const float clip){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float pred = __half2float(predictions[idx]);
		const float target = __half2float(targets[idx]);
		float gradient;
		if(idx%numCtrls < numButs){
			// Binary output (BCE loss)
			constexpr float epsilon = 1e-7f;
			const float pClamped = fminf(fmaxf(pred, epsilon), 1.0f - epsilon);
			if(target > 0.5f){
				gradient = (pClamped - 1.0f)/pClamped;
			} else{
				gradient = pClamped/(1.0f - pClamped);
			}
			gradient *= binaryScale;
		} else{
			// Continuous output (MSE loss)
			gradient = 2.0f*(pred - target);
			gradient *= continuousScale;
		}
		gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, gradient)));
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
		float imitationGrad, entropyGrad;
		float scale;
		const float adversarialGrad = -1.0f/(__half2float(discOutput[idx])+1e-6f); // Consistent computation
		if(idx%numCtrls < numButs){ // binary output
			imitationGrad = -expert / (pred + 1e-6f) + (1.0f - expert) / (1.0f - pred + 1e-6f);
			entropyGrad = -logf(pred + 1e-6f) + logf(1.0f - pred + 1e-6f);
			scale = butScale;
		} else{ // continuous output
			imitationGrad = 2.0f * (pred - expert);
			entropyGrad = 0.0f;
			scale = axiScale;
		}
		const float combinedGrad = imitationGrad + lambda * adversarialGrad + entropyCoeff * entropyGrad;
		gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, combinedGrad * scale)));
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
	// Reduce sum in shared memory
	for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(threadIdx.x < stride){ sharedGrad[threadIdx.x] += sharedGrad[threadIdx.x + stride]; }
		__syncthreads();
	}
	if(threadIdx.x == 0){
		// Write the reduced sum to the global bias gradient
		for(int i = 0; i < blockDim.x && blockIdx.x*blockDim.x + i < c; i++){ gradBias[blockIdx.x*blockDim.x + i] = __float2half(sharedGrad[i]); }
	}
}
extern "C" void BiasGradient(const __half* gradInput, __half* gradBias, const int c, const int batchSize){
	auto gridSize = div_ceil(c, BS);
	biasGradientsKernel<<<gridSize, BS, BS*sizeof(float)>>>(gradInput, gradBias, c, batchSize);
}
__global__ void LeakyReluKernel(__half* __restrict__ data, const int size, const __half negativeSlope){
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < size; i += blockDim.x*gridDim.x) if(data[i] < __half(0.0f)) data[i] *= negativeSlope;
}
extern "C" void LeakyReluForward(__half* data, const int size, const float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	LeakyReluKernel<<<gridSize, BS>>>(data, size, negativeSlope);
}
__global__ void LeakyReluBackwardKernel(__half* __restrict__ gradient, const __half* __restrict__ inData, const int size, const __half negativeSlope){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		gradient[idx] *= inData[idx] < __half(0.0f) ? negativeSlope : __half(1.0f);
	}
}
extern "C" void LeakyReluBackward(__half* grad, const __half* data, const int size, const float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	LeakyReluBackwardKernel<<<gridSize, BS>>>(grad, data, size, __float2half(negativeSlope));
}
__global__ void SwishKernel(__half* __restrict__ data, const int size){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		const auto x = __half2float(data[idx]);
		data[idx] = __float2half(x/(1.0f + exp(-x)));
	}
}
extern "C" void SwishForward(__half* data, const int size){
	auto gridSize = div_ceil(size, BS);
	SwishKernel<<<gridSize, BS>>>(data, size);
}
__global__ void SwishBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ output, const int size){
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += blockDim.x*gridDim.x){
		const auto swish = __half2float(output[idx]);
		const auto sig = swish/(1.0f + swish);
		grad[idx] = __float2half(__half2float(grad[idx])*(swish + sig*(1.0f - swish)));
	}
}
extern "C" void SwishBackward(__half* grad, const __half* data, const int size){
	auto gridSize = div_ceil(size, BS);
	SwishBackwardKernel<<<gridSize, BS>>>(grad, data, size);
}
__global__ void SigmoidForwardKernel(__half* data, const int numCtrls, const int numButs, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(data[idx]);
	data[idx] = __float2half(1.0f/(1.0f + expf(-val)));
}
extern "C" void SigmoidForward(__half* data, const int numCtrls, const int numButs, const int size){
	auto gridSize = div_ceil(size, BS);
	SigmoidForwardKernel<<<gridSize, BS>>>(data, numCtrls, numButs, size);
}
__global__ void SigmoidBackwardKernel(__half* grad, const __half* data, const int numCtrls, const int numButs, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(data[idx]);
	grad[idx] = __float2half(__half2float(grad[idx])*val*(1.0f - val));
}
extern "C" void SigmoidBackward(__half* grad, const __half* data, const int numCtrls, const int numButs, const int size){
	auto gridSize = div_ceil(size, BS);
	SigmoidBackwardKernel<<<gridSize, BS>>>(grad, data, numCtrls, numButs, size);
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
		const int c = (i/HW)%C;
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
__global__ void ComputeAttentionKernel(const __half* queryMap, const __half* keyMap, __half* attentionScores, const int inC, const int attC, const int inH, const int inW, const int size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		const int cPrime = idx/(inH*inW);
		const int n = idx%(inH*inW)/(inH*inW);
		const int h = idx/inW%inH;
		const int w = idx%inW;
		__half score = __float2half(0.0f);
		for(int c = 0; c<inC; ++c){
			const int qIdx = ((n*inC+c)*inH+h)*inW+w;
			const int kIdx = ((n*attC+cPrime)*inH+h)*inW+w;
			score += queryMap[qIdx]*keyMap[kIdx];
		}
		score = score/__float2half(sqrt(static_cast<float>(inC)));
		attentionScores[idx] = score;
	}
}
extern "C" void ComputeAttention(const __half* inData, const __half* attentionMap, __half* outData, int inC, int attC, int inH, int inW, int size){
	auto gridSize = div_ceil(size, BS);
	ComputeAttentionKernel<<<gridSize, BS>>>(inData, attentionMap, outData, inC, attC, inH, inW, size);
}
__global__ void ApplyAttentionKernel(const __half* valueMap, const __half* attentionScores, __half* output, const int inC, const int attC, const int inH, const int inW, const int size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		const int n = idx/(inC*inH*inW);
		const int c = idx/(inH*inW)%inC;
		const int h = idx/inW%inH;
		const int w = idx%inW;
		__half result = __float2half(0.0f);
		// Accumulate attention-weighted value map
		for(int ac = 0; ac<attC; ++ac){
			const int attIdx = ((n*attC+ac)*inH+h)*inW+w;
			const int valIdx = ((n*inC+c)*inH+h)*inW+w;
			result = __hfma(valueMap[valIdx], attentionScores[attIdx], result);
		}
		// Store the result in the output tensor
		output[idx] = result;
	}
}
extern "C" void ApplyAttention(const __half* valueMap, const __half* attentionScores, __half* output, int inC, int attC, int inH, int inW, int size){
	auto gridSize = div_ceil(size, BS);
	ApplyAttentionKernel<<<gridSize, BS>>>(valueMap, attentionScores, output, inC, attC, inH, inW, size);
}
__global__ void ApplyAttentionBackwardKernel(const __half* grad, const __half* valueMap, const __half* attentionScores, __half* gradValue, __half* gradAttention, const int inC, const int attC, const int inH, const int inW, const int size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		const int n = idx/(inC*inH*inW);
		const int h = idx/inW%inH;
		const int w = idx%inW;
		__half gradValueSum = __float2half(0.0f);
		for(int ac = 0; ac<attC; ++ac){
			const int attIdx = ((n*attC+ac)*inH+h)*inW+w;
			gradValueSum = __hfma(grad[idx], attentionScores[attIdx], gradValueSum);
		}
		gradValue[idx] = gradValueSum;
		for(int ac = 0; ac<attC; ++ac){
			const int attIdx = ((n*attC+ac)*inH+h)*inW+w;
			atomicAdd(&gradAttention[attIdx], grad[idx]*valueMap[idx]);
		}
	}
}
extern "C" void ApplyAttentionBackward(const __half* grad, const __half* valueMap, const __half* attentionScores, __half* gradValue, __half* gradAttention, const int inC, const int attC, const int inH, const int inW, const int size){
	auto gridSize = div_ceil(size, BS);
	ApplyAttentionBackwardKernel<<<gridSize, BS, BS*sizeof(__half)>>>(grad, valueMap, attentionScores, gradValue, gradAttention, inC, attC, inH, inW, size);
}
__global__ void ComputeQueryKeyGradKernel(const __half* gradAttention, const __half* queryMap, const __half* keyMap, __half* gradQuery, __half* gradKey, const int inC, const int attC, const int inH, const int inW, const int size){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<size){
		const int n = idx/(inC*inH*inW);
		const int h = idx/inW%inH;
		const int w = idx%inW;
		__half gradQuerySum = __float2half(0.0f);
		__half gradKeySum = __float2half(0.0f);
		for(int ac = 0; ac<attC; ++ac){
			const int attIdx = ((n*attC+ac)*inH+h)*inW+w;
			const __half gradAtt = gradAttention[attIdx];
			gradQuerySum = __hfma(gradAtt, keyMap[idx], gradQuerySum);
			gradKeySum = __hfma(gradAtt, queryMap[idx], gradKeySum);
		}
		const float scale = 1.0f/sqrt(static_cast<float>(inC));
		gradQuery[idx] = gradQuerySum*__float2half(scale);
		gradKey[idx] = gradKeySum*__float2half(scale);
	}
}
extern "C" void ComputeQueryKeyGrad(const __half* gradAttention, const __half* queryMap, const __half* keyMap, __half* gradQuery, __half* gradKey, int inC, int attC, int inH, int inW, int size){
	auto gridSize = div_ceil(size, BS);
	ComputeQueryKeyGradKernel<<<gridSize, BS>>>(gradAttention, queryMap, keyMap, gradQuery, gradKey, inC, attC, inH, inW, size);
}
#include <mma.h>
__global__ void SpatialAttentionForwardKernel(const __half* __restrict__ inData, const __half* __restrict__ keyWeights, const __half* __restrict__ queryWeights, const __half* __restrict__ valueWeights, __half* __restrict__ keyMap,
											__half* __restrict__ queryMap, __half* __restrict__ valueMap, __half* __restrict__ attentionScores, __half* __restrict__ outData, int batchSize, int inC, int attC, int inH, int inW, const float scale,
											const float residualScale){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int n = idx/(inH*inW);
	const int h = idx/inW%inH;
	const int w = idx%inW;
	if(n<batchSize&&h<inH&&w<inW){
		const int spatialIdx = n*inH*inW+h*inW+w;
		const bool useWMMA = h%16==0&&w%16==0&&h+15<inH&&w+15<inW&&attC%16==0;
		if(useWMMA){
			// Use Tensor Cores for main computation
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> aFrag;
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> bFragKey, bFragQuery, bFragValue;
			nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> cFrag;
			// Load input data
			load_matrix_sync(aFrag, inData+spatialIdx*inC, inW);
			// Key projection
			load_matrix_sync(bFragKey, keyWeights, inC);
			fill_fragment(cFrag, 0.0f);
			mma_sync(cFrag, aFrag, bFragKey, cFrag);
			store_matrix_sync(keyMap+spatialIdx*attC, cFrag, inW, nvcuda::wmma::mem_row_major);
			// Query projection
			load_matrix_sync(bFragQuery, queryWeights, inC);
			fill_fragment(cFrag, 0.0f);
			mma_sync(cFrag, aFrag, bFragQuery, cFrag);
			store_matrix_sync(queryMap+spatialIdx*attC, cFrag, inW, nvcuda::wmma::mem_row_major);
			// Value projection
			load_matrix_sync(bFragValue, valueWeights, inC);
			fill_fragment(cFrag, 0.0f);
			mma_sync(cFrag, aFrag, bFragValue, cFrag);
			store_matrix_sync(valueMap+spatialIdx*attC, cFrag, inW, nvcuda::wmma::mem_row_major);
		} else{
			// Use standard CUDA operations for remainder
			for(int ac = 0; ac<attC; ++ac){
				__half keySum = __float2half(0.0f);
				__half querySum = __float2half(0.0f);
				__half valueSum = __float2half(0.0f);
				for(int ic = 0; ic<inC; ++ic){
					const __half inVal = inData[spatialIdx*inC+ic];
					keySum = __hfma(inVal, keyWeights[ac*inC+ic], keySum);
					querySum = __hfma(inVal, queryWeights[ac*inC+ic], querySum);
					valueSum = __hfma(inVal, valueWeights[ac*inC+ic], valueSum);
				}
				keyMap[spatialIdx*attC+ac] = keySum;
				queryMap[spatialIdx*attC+ac] = querySum;
				valueMap[spatialIdx*attC+ac] = valueSum;
			}
		}
		// Compute Attention Scores (dot product of key and query)
		float score = 0.0f;
		for(int ac = 0; ac<attC; ++ac){ score += __half2float(queryMap[spatialIdx*attC+ac])*__half2float(keyMap[spatialIdx*attC+ac]); }
		score *= scale;
		// Softmax: Normalize Attention Scores
		extern __shared__ float sharedScores[];
		float maxScore = score;
		for(int i = 0; i<blockDim.x; i++){ maxScore = max(maxScore, __shfl_sync(0xffffffff, score, i)); }
		const float expScore = expf(score-maxScore);
		sharedScores[threadIdx.x] = expScore;
		__syncthreads();
		for(int stride = blockDim.x/2; stride>0; stride >>= 1){
			if(threadIdx.x<stride){ sharedScores[threadIdx.x] += sharedScores[threadIdx.x+stride]; }
			__syncthreads();
		}
		const float sumScores = sharedScores[0];
		score = sumScores>0 ? expScore/sumScores : 0.0f;
		attentionScores[spatialIdx] = __float2half(score);
		// Apply Attention to Value Map and add Residual Connection
		for(int ac = 0; ac<attC; ++ac){
			const __half weightedValue = __hmul(attentionScores[spatialIdx], valueMap[spatialIdx*attC+ac]);
			outData[spatialIdx*attC+ac] = __hfma(__float2half(residualScale), inData[spatialIdx*inC+ac%inC], weightedValue);
		}
	}
}
extern "C" void SpatialAttentionForward(const __half* __restrict__ inData, const __half* __restrict__ keyWeights, const __half* __restrict__ queryWeights, const __half* __restrict__ valueWeights, __half* __restrict__ keyMap,
														__half* __restrict__ queryMap, __half* __restrict__ valueMap, __half* __restrict__ attentionScores, __half* __restrict__ outData, const int batchSize, const int inC, const int attC, const int inH, const int inW){
	const int totalSize = batchSize*inH*inW*attC;
	const int gridSize = div_ceil(totalSize, BS);
	const float scale = 1.0f/sqrtf(static_cast<float>(inC));
	// Allocate shared memory for softmax
	size_t sharedMemorySize = BS*sizeof(float);
	// Launch the kernel
	SpatialAttentionForwardKernel<<<gridSize, BS, sharedMemorySize>>>(inData, keyWeights, queryWeights, valueWeights, keyMap, queryMap, valueMap, attentionScores, outData, batchSize, inC, attC, inH, inW, scale, 1.0f);
	const cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("SpatialAttentionForwardKernel launch failed: %s\n", cudaGetErrorString(err));
	}
}
__global__ void SpatialAttentionBackwardKernel(const __half* __restrict__ inData, const __half* __restrict__ keyWeights, const __half* __restrict__ queryWeights, const __half* __restrict__ valueWeights, const __half* __restrict__ keyMap,
												const __half* __restrict__ queryMap, const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, const __half* __restrict__ inGrad, __half* __restrict__ outGrad,
												__half* __restrict__ keyWeightsGrad, __half* __restrict__ queryWeightsGrad, __half* __restrict__ valueWeightsGrad, int batchSize, int inC, int attC, int inH, int inW, const float scale,
												const float residualScale){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int n = idx/(inH*inW);
	const int h = idx/inW%inH;
	const int w = idx%inW;
	if(n<batchSize&&h<inH&&w<inW){
		const int spatialIdx = n*inH*inW+h*inW+w;
		const bool useWMMA = h%16==0&&w%16==0&&h+15<inH&&w+15<inW&&attC%16==0;
		// Compute gradients for attention mechanism
		float attScoreGrad[16] = {0};
		float valueMapGrad[16] = {0};
		float gradSum = 0.0f;
		for(int ac = 0; ac<attC; ++ac){
			const float inGradVal = __half2float(inGrad[spatialIdx*attC+ac]);
			const float valueMapVal = __half2float(valueMap[spatialIdx*attC+ac]);
			const float attScoreVal = __half2float(attentionScores[spatialIdx]);
			attScoreGrad[ac%16] = inGradVal*valueMapVal;
			valueMapGrad[ac%16] = inGradVal*attScoreVal;
			gradSum += attScoreGrad[ac%16];
		}
		// Compute softmax gradient
		float maxGrad = -FLT_MAX;
		for(int ac = 0; ac<attC; ++ac){
			const float grad = attScoreGrad[ac%16]-gradSum*__half2float(attentionScores[spatialIdx]);
			attScoreGrad[ac%16] = grad*scale;
			maxGrad = max(maxGrad, fabsf(attScoreGrad[ac%16]));
		}
		// Gradient clipping and normalization
		const float clipValue = 1.0f;
		if(maxGrad>clipValue){
			const float normFactor = clipValue/maxGrad;
			for(int ac = 0; ac<attC; ++ac){
				attScoreGrad[ac%16] *= normFactor;
				valueMapGrad[ac%16] *= normFactor;
			}
		}
		if(useWMMA){
			// Use Tensor Cores for main computation
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::col_major> aFragKey, aFragQuery, aFragValue;
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> bFrag;
			nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> cFrag;
			// Load gradients
			for(int i = 0; i<bFrag.num_elements; i++){ bFrag.x[i] = __float2half(attScoreGrad[i]); }
			// Key weights gradient
			load_matrix_sync(aFragKey, keyMap+spatialIdx*attC, inW);
			fill_fragment(cFrag, 0.0f);
			mma_sync(cFrag, aFragKey, bFrag, cFrag);
			store_matrix_sync(keyWeightsGrad+spatialIdx*inC, cFrag, inC, nvcuda::wmma::mem_row_major);
			// Query weights gradient
			load_matrix_sync(aFragQuery, queryMap+spatialIdx*attC, inW);
			fill_fragment(cFrag, 0.0f);
			mma_sync(cFrag, aFragQuery, bFrag, cFrag);
			store_matrix_sync(queryWeightsGrad+spatialIdx*inC, cFrag, inC, nvcuda::wmma::mem_row_major);
			// Value weights gradient
			load_matrix_sync(aFragValue, valueMap+spatialIdx*attC, inW);
			for(int i = 0; i<bFrag.num_elements; i++){ bFrag.x[i] = __float2half(valueMapGrad[i]); }
			fill_fragment(cFrag, 0.0f);
			mma_sync(cFrag, aFragValue, bFrag, cFrag);
			store_matrix_sync(valueWeightsGrad+spatialIdx*inC, cFrag, inC, nvcuda::wmma::mem_row_major);
			// Input gradient
			load_matrix_sync(bFrag, inData+spatialIdx*inC, inW);
			fill_fragment(cFrag, 0.0f);
			mma_sync(cFrag, aFragKey, bFrag, cFrag);
			mma_sync(cFrag, aFragQuery, bFrag, cFrag);
			mma_sync(cFrag, aFragValue, bFrag, cFrag);
			store_matrix_sync(outGrad+spatialIdx*inC, cFrag, inC, nvcuda::wmma::mem_row_major);
		} else{
			// Use standard CUDA operations for remainder
			for(int ic = 0; ic<inC; ++ic){
				float keyGrad = 0.0f;
				float queryGrad = 0.0f;
				float valueGrad = 0.0f;
				float inGradVal = 0.0f;
				for(int ac = 0; ac<attC; ++ac){
					keyGrad += attScoreGrad[ac%16]*__half2float(keyWeights[ac*inC+ic]);
					queryGrad += attScoreGrad[ac%16]*__half2float(queryWeights[ac*inC+ic]);
					valueGrad += valueMapGrad[ac%16]*__half2float(valueWeights[ac*inC+ic]);
					inGradVal += attScoreGrad[ac%16]*__half2float(keyMap[spatialIdx*attC+ac]);
					inGradVal += attScoreGrad[ac%16]*__half2float(queryMap[spatialIdx*attC+ac]);
					inGradVal += valueMapGrad[ac%16]*__half2float(valueMap[spatialIdx*attC+ac]);
				}
				keyWeightsGrad[ic*attC+idx%attC] = __float2half(keyGrad);
				queryWeightsGrad[ic*attC+idx%attC] = __float2half(queryGrad);
				valueWeightsGrad[ic*attC+idx%attC] = __float2half(valueGrad);
				outGrad[spatialIdx*inC+ic] = __float2half(residualScale*__half2float(inGrad[spatialIdx*attC+ic%attC])+inGradVal);
			}
		}
	}
}
extern "C" void SpatialAttentionBackward(const __half* __restrict__ inData, const __half* __restrict__ keyWeights, const __half* __restrict__ queryWeights, const __half* __restrict__ valueWeights, const __half* __restrict__ keyMap,
										const __half* __restrict__ queryMap, const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, const __half* __restrict__ inGrad, __half* __restrict__ outGrad,
										__half* __restrict__ keyWeightsGrad, __half* __restrict__ queryWeightsGrad, __half* __restrict__ valueWeightsGrad, const int batchSize, const int inC, const int attC, const int inH, const int inW){
	const int totalSize = batchSize*attC*inH*inW;
	const int gridSize = div_ceil(totalSize, BS);
	const float scale = 1.0f/sqrtf(static_cast<float>(inC));
	// Launch the kernel
	SpatialAttentionBackwardKernel<<<gridSize, BS>>>(inData, keyWeights, queryWeights, valueWeights, keyMap, queryMap, valueMap, attentionScores, inGrad, outGrad, keyWeightsGrad, queryWeightsGrad, valueWeightsGrad, batchSize, inC, attC, inH, inW, scale, 1.0f);
	const cudaError_t err = cudaGetLastError();
	if(err!=cudaSuccess){ printf("SpatialAttentionBackwardKernel launch failed: %s\n", cudaGetErrorString(err)); }
}