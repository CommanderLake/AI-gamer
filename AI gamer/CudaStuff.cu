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
	int sharedMemSize = 2*BS*sizeof(float);
	mseLoss2Kernel<<<gridSize, BS, sharedMemSize>>>(dPredictions, dTargets, size, numButs, numCtrls);
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
__global__ void sgdHalfKernel(__half* param, const float learningRate, const __half* gradParam, const int n){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < n) param[idx] = __float2half(__half2float(param[idx]) - learningRate*__half2float(gradParam[idx]));
}
extern "C" void SGDHalf(__half* param, const float learningRate, const __half* gradParam, const int size){
	auto gridSize = div_ceil(size, BS);
	sgdHalfKernel<<<gridSize, BS>>>(param, learningRate, gradParam, size);
}
__global__ void sgdFloatKernel(float* param, const float learningRate, const float* gradParam, const int n){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < n){ param[idx] -= learningRate*gradParam[idx]; }
}
extern "C" void SGDFloat(float* param, const float learningRate, const float* gradParam, const int size){
	auto gridSize = div_ceil(size, BS);
	sgdFloatKernel<<<gridSize, BS>>>(param, learningRate, gradParam, size);
}
#define BETA1_F 0.9f
#define BETA2_F 0.999f
#define EPSILON_F 1e-6f
#define CLIP 128.0f
__global__ void AdamwKernelFloat(float* params, const float* grads, float* m, float* v, const float lr, const int t, const float wd, const int n){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<n){
		const float grad = fmaxf(fminf(grads[idx], CLIP), -CLIP);
		m[idx] = BETA1_F*m[idx]+(1.0f-BETA1_F)*grad;
		v[idx] = BETA2_F*v[idx]+(1.0f-BETA2_F)*grad*grad;
		params[idx] *= 1.0f - wd;
		params[idx] -= lr*static_cast<float>(m[idx]/(1.0 - pow(BETA1_F, t))/(sqrt(v[idx]/(1.0 - pow(BETA2_F, t))) + EPSILON_F));
	}
}
extern "C" void AdamWFloat(float* params, const float* grads, float* m, float* v, const float learningRate, const int t, const float weightDecay, const int size){
	auto gridSize = div_ceil(size, BS);
	AdamwKernelFloat<<<gridSize, BS>>>(params, grads, m, v, learningRate, t, weightDecay, size);
}
__global__ void AdamwKernelHalf(__half* params, const __half* grads, __half* m, __half* v, const float lr, const int t, const float wd, const int n){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<n){
		const auto grad = fmaxf(fminf(__half2float(grads[idx]), CLIP), -CLIP);
		const auto mF = BETA1_F*__half2float(m[idx])+(1.0f-BETA1_F)*grad;
		const auto vF = BETA2_F*__half2float(v[idx])+(1.0f-BETA2_F)*grad*grad;
		const auto param = __half2float(params[idx])*(1.0f - wd);
		params[idx] = __float2half(param - lr*static_cast<float>(mF/(1.0 - pow(BETA1_F, t))/(sqrt(vF/(1.0 - pow(BETA2_F, t))) + EPSILON_F)));
		m[idx] = __float2half(mF);
		v[idx] = __float2half(vF);
	}
}
extern "C" void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, const float lr, const int t, const float weightDecay, const int size){
	auto gridSize = div_ceil(size, BS);
	AdamwKernelHalf<<<gridSize, BS>>>(params, grads, m, v, lr, t, weightDecay, size);
}
__global__ void gradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const float scale, const float clip){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float diff = (__half2float(predictions[idx]) - __half2float(targets[idx]))*scale;
		const float clippedDiff = fminf(fmaxf(diff, -clip), clip);
		gradients[idx] = __float2half(fmaxf(fabs(clippedDiff), 1e-7f)*(clippedDiff >= 0.0f ? 1.0f : -1.0f));
	}
}
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale, const float clip){
	auto gridSize = div_ceil(size, BS);
	gradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale, clip);
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
		gradient *= scale;
		gradients[idx] = __float2half(gradient);
	}
}
extern "C" void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale){
	auto gridSize = div_ceil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
}
__global__ void GAILGradientKernel(__half* gradients, const __half* predictions, const __half* discOutput, const float* expertActions, const int size, const int numCtrls, const int numButs, const float lambda, const float entropyCoeff, const float butScale, const float axiScale, const float clip){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float pred = __half2float(predictions[idx]);
		const float expert = expertActions[idx];
		float imitationGrad, adversarialGrad, entropyGrad;
		float scale;
		if(idx%numCtrls < numButs){
			imitationGrad = (pred - expert)/(pred*(1.0f - pred) + 1e-6f);
			adversarialGrad = -1.0f/(pred + 1e-6f);
			entropyGrad = -logf(pred/(1.0f - pred + 1e-6f));
			scale = butScale;
		} else{
			imitationGrad = pred - expert;
			adversarialGrad = -pred;
			entropyGrad = pred*pred*-0.5f;
			scale = axiScale;
		}
		const float combinedGrad = imitationGrad + lambda*adversarialGrad*__half2float(discOutput[idx/numCtrls]) + entropyCoeff*entropyGrad;
		gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, combinedGrad))*scale);
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
	const int idxInBlock = threadIdx.x;
	if(channelIdx < c){
		float sum = 0.0f;
		for(int i = 0; i < batchSize; i++){ sum += __half2float(gradInput[i*c + channelIdx]); }
		sharedGrad[idxInBlock] = sum;
	} else{ sharedGrad[idxInBlock] = 0.0f; }
	__syncthreads();
	// Reduce sum in shared memory
	for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(idxInBlock < stride){ sharedGrad[idxInBlock] += sharedGrad[idxInBlock + stride]; }
		__syncthreads();
	}
	if(idxInBlock == 0){
		// Write the reduced sum to the global bias gradient
		for(int i = 0; i < blockDim.x && blockIdx.x*blockDim.x + i < c; i++){ gradBias[blockIdx.x*blockDim.x + i] = __float2half(sharedGrad[i]); }
	}
}
extern "C" void BiasGradient(const __half* gradInput, __half* gradBias, const int c, const int batchSize){
	auto gridSize = div_ceil(c, BS);
	size_t sharedMemSize = BS*sizeof(float);
	biasGradientsKernel<<<gridSize, BS, sharedMemSize>>>(gradInput, gradBias, c, batchSize);
}
__global__ void clipGradsKernel(__half* grad, const int size, const float lower, const float upper){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float gradValue = __half2float(grad[idx]);
		if(isnan(gradValue)){
			grad[idx] = __float2half(1e-6f);
		} else if(gradValue >= 0.0f && gradValue < lower){
			grad[idx] = __float2half(lower);
		} else if(gradValue <= -0.0f && gradValue > -lower){
			grad[idx] = __float2half(-lower);
		} else if(gradValue > upper){
			grad[idx] = __float2half(upper);
		} else if(gradValue < -upper){
			grad[idx] = __float2half(-upper);
		}
	}
}
extern "C" void ClipGrads(__half* grad, int size){
	auto gridSize = div_ceil(size, BS);
	clipGradsKernel<<<gridSize, BS>>>(grad, size, 1e-6f, 128.0f);
}
__global__ void scaleKernel(__half* data, int size, float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ data[idx] = data[idx]*__float2half(scale); }
}
extern "C" void Scale(__half* data, int size, float scale){
	auto gridSize = div_ceil(size, BS);
	scaleKernel<<<gridSize, BS>>>(data, size, scale);
}
__global__ void leakyReluKernel(__half* data, int size, __half negativeSlope){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size) return;
	if(data[idx] < __half(0.0f)) data[idx] *= negativeSlope;
}
extern "C" void LeakyRelu(__half* data, int size, float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	leakyReluKernel<<<gridSize, BS>>>(data, size, negativeSlope);
}
__global__ void leakyReluBackwardKernel(__half* gradient, const __half* inData, int size, __half negativeSlope){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ if(inData[idx] < __half(0.0f)){ gradient[idx] *= negativeSlope; } }
}
extern "C" void LeakyReluBackward(__half* gradient, const __half* inData, int size, float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	leakyReluBackwardKernel<<<gridSize, BS>>>(gradient, inData, size, __float2half(negativeSlope));
}
__global__ void sigmoidForwardKernel(__half* data, int numCtrls, int numButs, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(data[idx]);
	data[idx] = __float2half(1.0f/(1.0f + expf(-val)));
}
extern "C" void SigmoidForward(__half* data, int numCtrls, int numButs, int batchSize){
	int numBlocks = div_ceil(numCtrls*batchSize, BS);
	sigmoidForwardKernel<<<numBlocks, BS>>>(data, numCtrls, numButs, numCtrls*batchSize);
}
__global__ void sigmoidBackwardKernel(__half* grad, const __half* data, int numCtrls, int numButs, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(data[idx]);
	grad[idx] = __float2half(__half2float(grad[idx])*val*(1.0f - val));
}
extern "C" void SigmoidBackward(__half* grad, const __half* data, int numCtrls, int numButs, int batchSize){
	int numBlocks = div_ceil(numCtrls*batchSize, BS);
	sigmoidBackwardKernel<<<numBlocks, BS>>>(grad, data, numCtrls, numButs, numCtrls*batchSize);
}
__global__ void computeMeanVarianceKernel(const __half* data, float* mean, float* variance, int N, int C, int HW){
	extern __shared__ float sdata[];
	float* s_sum = sdata;
	float* s_sq_sum = &sdata[blockDim.x];
	const int tid = threadIdx.x;
	const int cid = blockIdx.x; // Changed from blockIdx.y to blockIdx.x
	if(cid >= C) return; // Guard against excessive blocks
	float thread_sum = 0.0f;
	float thread_sq_sum = 0.0f;
	for(int n = 0; n < N; ++n){
		for(int i = tid; i < HW; i += blockDim.x){
			const int idx = (n*C + cid)*HW + i;
			if(idx < N*C*HW){
				// Boundary check
				const float val = __half2float(data[idx]);
				thread_sum += val;
				thread_sq_sum += val*val;
			}
		}
	}
	s_sum[tid] = thread_sum;
	s_sq_sum[tid] = thread_sq_sum;
	__syncthreads();
	// Parallel reduction in shared memory
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
		if(tid < s){
			s_sum[tid] += s_sum[tid + s];
			s_sq_sum[tid] += s_sq_sum[tid + s];
		}
		__syncthreads();
	}
	if(tid == 0){
		const int total_elements = N*HW;
		mean[cid] = s_sum[0]/total_elements;
		variance[cid] = fmaxf(s_sq_sum[0]/total_elements - mean[cid]*mean[cid], 0.0f);
	}
}
__global__ void layerNormForwardKernel(__half* output, const __half* data, const float* gamma, const float* beta, const float* mean, const float* variance, int N, int C, int HW, float epsilon){
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const int cid = blockIdx.y;
	if(cid >= C) return; // Guard against excessive blocks
	if(tid < N*HW){
		const int idx = (cid*N + tid/HW)*HW + tid % HW;
		if(idx < N*C*HW){
			// Boundary check
			const float x = __half2float(data[idx]);
			const float norm = (x - mean[cid])/sqrtf(variance[cid] + epsilon);
			output[idx] = __float2half(norm*gamma[cid] + beta[cid]);
		}
	}
}
extern "C" void LayerNormForward(__half* output, const __half* data, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW, float epsilon){
	dim3 gridDim((C + TPG - 1)/TPG, 1, 1); // Ensure we cover all channels
	dim3 blockDim(TPG, 1, 1);
	int sharedMemSize = 2*TPG*sizeof(float);
	computeMeanVarianceKernel<<<gridDim, blockDim, sharedMemSize>>>(data, mean, variance, N, C, HW);
	// Check for kernel launch errors
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("computeMeanVarianceKernel launch failed: %s\n", cudaGetErrorString(err));
		return;
	}
	gridDim = dim3((N*HW + TPG - 1)/TPG, C, 1);
	layerNormForwardKernel<<<gridDim, blockDim>>>(output, data, gamma, beta, mean, variance, N, C, HW, epsilon);
	// Check for kernel launch errors
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("layerNormForwardKernel launch failed: %s\n", cudaGetErrorString(err));
		return;
	}
	err = cudaGetLastError();
	if(err != cudaSuccess){ printf("Kernel execution failed: %s\n", cudaGetErrorString(err)); }
}
__global__ void layerNormBackwardKernel(__half* gradIn, const __half* gradOut, const __half* data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW, const float epsilon){
	extern __shared__ float sdata[];
	float* sGradGamma = sdata;
	float* sGradBeta = &sdata[blockDim.x];
	const int tid = threadIdx.x;
	const int cid = blockIdx.y;
	float threadGradGamma = 0.0f;
	float threadGradBeta = 0.0f;
	const float invStd = rsqrtf(variance[cid] + epsilon);
	for(int n = 0; n < N; ++n){
		for(int i = tid; i < HW; i += blockDim.x){
			const int idx = (cid*N + n)*HW + i;
			const float x = __half2float(data[idx]);
			const float dy = __half2float(gradOut[idx]);
			const float xHat = (x - mean[cid])*invStd;
			threadGradGamma += xHat*dy;
			threadGradBeta += dy;
			if(gradIn != nullptr){
				const float dx = gamma[cid]*invStd*(dy - (xHat*threadGradGamma + threadGradBeta)/(N*HW));
				gradIn[idx] = __float2half(dx);
			}
		}
	}
	sGradGamma[tid] = threadGradGamma;
	sGradBeta[tid] = threadGradBeta;
	__syncthreads();
	// Parallel reduction in shared memory
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
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
}
extern "C" void LayerNormBackward(__half* gradIn, const __half* gradOut, const __half* data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW, const float epsilon){
	dim3 gridDim(1, C, 1);
	dim3 blockDim(TPG, 1, 1);
	int sharedMemSize = 2*TPG*sizeof(float);
	layerNormBackwardKernel<<<gridDim, blockDim, sharedMemSize>>>(gradIn, gradOut, data, gamma, gradGamma, gradBeta, mean, variance, N, C, HW, epsilon);
}

__device__ int deviceResult;
__global__ void isNaNKernel(__half* data, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){
		atomicExch(&deviceResult, 1);
	}
}
extern "C" bool IsnanHalf(__half* data, int size){
	int h_result = 0;
	cudaMemcpyToSymbol(deviceResult, &h_result, sizeof(int));
	auto gridSize = div_ceil(size, BS);
	isNaNKernel<<<gridSize, BS>>>(data, size);
	cudaMemcpyFromSymbol(&h_result, deviceResult, sizeof(int));
	return h_result != 0;
}