#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "common.h"
#include <cuda.h>
#include <curand.h>
__device__ __host__ int div_ceil(int a, int b){ return (a % b != 0) ? (a/b + 1) : (a/b); }
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
		diff *= diff; // squared error
	}
	sdata[tid] = diff;
	__syncwarp();
	for(int i = 16; i > 0; i >>= 1){
		if(tid < i){ sdata[tid] += sdata[tid + i]; }
		__syncwarp();
	}
	if(tid == 0){ d_loss = sdata[0]; }
}
extern "C" float MseLoss(const __half* d_predictions, const float* d_targets, int size){
	auto gridSize = div_ceil(size, BS);
	mseLossKernel<<<gridSize, BS, BS*sizeof(float)>>>(d_predictions, d_targets, size);
	float h_loss;
	cudaMemcpyFromSymbol(&h_loss, d_loss, sizeof(float));
	return h_loss/size;
}
__global__ void convertAndNormalizeKernel(__half* output, unsigned char* input, size_t size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ output[idx] = __float2half(static_cast<float>(input[idx])/255.0f); }
}
extern "C" void ConvertAndNormalize(__half* output, unsigned char* input, size_t size){
	auto gridSize = div_ceil(size, BS);
	convertAndNormalizeKernel<<<gridSize, BS>>>(output, input, size);
}
__global__ void UnConvertAndUnNormalizeKernel(unsigned char* output, const __half* input, size_t size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ output[idx] = static_cast<unsigned char>(__half2float(input[idx])*255.0f); }
}
extern "C" void UnConvertAndUnNormalize(unsigned char* output, const __half* input, size_t size){
	auto gridSize = div_ceil(size, BS);
	UnConvertAndUnNormalizeKernel<<<gridSize, BS>>>(output, input, size);
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
__global__ void HeInitKernel(__half* halfWeights, float* weights, int n, float scale){
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
	if(idx < n){ param[idx] -= __float2half(learningRate*__half2float(gradParam[idx])); }
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
__device__ const float beta1F = 0.9f;
__device__ const float beta2F = 0.999f;
__device__ const float epsilonF = 1e-8f;
__global__ void adamKernelHalf(__half* param, float* m, float* v, const float learningRate, const __half* gradParam, const int n, const int t){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < n){
		const float grad = __half2float(gradParam[idx]);
		float param_value = __half2float(param[idx]);
		m[idx] = beta1F*m[idx] + (1.0f - beta1F)*grad;
		v[idx] = beta2F*v[idx] + (1.0f - beta2F)*grad*grad;
		const float m_hat = m[idx]/(1.0f - powf(beta1F, t));
		const float v_hat = v[idx]/(1.0f - powf(beta2F, t));
		param_value -= learningRate*m_hat/(sqrtf(v_hat) + epsilonF);
		param[idx] = __float2half(param_value);
	}
}
void AdamHalf(__half* param, float* m, float* v, const float learningRate, const __half* gradParam, const int size, const int t){
	auto gridSize = div_ceil(size, BS);
	adamKernelHalf<<<gridSize, BS>>>(param, m, v, learningRate, gradParam, size, t);
}
__global__ void adamKernelFloat(float* param, float* m, float* v, const float learningRate, const float* gradParam, const int n, const int t){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < n){
		const float grad = gradParam[idx];
		float param_value = param[idx];
		m[idx] = beta1F*m[idx] + (1.0f - beta1F)*grad;
		v[idx] = beta2F*v[idx] + (1.0f - beta2F)*grad*grad;
		const float m_hat = m[idx]/(1.0f - powf(beta1F, t));
		const float v_hat = v[idx]/(1.0f - powf(beta2F, t));
		param_value -= learningRate*m_hat/(sqrtf(v_hat) + epsilonF);
		param[idx] = param_value;
	}
}
void AdamFloat(float* param, float* m, float* v, const float learningRate, const float* gradParam, const int size, const int t){
	auto gridSize = div_ceil(size, BS);
	adamKernelFloat<<<gridSize, BS>>>(param, m, v, learningRate, gradParam, size, t);
}
__global__ void adamWKernelHalf(__half* param, float* m, float* v, const float learningRate, const __half* gradParam, const int n, const int t, const float weightDecay){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < n){
		const float grad = __half2float(gradParam[idx]);
		float param_value = __half2float(param[idx]);
		m[idx] = beta1F*m[idx] + (1.0f - beta1F)*grad;
		v[idx] = beta2F*v[idx] + (1.0f - beta2F)*grad*grad;
		const float m_hat = m[idx]/(1.0f - powf(beta1F, t));
		const float v_hat = v[idx]/(1.0f - powf(beta2F, t));
		param_value -= learningRate*(m_hat / (sqrtf(v_hat) + epsilonF));
		param_value -= learningRate*weightDecay*param_value;
		param[idx] = __float2half(param_value);
	}
}
void AdamWHalf(__half* param, float* m, float* v, const float learningRate, const __half* gradParam, const int size, const int t, const float weightDecay){
	auto gridSize = div_ceil(size, BS);
	adamWKernelHalf<<<gridSize, BS>>>(param, m, v, learningRate, gradParam, size, t, weightDecay);
}
__global__ void adamWKernelFloat(float* param, float* m, float* v, const float learningRate, const float* gradParam, const int n, const int t, const float weightDecay){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < n){
		const float grad = gradParam[idx];
		float param_value = param[idx];
		m[idx] = beta1F*m[idx] + (1.0f - beta1F)*grad;
		v[idx] = beta2F*v[idx] + (1.0f - beta2F)*grad*grad;
		const float m_hat = m[idx]/(1.0f - powf(beta1F, t));
		const float v_hat = v[idx]/(1.0f - powf(beta2F, t));
		param_value -= learningRate*(m_hat / (sqrtf(v_hat) + epsilonF));
		param_value -= learningRate*weightDecay*param_value;
		param[idx] = param_value;
	}
}
void AdamWFloat(float* param, float* m, float* v, const float learningRate, const float* gradParam, const int size, const int t, const float weightDecay){
	auto gridSize = div_ceil(size, BS);
	adamWKernelFloat<<<gridSize, BS>>>(param, m, v, learningRate, gradParam, size, t, weightDecay);
}
__global__ void warp8SmemKernel(int N, int K, const half*__restrict__ A, const half*__restrict__ B, half*__restrict__ C, int BS, int TPG, int CPB){
	extern __shared__ half A_smem[];
	const int A_smem_iters = div_ceil(K, BS);
#pragma unroll
	for(int i = 0; i < A_smem_iters; ++i){
		const int idx = i*BS + threadIdx.x;
		A_smem[idx] = A[idx];
	}
	__syncthreads();
	const int group_id = threadIdx.x/TPG;
	const int group_col = blockIdx.x*CPB + group_id;
	if(group_col >= N){ return; }
	const int K_iters = div_ceil(K, TPG);
	const int group_lane_id = threadIdx.x % TPG;
	float tmp = 0.0;
#pragma unroll
	for(int i = 0; i < K_iters; ++i){
		const int A_idx = i*TPG + group_lane_id;
		const int B_idx = i*TPG + group_lane_id + group_col*K;
		tmp += __half2float(A_smem[A_idx])*__half2float(B[B_idx]);
	}
	constexpr unsigned int mask = 0xffffffff;
#pragma unroll
	for(int i = TPG/2; i >= 1; i /= 2){ tmp += __shfl_xor_sync(mask, tmp, i); }
	if(group_lane_id == 0){ C[group_col] = __float2half(tmp); }
}
extern "C" void Hgemv(const half* A, const half* B, half* C, int N, int K){
	static int smem_max_size = K*sizeof(half);
	dim3 block(BS);
	dim3 grid(div_ceil(N, CPB));
	warp8SmemKernel<<<grid, block, smem_max_size>>>(N, K, A, B, C, BS, TPG, CPB);
}
__global__ void gemmHFFKernel(bool transA, bool transB, int M, int N, int K, const half*__restrict__ A, int lda, const float*__restrict__ B, int ldb, float*__restrict__ C, int ldc, int BS, int TPG, int RPB, int CPB){
	extern __shared__ half A_smem[];
	const int A_smem_iters = div_ceil(transA ? M : K, BS);
#pragma unroll
	for(int i = 0; i < A_smem_iters; ++i){
		const int idx = i*BS + threadIdx.x;
		if(idx < (transA ? M : K)){ A_smem[idx] = A[transA ? (idx*lda) : (idx)]; }
	}
	__syncthreads();
	const int group_id = threadIdx.x/TPG;
	const int group_row = blockIdx.y*RPB + group_id;
	const int group_col = blockIdx.x*CPB + group_id;
	if(group_row >= M || group_col >= N){ return; }
	const int K_iters = div_ceil(K, TPG);
	const int group_lane_id = threadIdx.x % TPG;
	float tmp = 0.0f;
#pragma unroll
	for(int i = 0; i < K_iters; ++i){
		const int k = i*TPG + group_lane_id;
		if(k < K){
			const int A_idx = transA ? (k*M + group_row) : (group_row*K + k);
			const int B_idx = transB ? (group_col*ldb + k) : (k*ldb + group_col);
			tmp += __half2float(A_smem[A_idx])*B[B_idx];
		}
	}
	constexpr unsigned int mask = 0xffffffff;
#pragma unroll
	for(int i = TPG/2; i >= 1; i /= 2){ tmp += __shfl_xor_sync(mask, tmp, i); }
	if(group_lane_id == 0){ C[group_row*ldc + group_col] = tmp; }
}
extern "C" void GemmHff(int M, int N, int K, bool transA, bool transB, const half* A, int lda, const float* B, int ldb, float* C, int ldc){
	int smem_size = (transA ? M : K)*sizeof(half);
	dim3 block(BS);
	dim3 grid(div_ceil(N, CPB), div_ceil(M, RPB));
	gemmHFFKernel<<<grid, block, smem_size>>>(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, BS, TPG, RPB, CPB);
}
__global__ void gradientKernel(__half* gradients, const __half* predictions, const float* targets, float delta, int size, float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float diff = __half2float(predictions[idx]) - targets[idx];
		float gradient;
		if(fabs(diff) <= delta){
			gradient = diff;
		} else{
			gradient = delta*((diff > 0) - (diff < 0));
		}
		gradients[idx] = __float2half(gradient*scale);
	}
}
extern "C" void Gradient(__half* d_gradient, const __half* d_predictions, const float* d_targets, int batchSize, int n, float scale){
	gradientKernel<<<n, batchSize>>>(d_gradient, d_predictions, d_targets, 1.0f, n*batchSize, scale);
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
extern "C" void BiasGradient(const __half* gradInput, __half* gradBias, int c, int batchSize){
	auto gridSize = div_ceil(c, BS);
	size_t sharedMemSize = BS*sizeof(float);
	biasGradientsKernel<<<gridSize, BS, sharedMemSize>>>(gradInput, gradBias, c, batchSize);
}
__global__ void clipGradsKernel(__half* grad, int size, float lower, float upper){
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
	if(idx < size){ if(data[idx] < __half(0)){ data[idx] *= negativeSlope; } }
}
extern "C" void LeakyRelu(__half* data, int size, float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	leakyReluKernel<<<gridSize, BS>>>(data, size, __float2half(negativeSlope));
}
__global__ void leakyReluBackwardKernel(__half* gradient, const __half* inData, int size, __half negativeSlope){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ if(inData[idx] < __half(0.0f)){ gradient[idx] *= negativeSlope; } }
}
extern "C" void LeakyReluBackward(__half* gradient, const __half* inData, int size, float negativeSlope){
	auto gridSize = div_ceil(size, BS);
	leakyReluBackwardKernel<<<gridSize, BS>>>(gradient, inData, size, __float2half(negativeSlope));
}
__global__ void sigmoidForwardKernel(__half* data, int numSigmoidOutputs, int batchSize, int outputSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < batchSize*outputSize){
		if(idx % outputSize < numSigmoidOutputs){
			const float val = __half2float(data[idx]);
			const float sigmoid = 1.0f/(1.0f + expf(-val));
			data[idx] = __float2half(sigmoid);
		}
	}
}
extern "C" void SigmoidForward(__half* data, int numSigmoidOutputs, int batchSize, int outputSize){
	int numBlocks = div_ceil(batchSize*outputSize, BS);
	sigmoidForwardKernel<<<numBlocks, BS>>>(data, numSigmoidOutputs, batchSize, outputSize);
}
__global__ void sigmoidBackwardKernel(__half* grad, const __half* data, int numSigmoidOutputs, int batchSize, int outputSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < batchSize*outputSize){
		if(idx % outputSize < numSigmoidOutputs){
			const float val = __half2float(data[idx]);
			grad[idx] = __float2half(__half2float(grad[idx])*val*(1.0f - val));
		}
	}
}
extern "C" void SigmoidBackward(__half* grad, const __half* data, int numSigmoidOutputs, int batchSize, int outputSize){
	int numBlocks = div_ceil(batchSize*outputSize, BS);
	sigmoidBackwardKernel<<<numBlocks, BS>>>(grad, data, numSigmoidOutputs, batchSize, outputSize);
}
__global__ void computeMeanVarianceKernel(const __half* data, float* mean, float* variance, int N, int C, int HW){
	extern __shared__ float sdata[];
	float* s_sum = sdata;
	float* s_sq_sum = &sdata[blockDim.x];
	int tid = threadIdx.x;
	int cid = blockIdx.x; // Changed from blockIdx.y to blockIdx.x
	if(cid >= C) return; // Guard against excessive blocks
	float thread_sum = 0.0f;
	float thread_sq_sum = 0.0f;
	for(int n = 0; n < N; ++n){
		for(int i = tid; i < HW; i += blockDim.x){
			int idx = (n*C + cid)*HW + i;
			if(idx < N*C*HW){
				// Boundary check
				float val = __half2float(data[idx]);
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
		int total_elements = N*HW;
		mean[cid] = s_sum[0]/total_elements;
		variance[cid] = fmaxf(s_sq_sum[0]/total_elements - mean[cid]*mean[cid], 0.0f);
	}
}
__global__ void layerNormForwardKernel(__half* output, const __half* data, const float* gamma, const float* beta, const float* mean, const float* variance, int N, int C, int HW, float epsilon){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int cid = blockIdx.y;
	if(cid >= C) return; // Guard against excessive blocks
	if(tid < N*HW){
		int idx = (cid*N + tid/HW)*HW + tid % HW;
		if(idx < N*C*HW){
			// Boundary check
			float x = __half2float(data[idx]);
			float norm = (x - mean[cid])/sqrtf(variance[cid] + epsilon);
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
	// Synchronize to catch any errors during kernel execution
	cudaDeviceSynchronize();
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
__device__ bool deviceResult;
__global__ void isNaNKernel(__half* data, int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ bool foundNaN;
	if(threadIdx.x == 0){ foundNaN = false; }
	__syncthreads();
	if(idx < size && __hisnan(data[idx])){ foundNaN = true; }
	__syncthreads();
	if(threadIdx.x == 0 && foundNaN){ deviceResult = true; }
}
extern "C" bool isnanHalf(__half* data, int size){
	bool h_result = false;
	cudaMemcpyToSymbol(deviceResult, &h_result, sizeof(bool));
	auto gridSize = div_ceil(size, BS);
	isNaNKernel<<<gridSize, BS>>>(data, size);
	cudaMemcpyFromSymbol(&h_result, deviceResult, sizeof(bool));
	return h_result;
}