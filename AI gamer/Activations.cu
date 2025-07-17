#include "CuCommon.cuh"
__global__ void LeakyReluKernel(const __half* __restrict__ dataIn, __half* __restrict__ dataOut, const int size, const float negativeSlope){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const auto val = __half2float(dataIn[idx]);
		dataOut[idx] = __float2half(val < 0.0f ? val*negativeSlope : 0.0f);
	}
}
void LeakyReluForward(const __half* dataIn, __half* dataOut, const int size, const float negativeSlope, cudaStream_t stream){
	int blocks, tpb = 128;
	GetLaunchConfig(size, blocks, tpb);
	LeakyReluKernel<<<blocks, tpb, 0, stream>>>(dataIn, dataOut, size, negativeSlope);
}
__global__ void LeakyReluBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ dataIn, const int size, const float negativeSlope){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ grad[idx] *= __float2half(__half2float(dataIn[idx]) < 0.0f ? negativeSlope : 1.0f); }
}
void LeakyReluBackward(__half* grad, const __half* dataIn, const int size, const float negativeSlope, cudaStream_t stream){
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
void SwishForward(const __half* dataIn, __half* outData, const int size, cudaStream_t stream){
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
void SwishBackward(__half* grad, const __half* dataIn, const int size, cudaStream_t stream){
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
void SigmoidForward(const __half* dataIn, __half* dataOut, const int numCtrls, const int numButs, const int size, cudaStream_t cudaStream){
	auto gridSize = DivCeil(size, BS);
	SigmoidForwardKernel<<<gridSize, BS, 0, cudaStream>>>(dataIn, dataOut, numCtrls, numButs, size);
}
__global__ void SigmoidBackwardKernel(__half* __restrict__ grad, const __half* __restrict__ dataIn, const int numCtrls, const int numButs, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size || idx % numCtrls >= numButs) return;
	const float val = __half2float(dataIn[idx]);
	grad[idx] = __float2half(__half2float(grad[idx])*val*(1.0f - val));
}
void SigmoidBackward(__half* grad, const __half* dataIn, const int numCtrls, const int numButs, const int size, cudaStream_t cudaStream){
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
void GELUForward(const __half* dataIn, __half* dataOut, const int size, cudaStream_t stream){
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
void GELUBackward(__half* grad, const __half* dataIn, const int size, cudaStream_t stream){
	int blocks, threads = 512;
	GetLaunchConfig(size, blocks, threads);
	GELUBackwardKernel<<<blocks, threads, 0, stream>>>(grad, dataIn, size);
}