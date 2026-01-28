#include "CuCommon.cuh"
#include <device_launch_parameters.h>
// Constants
constexpr float SQRT_2_PI = 0.7978845608028654f;
constexpr float GELU_COEF_A = 0.044715f;
constexpr int DEFAULT_BLOCK_SIZE = 256;
// ==================== LeakyReLU ====================
__global__ void LeakyReluKernel(const half* __restrict__ dataIn, half* __restrict__ dataOut, const int size, const half negativeSlope){
	const int stride = blockDim.x*gridDim.x;
	const half zero = __float2half(0.0f);
	const half one = __float2half(1.0f);
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const half val = dataIn[idx];
		const half scale = __hlt(val, zero) ? negativeSlope : one;
		dataOut[idx] = __hmul(val, scale);
	}
}
void LeakyReluForward(const half* dataIn, half* dataOut, const int size, const float negativeSlope, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	const half negSlopeHalf = __float2half(negativeSlope);
	LeakyReluKernel<<<blocks, threads, 0, stream>>>(dataIn, dataOut, size, negSlopeHalf);
	checkCUDA(cudaGetLastError());
}
__global__ void LeakyReluBackwardKernel(half* __restrict__ grad, const half* __restrict__ dataIn, const int size, const half negativeSlope){
	const int stride = blockDim.x*gridDim.x;
	const half zero = __float2half(0.0f);
	const half one = __float2half(1.0f);
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const half scale = __hlt(dataIn[idx], zero) ? negativeSlope : one;
		grad[idx] = __hmul(grad[idx], scale);
	}
}
void LeakyReluBackward(half* grad, const half* dataIn, const int size, const float negativeSlope, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	const half negSlopeHalf = __float2half(negativeSlope);
	LeakyReluBackwardKernel<<<blocks, threads, 0, stream>>>(grad, dataIn, size, negSlopeHalf);
	checkCUDA(cudaGetLastError());
}
// ==================== Swish ====================
__global__ void SwishKernel(const half* __restrict__ dataIn, half* __restrict__ outData, const int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const float val = __half2float(dataIn[idx]);
		const float sigmoid = 1.0f/(1.0f + expf(-val));
		outData[idx] = __float2half(val*sigmoid);
	}
}
void SwishForward(const half* dataIn, half* outData, const int size, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	SwishKernel<<<blocks, threads, 0, stream>>>(dataIn, outData, size);
	checkCUDA(cudaGetLastError());
}
__global__ void SwishBackwardKernel(half* __restrict__ grad, const half* __restrict__ dataIn, const int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const float val = __half2float(dataIn[idx]);
		const float sigmoid = 1.0f/(1.0f + expf(-val));
		const float derivative = sigmoid*(1.0f + val*(1.0f - sigmoid));
		grad[idx] = __float2half(__half2float(grad[idx])*derivative);
	}
}
void SwishBackward(half* grad, const half* dataIn, const int size, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	SwishBackwardKernel<<<blocks, threads, 0, stream>>>(grad, dataIn, size);
	checkCUDA(cudaGetLastError());
}
// ==================== Sigmoid ====================
__global__ void SigmoidKernel(const half* __restrict__ dataIn, half* __restrict__ dataOut, const int numCtrls, const int numButs, const int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		if(idx % numCtrls < numButs){
			const float val = __half2float(dataIn[idx]);
			dataOut[idx] = __float2half(1.0f/(1.0f + expf(-val)));
		}
	}
}
void SigmoidForward(const half* dataIn, half* dataOut, const int numCtrls, const int numButs, const int size, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	SigmoidKernel<<<blocks, threads, 0, stream>>>(dataIn, dataOut, numCtrls, numButs, size);
	checkCUDA(cudaGetLastError());
}
__global__ void SigmoidBackwardKernel(half* __restrict__ grad, const half* __restrict__ dataIn, const int numCtrls, const int numButs, const int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		if(idx % numCtrls < numButs){
			const float val = __half2float(dataIn[idx]);
			const float derivative = val*(1.0f - val);
			grad[idx] = __float2half(__half2float(grad[idx])*derivative);
		}
	}
}
void SigmoidBackward(half* grad, const half* dataIn, const int numCtrls, const int numButs, const int size, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	SigmoidBackwardKernel<<<blocks, threads, 0, stream>>>(grad, dataIn, numCtrls, numButs, size);
	checkCUDA(cudaGetLastError());
}
// ==================== GELU ====================
__global__ void GELUForwardKernel(const half* __restrict__ dataIn, half* __restrict__ dataOut, const int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const float x = __half2float(dataIn[idx]);
		const float xCubed = x*x*x;
		const float tanhArg = SQRT_2_PI*(x + GELU_COEF_A*xCubed);
		const float cdf = 0.5f*(1.0f + tanhf(tanhArg));
		dataOut[idx] = __float2half(x*cdf);
	}
}
void GELUForward(const half* dataIn, half* dataOut, const int size, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	GELUForwardKernel<<<blocks, threads, 0, stream>>>(dataIn, dataOut, size);
	checkCUDA(cudaGetLastError());
}
__global__ void GELUBackwardKernel(half* __restrict__ grad, const half* __restrict__ dataIn, const int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const float x = __half2float(dataIn[idx]);
		const float xSquared = x*x;
		const float xCubed = xSquared*x;
		const float tanhArg = SQRT_2_PI*(x + GELU_COEF_A*xCubed);
		const float tanhVal = tanhf(tanhArg);
		const float cdf = 0.5f*(1.0f + tanhVal);
		const float sechSquared = 1.0f - tanhVal*tanhVal;
		const float pdf = 0.5f*SQRT_2_PI*(1.0f + 3.0f*GELU_COEF_A*xSquared)*sechSquared;
		const float derivative = cdf + x*pdf;
		grad[idx] = __float2half(__half2float(grad[idx])*derivative);
	}
}
void GELUBackward(half* grad, const half* dataIn, const int size, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	GELUBackwardKernel<<<blocks, threads, 0, stream>>>(grad, dataIn, size);
	checkCUDA(cudaGetLastError());
}
// ==================== Asinh ====================
__global__ void AsinhForwardKernel(const half* __restrict__ x, half* __restrict__ y, int size, float alpha){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		y[idx] = __float2half(asinhf(__half2float(x[idx])/alpha));
	}
}
void AsinhForward(const half* x, half* y, int size, float alpha, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	AsinhForwardKernel<<<blocks, threads, 0, stream>>>(x, y, size, alpha);
	checkCUDA(cudaGetLastError());
}
__global__ void AsinhBackwardKernel(half* __restrict__ grad, const half* __restrict__ y, int size, float alpha){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		float deriv = 1.0f/coshf(__half2float(y[idx]));
		deriv *= 1.0f/alpha;
		grad[idx] = __float2half(__half2float(grad[idx])*deriv);
	}
}
void AsinhBackward(half* grad, const half* activated, int size, float alpha, cudaStream_t stream){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	AsinhBackwardKernel<<<blocks, threads, 0, stream>>>(grad, activated, size, alpha);
	checkCUDA(cudaGetLastError());
}
// ==================== Tanh ====================
__global__ void TanhHalfKernel(__half* data, int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const float val = tanhf(__half2float(data[idx]));
		data[idx] = __float2half(val);
	}
}
void TanhInPlace(__half* data, int size){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	TanhHalfKernel<<<blocks, threads>>>(data, size);
	checkCUDA(cudaGetLastError());
}
__global__ void TanhBackwardHalfKernel(__half* grad, const __half* activations, int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const float act = __half2float(activations[idx]);
		const float g = __half2float(grad[idx]);
		const float derivative = 1.0f - act*act;
		grad[idx] = __float2half(g*derivative);
	}
}
void TanhBackward(__half* grad, const __half* activations, int size){
	size_t blocks, threads = DEFAULT_BLOCK_SIZE;
	GetLaunchConfigGridStride(size, blocks, threads);
	TanhBackwardHalfKernel<<<blocks, threads>>>(grad, activations, size);
	checkCUDA(cudaGetLastError());
}