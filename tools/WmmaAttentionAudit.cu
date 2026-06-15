#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

size_t MPC = 80;
size_t GS = 2560;
size_t CPM = 64;
curandGenerator_t cuRandGen = nullptr;

void GetLaunchConfigGridStride(const size_t n, size_t& blocks, size_t& tpb){
	if(tpb == 0 || tpb > 1024){ tpb = 256; }
	blocks = std::max<size_t>(1, std::min((n + tpb - 1)/tpb, GS));
}

struct Shape{
	int batch;
	int heads;
	int tokens;
	int headDim;
};

struct ErrorStats{
	float maxAbs = 0.0f;
	float rmse = 0.0f;
	float maxReference = 0.0f;
	size_t maxIndex = 0;
};

static void CheckCuda(cudaError_t status, const char* operation){
	if(status != cudaSuccess){
		std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
		std::exit(2);
	}
}

static float QuantizeHalf(float value){
	return __half2float(__float2half(value));
}

static size_t EmbeddingIndex(const Shape& shape, int batch, int head, int token, int dim){
	return ((static_cast<size_t>(batch)*shape.heads + head)*shape.tokens + token)*shape.headDim + dim;
}

static size_t AttentionIndex(const Shape& shape, int batch, int head, int row, int col){
	return ((static_cast<size_t>(batch)*shape.heads + head)*shape.tokens + row)*shape.tokens + col;
}

static ErrorStats Compare(const std::vector<float>& actual, const std::vector<float>& expected){
	ErrorStats stats;
	double squareSum = 0.0;
	for(size_t i = 0; i < actual.size(); ++i){
		const float error = std::fabs(actual[i] - expected[i]);
		if(error > stats.maxAbs){
			stats.maxAbs = error;
			stats.maxIndex = i;
		}
		stats.maxReference = std::max(stats.maxReference, std::fabs(expected[i]));
		squareSum += static_cast<double>(error)*error;
	}
	stats.rmse = std::sqrt(static_cast<float>(squareSum/actual.size()));
	return stats;
}

static std::vector<float> HalfToFloat(const std::vector<__half>& input){
	std::vector<float> output(input.size());
	for(size_t i = 0; i < input.size(); ++i){ output[i] = __half2float(input[i]); }
	return output;
}

static bool TestPacking(){
	const int batch = 2;
	const int tokens = 17;
	const int heads = 3;
	const int headDim = 32;
	const int embedDim = heads*headDim;
	const size_t elements = static_cast<size_t>(batch)*tokens*embedDim;
	const size_t bytes = elements*sizeof(__half);
	std::vector<__half> q(elements);
	std::vector<__half> k(elements);
	std::vector<__half> v(elements);
	for(size_t i = 0; i < elements; ++i){
		q[i] = __float2half(static_cast<float>((static_cast<int>(i) % 251) - 125)/64.0f);
		k[i] = __float2half(static_cast<float>((static_cast<int>(i*3) % 241) - 120)/64.0f);
		v[i] = __float2half(static_cast<float>((static_cast<int>(i*7) % 239) - 119)/64.0f);
	}

	__half* dQ = nullptr;
	__half* dK = nullptr;
	__half* dV = nullptr;
	__half* packedQ = nullptr;
	__half* packedK = nullptr;
	__half* packedV = nullptr;
	__half* roundQ = nullptr;
	__half* roundK = nullptr;
	__half* roundV = nullptr;
	CheckCuda(cudaMalloc(&dQ, bytes), "cudaMalloc packing q");
	CheckCuda(cudaMalloc(&dK, bytes), "cudaMalloc packing k");
	CheckCuda(cudaMalloc(&dV, bytes), "cudaMalloc packing v");
	CheckCuda(cudaMalloc(&packedQ, bytes), "cudaMalloc packing packedQ");
	CheckCuda(cudaMalloc(&packedK, bytes), "cudaMalloc packing packedK");
	CheckCuda(cudaMalloc(&packedV, bytes), "cudaMalloc packing packedV");
	CheckCuda(cudaMalloc(&roundQ, bytes), "cudaMalloc packing roundQ");
	CheckCuda(cudaMalloc(&roundK, bytes), "cudaMalloc packing roundK");
	CheckCuda(cudaMalloc(&roundV, bytes), "cudaMalloc packing roundV");
	CheckCuda(cudaMemcpy(dQ, q.data(), bytes, cudaMemcpyHostToDevice), "copy packing q");
	CheckCuda(cudaMemcpy(dK, k.data(), bytes, cudaMemcpyHostToDevice), "copy packing k");
	CheckCuda(cudaMemcpy(dV, v.data(), bytes, cudaMemcpyHostToDevice), "copy packing v");

	PackColumnsToHeads(dQ, dK, dV, packedQ, packedK, packedV, batch, tokens, embedDim, heads);
	PackHeadsToColumns(packedQ, packedK, packedV, roundQ, roundK, roundV, batch, tokens, embedDim, heads);
	std::vector<__half> actualQ(elements);
	std::vector<__half> actualK(elements);
	std::vector<__half> actualV(elements);
	CheckCuda(cudaMemcpy(actualQ.data(), roundQ, bytes, cudaMemcpyDeviceToHost), "copy packing roundQ");
	CheckCuda(cudaMemcpy(actualK.data(), roundK, bytes, cudaMemcpyDeviceToHost), "copy packing roundK");
	CheckCuda(cudaMemcpy(actualV.data(), roundV, bytes, cudaMemcpyDeviceToHost), "copy packing roundV");

	PackColumnsToHeads(dQ, packedQ, batch, tokens, embedDim, heads);
	PackHeadsToColumns(packedQ, roundQ, batch, tokens, embedDim, heads);
	std::vector<__half> actualSingle(elements);
	CheckCuda(cudaMemcpy(actualSingle.data(), roundQ, bytes, cudaMemcpyDeviceToHost), "copy packing single");

	bool passed = true;
	for(size_t i = 0; i < elements; ++i){
		passed = passed &&
			__half2float(q[i]) == __half2float(actualQ[i]) &&
			__half2float(k[i]) == __half2float(actualK[i]) &&
			__half2float(v[i]) == __half2float(actualV[i]) &&
			__half2float(q[i]) == __half2float(actualSingle[i]);
	}
	std::printf("Packing round-trip: %s\n", passed ? "PASS" : "FAIL");

	cudaFree(dQ);
	cudaFree(dK);
	cudaFree(dV);
	cudaFree(packedQ);
	cudaFree(packedK);
	cudaFree(packedV);
	cudaFree(roundQ);
	cudaFree(roundK);
	cudaFree(roundV);
	return passed;
}

static void CpuReference(
	const Shape& shape,
	const std::vector<float>& q,
	const std::vector<float>& k,
	const std::vector<float>& v,
	const std::vector<float>& dOut,
	const std::vector<float>& mask,
	const std::vector<float>& relBias,
	const std::vector<int>& relIndex,
	std::vector<float>& output,
	std::vector<float>& attention,
	std::vector<float>& dLogit,
	std::vector<float>& dQ,
	std::vector<float>& dK,
	std::vector<float>& dV){
	const float scale = 1.0f/std::sqrt(static_cast<float>(shape.headDim));
	const size_t embeddingElements = static_cast<size_t>(shape.batch)*shape.heads*shape.tokens*shape.headDim;
	const size_t attentionElements = static_cast<size_t>(shape.batch)*shape.heads*shape.tokens*shape.tokens;
	output.assign(embeddingElements, 0.0f);
	attention.assign(attentionElements, 0.0f);
	dLogit.assign(attentionElements, 0.0f);
	dQ.assign(embeddingElements, 0.0f);
	dK.assign(embeddingElements, 0.0f);
	dV.assign(embeddingElements, 0.0f);
	std::vector<float> rawDAtt(attentionElements, 0.0f);
	std::vector<float> exponentials(shape.tokens);

	for(int b = 0; b < shape.batch; ++b){
		for(int h = 0; h < shape.heads; ++h){
			for(int row = 0; row < shape.tokens; ++row){
				float maxLogit = -std::numeric_limits<float>::infinity();
				for(int col = 0; col < shape.tokens; ++col){
					const float maskValue = mask[static_cast<size_t>(row)*shape.tokens + col];
					float logit = maskValue;
					logit += relBias[static_cast<size_t>(h)*relIndex.size() + relIndex[static_cast<size_t>(row)*shape.tokens + col]];
					for(int d = 0; d < shape.headDim; ++d){
						const float qScaled = QuantizeHalf(q[EmbeddingIndex(shape, b, h, row, d)]*scale);
						logit += qScaled*k[EmbeddingIndex(shape, b, h, col, d)];
					}
					exponentials[col] = logit;
					if(maskValue > -1000.0f){ maxLogit = std::max(maxLogit, logit); }
				}
				float sum = 0.0f;
				for(int col = 0; col < shape.tokens; ++col){
					const float diff = exponentials[col] - maxLogit;
					const bool isMasked = mask[static_cast<size_t>(row)*shape.tokens + col] <= -1000.0f;
					const float exponential = !isMasked && diff > -12.0f && diff < 20.0f ? std::exp(diff) : 0.0f;
					exponentials[col] = exponential;
					sum += exponential;
				}
				for(int col = 0; col < shape.tokens; ++col){
					const size_t attIdx = AttentionIndex(shape, b, h, row, col);
					attention[attIdx] = QuantizeHalf(sum > 1e-10f ? exponentials[col]/sum : 0.0f);
				}
				for(int d = 0; d < shape.headDim; ++d){
					float accum = 0.0f;
					for(int col = 0; col < shape.tokens; ++col){
						accum += QuantizeHalf(exponentials[col])*v[EmbeddingIndex(shape, b, h, col, d)];
					}
					output[EmbeddingIndex(shape, b, h, row, d)] = QuantizeHalf(sum > 1e-10f ? accum/sum : 0.0f);
				}
			}
		}
	}

	for(int b = 0; b < shape.batch; ++b){
		for(int h = 0; h < shape.heads; ++h){
			for(int row = 0; row < shape.tokens; ++row){
				float rowSum = 0.0f;
				for(int col = 0; col < shape.tokens; ++col){
					float raw = 0.0f;
					for(int d = 0; d < shape.headDim; ++d){
						raw += dOut[EmbeddingIndex(shape, b, h, row, d)]*v[EmbeddingIndex(shape, b, h, col, d)];
					}
					const size_t attIdx = AttentionIndex(shape, b, h, row, col);
					rawDAtt[attIdx] = raw;
					rowSum += raw*attention[attIdx];
				}
				for(int col = 0; col < shape.tokens; ++col){
					const size_t attIdx = AttentionIndex(shape, b, h, row, col);
					dLogit[attIdx] = attention[attIdx]*(rawDAtt[attIdx] - rowSum);
				}
			}
			for(int row = 0; row < shape.tokens; ++row){
				for(int d = 0; d < shape.headDim; ++d){
					float qAccum = 0.0f;
					for(int col = 0; col < shape.tokens; ++col){
						qAccum += QuantizeHalf(dLogit[AttentionIndex(shape, b, h, row, col)])*k[EmbeddingIndex(shape, b, h, col, d)];
					}
					dQ[EmbeddingIndex(shape, b, h, row, d)] = QuantizeHalf(qAccum*scale);
				}
			}
			for(int col = 0; col < shape.tokens; ++col){
				for(int d = 0; d < shape.headDim; ++d){
					float kAccum = 0.0f;
					float vAccum = 0.0f;
					for(int row = 0; row < shape.tokens; ++row){
						kAccum += QuantizeHalf(dLogit[AttentionIndex(shape, b, h, row, col)])*q[EmbeddingIndex(shape, b, h, row, d)];
						vAccum += attention[AttentionIndex(shape, b, h, row, col)]*dOut[EmbeddingIndex(shape, b, h, row, d)];
					}
					dK[EmbeddingIndex(shape, b, h, col, d)] = QuantizeHalf(kAccum*scale);
					dV[EmbeddingIndex(shape, b, h, col, d)] = QuantizeHalf(vAccum);
				}
			}
		}
	}
}

template<typename T>
static T* CopyToDevice(const std::vector<T>& host){
	T* device = nullptr;
	CheckCuda(cudaMalloc(&device, host.size()*sizeof(T)), "cudaMalloc");
	CheckCuda(cudaMemcpy(device, host.data(), host.size()*sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy host to device");
	return device;
}

template<typename T>
static std::vector<T> CopyFromDevice(const T* device, size_t count){
	std::vector<T> host(count);
	CheckCuda(cudaMemcpy(host.data(), device, count*sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
	return host;
}

static bool RunCase(const Shape& shape, int benchmarkIterations){
	const size_t embeddingElements = static_cast<size_t>(shape.batch)*shape.heads*shape.tokens*shape.headDim;
	const size_t attentionElements = static_cast<size_t>(shape.batch)*shape.heads*shape.tokens*shape.tokens;
	const int relPosSize = shape.tokens*shape.tokens;
	std::vector<float> qFloat(embeddingElements);
	std::vector<float> kFloat(embeddingElements);
	std::vector<float> vFloat(embeddingElements);
	std::vector<float> dOutFloat(embeddingElements);
	for(size_t i = 0; i < embeddingElements; ++i){
		qFloat[i] = QuantizeHalf(0.35f*std::sin(0.071f*static_cast<float>(i + 1)));
		kFloat[i] = QuantizeHalf(0.30f*std::cos(0.053f*static_cast<float>(i + 3)));
		vFloat[i] = QuantizeHalf(0.40f*std::sin(0.037f*static_cast<float>(i + 7)));
		dOutFloat[i] = QuantizeHalf(0.25f*std::cos(0.043f*static_cast<float>(i + 11)));
	}
	std::vector<__half> qHalf(embeddingElements);
	std::vector<__half> kHalf(embeddingElements);
	std::vector<__half> vHalf(embeddingElements);
	std::vector<__half> dOutHalf(embeddingElements);
	for(size_t i = 0; i < embeddingElements; ++i){
		qHalf[i] = __float2half(qFloat[i]);
		kHalf[i] = __float2half(kFloat[i]);
		vHalf[i] = __float2half(vFloat[i]);
		dOutHalf[i] = __float2half(dOutFloat[i]);
	}
	std::vector<float> mask(attentionElements/(shape.batch*shape.heads), 0.0f);
	for(int row = 0; row < shape.tokens; ++row){
		for(int col = 0; col < shape.tokens; ++col){
			if((row < shape.tokens/2) != (col < shape.tokens/2)){ mask[static_cast<size_t>(row)*shape.tokens + col] = -10000.0f; }
		}
	}
	for(int col = 0; col < shape.tokens; ++col){ mask[col] = -10000.0f; }
	std::vector<int> relIndex(static_cast<size_t>(shape.tokens)*shape.tokens);
	for(size_t i = 0; i < relIndex.size(); ++i){ relIndex[i] = static_cast<int>(i); }
	std::vector<float> relBias(static_cast<size_t>(shape.heads)*relPosSize);
	for(size_t i = 0; i < relBias.size(); ++i){ relBias[i] = 0.02f*std::sin(0.017f*static_cast<float>(i)); }

	std::vector<float> expectedOutput;
	std::vector<float> expectedAttention;
	std::vector<float> expectedDLogit;
	std::vector<float> expectedDQ;
	std::vector<float> expectedDK;
	std::vector<float> expectedDV;
	CpuReference(shape, qFloat, kFloat, vFloat, dOutFloat, mask, relBias, relIndex, expectedOutput, expectedAttention, expectedDLogit, expectedDQ, expectedDK, expectedDV);

	__half* q = CopyToDevice(qHalf);
	__half* k = CopyToDevice(kHalf);
	__half* v = CopyToDevice(vHalf);
	__half* dOut = CopyToDevice(dOutHalf);
	float* deviceMask = CopyToDevice(mask);
	float* deviceRelBias = CopyToDevice(relBias);
	int* deviceRelIndex = CopyToDevice(relIndex);
	__half* output = nullptr;
	__half* attention = nullptr;
	__half* dQ = nullptr;
	__half* dK = k;
	__half* dV = nullptr;
	float* dLogit = nullptr;
	const bool usesInPlaceDAtt = shape.headDim == 32 && shape.tokens <= 64;
	CheckCuda(cudaMalloc(&output, embeddingElements*sizeof(__half)), "cudaMalloc output");
	CheckCuda(cudaMalloc(&attention, attentionElements*sizeof(__half)), "cudaMalloc attention");
	CheckCuda(cudaMalloc(&dQ, embeddingElements*sizeof(__half)), "cudaMalloc dQ");
	CheckCuda(cudaMalloc(&dV, embeddingElements*sizeof(__half)), "cudaMalloc dV");
	if(!usesInPlaceDAtt){ CheckCuda(cudaMalloc(&dLogit, attentionElements*sizeof(float)), "cudaMalloc dLogit"); }

	WmmaAttention(q, k, v, output, attention, deviceMask, deviceRelBias, deviceRelIndex, relPosSize, shape.batch, shape.tokens, shape.headDim, shape.heads, 1, 1);
	CheckCuda(cudaDeviceSynchronize(), "WmmaAttention synchronize");
	const auto actualOutput = HalfToFloat(CopyFromDevice(output, embeddingElements));
	const auto actualAttention = HalfToFloat(CopyFromDevice(attention, attentionElements));
	WmmaAttentionBackward(q, k, v, dOut, attention, dQ, dK, dV, dLogit, usesInPlaceDAtt ? 0 : attentionElements, shape.batch, shape.tokens, shape.headDim, shape.heads);
	CheckCuda(cudaDeviceSynchronize(), "WmmaAttentionBackward synchronize");

	const auto actualDQ = HalfToFloat(CopyFromDevice(dQ, embeddingElements));
	const auto actualDK = HalfToFloat(CopyFromDevice(dK, embeddingElements));
	const auto actualDV = HalfToFloat(CopyFromDevice(dV, embeddingElements));
	const auto actualDLogit = usesInPlaceDAtt ?
		HalfToFloat(CopyFromDevice(attention, attentionElements)) : CopyFromDevice(dLogit, attentionElements);
	CheckCuda(cudaMemcpy(k, kHalf.data(), embeddingElements*sizeof(__half), cudaMemcpyHostToDevice), "restore aliased K");

	const ErrorStats outputError = Compare(actualOutput, expectedOutput);
	const ErrorStats attentionError = Compare(actualAttention, expectedAttention);
	const ErrorStats dLogitError = Compare(actualDLogit, expectedDLogit);
	const ErrorStats dQError = Compare(actualDQ, expectedDQ);
	const ErrorStats dKError = Compare(actualDK, expectedDK);
	const ErrorStats dVError = Compare(actualDV, expectedDV);

	std::printf("\nShape B=%d H=%d T=%d D=%d\n", shape.batch, shape.heads, shape.tokens, shape.headDim);
	auto printError = [](const char* name, const ErrorStats& error){
		std::printf("  %-10s maxAbs=%-12.6g rmse=%-12.6g refMax=%-12.6g index=%zu\n", name, error.maxAbs, error.rmse, error.maxReference, error.maxIndex);
	};
	printError("output", outputError);
	printError("attention", attentionError);
	printError("dLogit", dLogitError);
	printError("dQ", dQError);
	printError("dK", dKError);
	printError("dV", dVError);

	cudaEvent_t start;
	cudaEvent_t stop;
	CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
	CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
	for(int i = 0; i < 10; ++i){
		WmmaAttention(q, k, v, output, attention, deviceMask, deviceRelBias, deviceRelIndex, relPosSize, shape.batch, shape.tokens, shape.headDim, shape.heads, 1, 1);
	}
	CheckCuda(cudaDeviceSynchronize(), "warmup synchronize");
	CheckCuda(cudaEventRecord(start), "cudaEventRecord start");
	for(int i = 0; i < benchmarkIterations; ++i){
		WmmaAttention(q, k, v, output, attention, deviceMask, deviceRelBias, deviceRelIndex, relPosSize, shape.batch, shape.tokens, shape.headDim, shape.heads, 1, 1);
	}
	CheckCuda(cudaEventRecord(stop), "cudaEventRecord stop");
	CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
	float forwardMs = 0.0f;
	CheckCuda(cudaEventElapsedTime(&forwardMs, start, stop), "cudaEventElapsedTime");
	CheckCuda(cudaEventRecord(start), "cudaEventRecord start");
	for(int i = 0; i < benchmarkIterations; ++i){
		WmmaAttention(q, k, v, output, attention, deviceMask, deviceRelBias, deviceRelIndex, relPosSize, shape.batch, shape.tokens, shape.headDim, shape.heads, 1, 1);
		WmmaAttentionBackward(q, k, v, dOut, attention, dQ, dK, dV, dLogit, usesInPlaceDAtt ? 0 : attentionElements, shape.batch, shape.tokens, shape.headDim, shape.heads);
	}
	CheckCuda(cudaEventRecord(stop), "cudaEventRecord stop");
	CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
	float backwardMs = 0.0f;
	CheckCuda(cudaEventElapsedTime(&backwardMs, start, stop), "cudaEventElapsedTime");
	std::printf("  timing     forward=%.4f ms forward+backward=%.4f ms (%d iterations)\n", forwardMs/benchmarkIterations, backwardMs/benchmarkIterations, benchmarkIterations);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(q);
	cudaFree(k);
	cudaFree(v);
	cudaFree(dOut);
	cudaFree(deviceMask);
	cudaFree(deviceRelBias);
	cudaFree(deviceRelIndex);
	cudaFree(output);
	cudaFree(attention);
	cudaFree(dQ);
	cudaFree(dV);
	cudaFree(dLogit);

	const bool passed = outputError.maxAbs < 0.003f && outputError.rmse < 0.001f &&
		attentionError.maxAbs < 0.002f && dLogitError.maxAbs < 0.002f &&
		dQError.maxAbs < 0.002f && dKError.maxAbs < 0.002f && dVError.maxAbs < 0.002f;
	std::printf("  result     %s\n", passed ? "PASS" : "FAIL");
	return passed;
}

static void BenchmarkShape(const Shape& shape, int iterations){
	const size_t embeddingElements = static_cast<size_t>(shape.batch)*shape.heads*shape.tokens*shape.headDim;
	const size_t attentionElements = static_cast<size_t>(shape.batch)*shape.heads*shape.tokens*shape.tokens;
	std::vector<__half> embedding(embeddingElements, __float2half(0.01f));
	std::vector<float> mask(static_cast<size_t>(shape.tokens)*shape.tokens, 0.0f);
	std::vector<int> relIndex(static_cast<size_t>(shape.tokens)*shape.tokens);
	const int relPosSize = std::max(1, (2*static_cast<int>(std::sqrt(static_cast<float>(shape.tokens))) - 1));
	const int relPosElements = relPosSize*relPosSize;
	for(size_t i = 0; i < relIndex.size(); ++i){ relIndex[i] = static_cast<int>(i % relPosElements); }
	std::vector<float> relBias(static_cast<size_t>(shape.heads)*relPosElements, 0.0f);
	__half* q = CopyToDevice(embedding);
	__half* k = CopyToDevice(embedding);
	__half* v = CopyToDevice(embedding);
	__half* dOut = CopyToDevice(embedding);
	float* deviceMask = CopyToDevice(mask);
	int* deviceRelIndex = CopyToDevice(relIndex);
	float* deviceRelBias = CopyToDevice(relBias);
	__half* output = nullptr;
	__half* attention = nullptr;
	__half* dQ = nullptr;
	__half* dK = nullptr;
	__half* dV = nullptr;
	float* dLogit = nullptr;
	CheckCuda(cudaMalloc(&output, embeddingElements*sizeof(__half)), "cudaMalloc benchmark output");
	CheckCuda(cudaMalloc(&attention, attentionElements*sizeof(__half)), "cudaMalloc benchmark attention");
	CheckCuda(cudaMalloc(&dQ, embeddingElements*sizeof(__half)), "cudaMalloc benchmark dQ");
	CheckCuda(cudaMalloc(&dK, embeddingElements*sizeof(__half)), "cudaMalloc benchmark dK");
	CheckCuda(cudaMalloc(&dV, embeddingElements*sizeof(__half)), "cudaMalloc benchmark dV");
	const bool usesInPlaceDAtt = shape.headDim == 32 && shape.tokens <= 64;
	if(!usesInPlaceDAtt){ CheckCuda(cudaMalloc(&dLogit, attentionElements*sizeof(float)), "cudaMalloc benchmark dLogit"); }
	for(int i = 0; i < 5; ++i){
		WmmaAttention(q, k, v, output, attention, deviceMask, deviceRelBias, deviceRelIndex, relPosElements, shape.batch, shape.tokens, shape.headDim, shape.heads, 1, 1);
		WmmaAttentionBackward(q, k, v, dOut, attention, dQ, dK, dV, dLogit, usesInPlaceDAtt ? 0 : attentionElements, shape.batch, shape.tokens, shape.headDim, shape.heads);
	}
	CheckCuda(cudaDeviceSynchronize(), "benchmark warmup");
	cudaEvent_t start;
	cudaEvent_t stop;
	CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
	CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
	CheckCuda(cudaEventRecord(start), "cudaEventRecord");
	for(int i = 0; i < iterations; ++i){
		WmmaAttention(q, k, v, output, attention, deviceMask, deviceRelBias, deviceRelIndex, relPosElements, shape.batch, shape.tokens, shape.headDim, shape.heads, 1, 1);
	}
	CheckCuda(cudaEventRecord(stop), "cudaEventRecord");
	CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
	float forwardMs = 0.0f;
	CheckCuda(cudaEventElapsedTime(&forwardMs, start, stop), "cudaEventElapsedTime");
	CheckCuda(cudaEventRecord(start), "cudaEventRecord");
	for(int i = 0; i < iterations; ++i){
		WmmaAttention(q, k, v, output, attention, deviceMask, deviceRelBias, deviceRelIndex, relPosElements, shape.batch, shape.tokens, shape.headDim, shape.heads, 1, 1);
		WmmaAttentionBackward(q, k, v, dOut, attention, dQ, dK, dV, dLogit, usesInPlaceDAtt ? 0 : attentionElements, shape.batch, shape.tokens, shape.headDim, shape.heads);
	}
	CheckCuda(cudaEventRecord(stop), "cudaEventRecord");
	CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
	float combinedMs = 0.0f;
	CheckCuda(cudaEventElapsedTime(&combinedMs, start, stop), "cudaEventElapsedTime");
	const float backwardMs = std::max(0.0f, combinedMs - forwardMs);
	std::printf("  B=%-4d H=%-3d T=%-3d D=%-3d forward=%8.4f ms backward=%8.4f ms attention=%6.1f MiB dAtt=%6.1f MiB\n",
		shape.batch, shape.heads, shape.tokens, shape.headDim, forwardMs/iterations, backwardMs/iterations,
		attentionElements*sizeof(__half)/(1024.0f*1024.0f), usesInPlaceDAtt ? 0.0f : attentionElements*sizeof(float)/(1024.0f*1024.0f));
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(q);
	cudaFree(k);
	cudaFree(v);
	cudaFree(dOut);
	cudaFree(deviceMask);
	cudaFree(deviceRelIndex);
	cudaFree(deviceRelBias);
	cudaFree(output);
	cudaFree(attention);
	cudaFree(dQ);
	cudaFree(dK);
	cudaFree(dV);
	cudaFree(dLogit);
}

int main(int argc, char** argv){
	cudaDeviceProp properties = {};
	CheckCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
	std::printf("GPU: %s, compute capability %d.%d\n", properties.name, properties.major, properties.minor);
	if(argc > 1 && std::string(argv[1]) == "--profile-stage0"){
		BenchmarkShape({640, 8, 64, 32}, 20);
		return 0;
	}
	const Shape cases[] = {
		{1, 2, 4, 32},
		{1, 2, 16, 32},
		{1, 2, 17, 24},
		{1, 2, 32, 32},
		{1, 2, 64, 32},
		{1, 2, 128, 32},
		{1, 2, 256, 32}
	};
	bool passed = TestPacking();
	for(const Shape& shape : cases){ passed = RunCase(shape, 100) && passed; }
	std::printf("\nCurrent Swin stage kernel timings\n");
	BenchmarkShape({640, 8, 64, 32}, 20);
	BenchmarkShape({160, 16, 64, 32}, 20);
	BenchmarkShape({160, 32, 16, 32}, 20);
	BenchmarkShape({160, 64, 4, 32}, 20);
	return passed ? 0 : 1;
}
