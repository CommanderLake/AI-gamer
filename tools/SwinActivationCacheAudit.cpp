#include "APICommon.h"
#include "SwinBlockLayer.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace{
	void CheckCuda(const cudaError_t status, const char* operation){
		if(status != cudaSuccess){ throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status)); }
	}

	std::vector<__half> CopyToHost(const __half* device, const size_t elements){
		std::vector<__half> host(elements);
		CheckCuda(cudaMemcpy(host.data(), device, elements*sizeof(__half), cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
		return host;
	}

	float MaxAbsDifference(const std::vector<__half>& a, const std::vector<__half>& b){
		if(a.size() != b.size()){ throw std::runtime_error("comparison size mismatch"); }
		float maxDifference = 0.0f;
		for(size_t i = 0; i < a.size(); ++i){
			maxDifference = std::max(maxDifference, std::fabs(__half2float(a[i]) - __half2float(b[i])));
		}
		return maxDifference;
	}

	void CopyParameters(SwinBlockLayer& source, SwinBlockLayer& destination, const std::string& path){
		std::vector<unsigned char> buffer(source.GetParameterSize());
		{
			std::ofstream file(path, std::ios::binary | std::ios::trunc);
			if(!file){ throw std::runtime_error("failed to create parameter file"); }
			source.SaveParameters(file, buffer.data());
		}
		{
			std::ifstream file(path, std::ios::binary);
			if(!file){ throw std::runtime_error("failed to reopen parameter file"); }
			destination.LoadParameters(file, buffer.data());
		}
		std::remove(path.c_str());
	}
}

int main(){
	try{
		InitCUDA();
		if(InitCublas() != CLNN_STATUS_SUCCESS){ throw std::runtime_error("InitCublas failed"); }
		constexpr int batchSize = 1;
		constexpr int tokens = 64;
		constexpr int embedDim = 32;
		constexpr int elements = batchSize*tokens*embedDim;
		const auto makeBlock = [=](const bool cacheActivations){
			return std::unique_ptr<SwinBlockLayer>(new SwinBlockLayer(
				batchSize, tokens, embedDim, embedDim*4, 1, 8, 8, 8, 8, 0, 0, 0.0f,
				cacheActivations ? "Cached" : "Recomputed", true, 0.0f, 1, Xavier,
				nullptr, nullptr, nullptr, nullptr, true, nullptr, nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr, nullptr, cacheActivations));
		};

		std::unique_ptr<SwinBlockLayer> recomputed;
		std::unique_ptr<SwinBlockLayer> cached;
		do{
			const auto constructionSecond = std::time(nullptr);
			recomputed = makeBlock(false);
			cached = makeBlock(true);
			if(std::time(nullptr) == constructionSecond){ break; }
			recomputed.reset();
			cached.reset();
		} while(true);
		CopyParameters(*recomputed, *cached, "swin-cache-audit-parameters.bin");

		std::vector<__half> input(elements);
		std::vector<__half> gradient(elements);
		for(int i = 0; i < elements; ++i){
			input[i] = __float2half(static_cast<float>((i*17)%257 - 128)/128.0f);
			gradient[i] = __float2half(static_cast<float>((i*29)%251 - 125)/256.0f);
		}
		__half* inputRecomputed = nullptr;
		__half* inputCached = nullptr;
		__half* gradRecomputed = nullptr;
		__half* gradCached = nullptr;
		const size_t bytes = static_cast<size_t>(elements)*sizeof(__half);
		CheckCuda(cudaMalloc(&inputRecomputed, bytes), "cudaMalloc inputRecomputed");
		CheckCuda(cudaMalloc(&inputCached, bytes), "cudaMalloc inputCached");
		CheckCuda(cudaMalloc(&gradRecomputed, bytes), "cudaMalloc gradRecomputed");
		CheckCuda(cudaMalloc(&gradCached, bytes), "cudaMalloc gradCached");
		CheckCuda(cudaMemcpy(inputRecomputed, input.data(), bytes, cudaMemcpyHostToDevice), "copy inputRecomputed");
		CheckCuda(cudaMemcpy(inputCached, input.data(), bytes, cudaMemcpyHostToDevice), "copy inputCached");
		CheckCuda(cudaMemcpy(gradRecomputed, gradient.data(), bytes, cudaMemcpyHostToDevice), "copy gradRecomputed");
		CheckCuda(cudaMemcpy(gradCached, gradient.data(), bytes, cudaMemcpyHostToDevice), "copy gradCached");

		const auto outputRecomputed = CopyToHost(recomputed->Forward(inputRecomputed), elements);
		const auto outputCached = CopyToHost(cached->Forward(inputCached), elements);
		const auto inputGradRecomputed = CopyToHost(recomputed->Backward(gradRecomputed), elements);
		const auto inputGradCached = CopyToHost(cached->Backward(gradCached), elements);
		CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

		const float outputDifference = MaxAbsDifference(outputRecomputed, outputCached);
		const float gradientDifference = MaxAbsDifference(inputGradRecomputed, inputGradCached);
		std::printf("Swin activation cache audit: output max abs %.8g, input-gradient max abs %.8g\n", outputDifference, gradientDifference);

		cudaFree(inputRecomputed);
		cudaFree(inputCached);
		cudaFree(gradRecomputed);
		cudaFree(gradCached);
		return outputDifference == 0.0f && gradientDifference == 0.0f ? 0 : 1;
	} catch(const std::exception& error){
		std::fprintf(stderr, "Swin activation cache audit failed: %s\n", error.what());
		return 2;
	}
}
