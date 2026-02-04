#include "NNCommon.h"
#include <mkl.h>
#include <mkl_lapacke.h>
#include <mkl_vsl.h>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <malloc.h>
#include <cuda_runtime_api.h>
__half* matrixH_ = nullptr;
float* matrixF_ = nullptr;
size_t matrixSize_ = 0;
float* tau_ = nullptr;
size_t tauSize_ = 0;
float* work_ = nullptr;
size_t workSize_ = 0;
float* matrixT_ = nullptr;
size_t matrixTSize_ = 0;
VSLStreamStatePtr stream_ = nullptr;
static bool EnsureMatrixStorage(const size_t matrixSize){
	if(matrixSize_ >= matrixSize){ return true; }
	if(matrixSize_ > 0){
		_mm_free(matrixF_);
		_mm_free(matrixH_);
		matrixF_ = nullptr;
		matrixH_ = nullptr;
		matrixSize_ = 0;
	}
	matrixF_ = static_cast<float*>(_mm_malloc(matrixSize*sizeof(float), 64));
	matrixH_ = static_cast<__half*>(_mm_malloc(matrixSize*sizeof(__half), 64));
	if(!matrixF_ || !matrixH_){
		if(matrixF_){ _mm_free(matrixF_); matrixF_ = nullptr; }
		if(matrixH_){ _mm_free(matrixH_); matrixH_ = nullptr; }
		std::cout << "Allocation error: matrixF_/matrixH_ null (size=" << matrixSize << ")\n";
		return false;
	}
	matrixSize_ = matrixSize;
	return true;
}
static bool EnsureTransposeStorage(const size_t matrixSize){
	if(matrixTSize_ >= matrixSize){ return true; }
	if(matrixTSize_ > 0){
		_mm_free(matrixT_);
		matrixT_ = nullptr;
		matrixTSize_ = 0;
	}
	matrixT_ = static_cast<float*>(_mm_malloc(matrixSize*sizeof(float), 64));
	if(!matrixT_){
		std::cout << "Allocation error: matrixT_ null (size=" << matrixSize << ")\n";
		return false;
	}
	matrixTSize_ = matrixSize;
	return true;
}
static bool EnsureTauStorage(const size_t size){
	if(tauSize_ >= size){ return true; }
	if(tauSize_ > 0){
		_mm_free(tau_);
		tau_ = nullptr;
		tauSize_ = 0;
	}
	tau_ = static_cast<float*>(_mm_malloc(size*sizeof(float), 64));
	if(!tau_){
		std::cout << "Allocation error: tau_ null (k=" << size << ")\n";
		return false;
	}
	tauSize_ = size;
	return true;
}
static bool EnsureWorkStorage(const size_t size){
	if(workSize_ >= size){ return true; }
	if(workSize_ > 0){
		_mm_free(work_);
		work_ = nullptr;
		workSize_ = 0;
	}
	work_ = static_cast<float*>(_mm_malloc(size*sizeof(float), 64));
	if(!work_){
		std::cout << "Allocation error: work_ null (needWork=" << size << ")\n";
		return false;
	}
	workSize_ = size;
	return true;
}
static bool EnsureRandomStream(){
	if(stream_){ return true; }
	const int st = vslNewStream(&stream_, VSL_BRNG_SFMT19937, static_cast<unsigned int>(time(nullptr)));
	if(st != VSL_STATUS_OK){
		std::cout << "VSL error: vslNewStream failed, status=" << st << '\n';
		return false;
	}
	return true;
}
static void FallbackGaussianInit(__half* dWeights, int rows, int cols, WeightInitMethod method){
	if(rows <= 0){ rows = 1; }
	if(cols <= 0){ cols = 1; }
	const size_t elems = static_cast<size_t>(rows)*static_cast<size_t>(cols);
	if(elems == 0){
		std::cout << "FallbackGaussianInit: invalid element count for rows=" << rows << " cols=" << cols << '\n';
		return;
	}
	if(!EnsureMatrixStorage(elems)){ return; }
	if(!EnsureRandomStream()){ return; }
	const int status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream_, static_cast<int>(elems), matrixF_, 0.0f, 1.0f);
	if(status != VSL_STATUS_OK){
		std::cout << "VSL error: vsRngGaussian failed during fallback, status=" << status << '\n';
		return;
	}
	const float fanIn = static_cast<float>(cols);
	float scale = 1.0f;
	if(method == He && fanIn > 0.0f){ scale = std::sqrt(2.0f/fanIn); } else if(method == Xavier && fanIn > 0.0f){ scale = std::sqrt(1.0f/fanIn); }
	for(size_t i = 0; i < elems; ++i){ matrixF_[i] *= scale; }
	FloatToHalfAsm(matrixF_, matrixH_, static_cast<int>(elems));
	const cudaError_t cerr = cudaMemcpy(dWeights, matrixH_, elems*sizeof(__half), cudaMemcpyHostToDevice);
	if(cerr != cudaSuccess){
		std::cout << "CUDA error: cudaMemcpy H2D failed: " << cudaGetErrorString(cerr) << '\n';
	}
}
void OrthogonalInit(__half* dWeights, const int rows, const int cols, WeightInitMethod method){
	if(rows <= 0 || cols <= 0){
		std::cout << "Error: rows and cols must be positive. Got rows=" << rows << " cols=" << cols << '\n';
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	bool transpose_output = false;
	lapack_int m = rows, n = cols;
	if(rows < cols){
		transpose_output = true;
		m = cols;
		n = rows;
	}
	const size_t matrixSize = static_cast<size_t>(m)*static_cast<size_t>(n);
	if(!EnsureMatrixStorage(matrixSize)){
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	if(transpose_output && !EnsureTransposeStorage(matrixSize)){
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	const lapack_int k = std::min(m, n);
	if(!EnsureTauStorage(static_cast<size_t>(k))){
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	if(!EnsureRandomStream()){
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	const int st = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream_, static_cast<int>(matrixSize), matrixF_, 0.0f, 1.0f);
	if(st != VSL_STATUS_OK){
		std::cout << "VSL error: vsRngGaussian failed, status=" << st << '\n';
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	float workq = 0.0f;
	lapack_int info = LAPACKE_sgeqrf_work(LAPACK_COL_MAJOR, m, n, matrixF_, m, tau_, &workq, -1);
	if(info != 0){
		std::cout << "LAPACK error: sgeqrf(lwork=-1) info=" << info << '\n';
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	const lapack_int lworkGeqrf = std::max<lapack_int>(1, static_cast<lapack_int>(workq));
	workq = 0.0f;
	info = LAPACKE_sorgqr_work(LAPACK_COL_MAJOR, m, n, k, matrixF_, m, tau_, &workq, -1);
	if(info != 0){
		std::cout << "LAPACK error: sorgqr(lwork=-1) info=" << info << '\n';
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	const lapack_int lworkOrgqr = std::max<lapack_int>(1, static_cast<lapack_int>(workq));
	const size_t needWork = static_cast<size_t>(std::max(lworkGeqrf, lworkOrgqr));
	if(!EnsureWorkStorage(needWork)){
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	info = LAPACKE_sgeqrf_work(LAPACK_COL_MAJOR, m, n, matrixF_, m, tau_, work_, static_cast<lapack_int>(workSize_));
	if(info != 0){
		std::cout << "LAPACK error: sgeqrf info=" << info << '\n';
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	info = LAPACKE_sorgqr_work(LAPACK_COL_MAJOR, m, n, k, matrixF_, m, tau_, work_, static_cast<lapack_int>(workSize_));
	if(info != 0){
		std::cout << "LAPACK error: sorgqr info=" << info << '\n';
		FallbackGaussianInit(dWeights, rows, cols, method);
		return;
	}
	const int fanIn = cols;
	float scale = 1.0f;
	if(method == He && fanIn > 0){ scale = std::sqrt(2.0f/static_cast<float>(fanIn)); } else if(method == Xavier && fanIn > 0){ scale = std::sqrt(1.0f/static_cast<float>(fanIn)); }
	float* outF = matrixF_;
	if(transpose_output){
		mkl_somatcopy('C', 'T', m, n, 1.0f, matrixF_, m, matrixT_, n);
		outF = matrixT_;
	}
	const size_t elems = static_cast<size_t>(rows)*static_cast<size_t>(cols);
	for(size_t i = 0; i < elems; ++i){ outF[i] *= scale; }
	FloatToHalfAsm(outF, matrixH_, static_cast<int>(elems));
	const cudaError_t cerr = cudaMemcpy(dWeights, matrixH_, elems*sizeof(__half), cudaMemcpyHostToDevice);
	if(cerr != cudaSuccess){
		std::cout << "CUDA error: cudaMemcpy H2D failed: " << cudaGetErrorString(cerr) << '\n';
	}
}