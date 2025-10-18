#include "common.h"
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <windows.h>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "\ncuDNN error: " << cudnnGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuDNN error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudnnGetErrorString(status)); \
    } \
}
#define checkCUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "\nCUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(status)); \
    } \
}
DebugOptions gDebugOptions;

static bool GetEnvFlag(const char* name){
        const char* value = std::getenv(name);
        if(!value) return false;
        if(value[0] == '\0') return false;
        switch(value[0]){
        case '1':
        case 't':
        case 'T':
        case 'y':
        case 'Y':
                return true;
        default:
                break;
        }
        return false;
}

static int GetEnvInt(const char* name, int defaultValue){
        const char* value = std::getenv(name);
        if(!value) return defaultValue;
        return std::max(0, std::atoi(value));
}

static float GetEnvFloat(const char* name, float defaultValue){
        const char* value = std::getenv(name);
        if(!value) return defaultValue;
        return std::max(0.0f, static_cast<float>(atof(value)));
}

void InitializeDebugOptionsFromEnv(){
        gDebugOptions.logAdamUpdateStats = GetEnvFlag("AI_DEBUG_LOG_ADAM");
        gDebugOptions.gradientStripeTest = GetEnvFlag("AI_DEBUG_GRAD_STRIPE");
        gDebugOptions.gradientStripeLogitIndex = GetEnvInt("AI_DEBUG_GRAD_LOGIT", gDebugOptions.gradientStripeLogitIndex);
        gDebugOptions.gradientStripeSampleIndex = GetEnvInt("AI_DEBUG_GRAD_SAMPLE", gDebugOptions.gradientStripeSampleIndex);
        gDebugOptions.useReferenceAttention = GetEnvFlag("AI_DEBUG_REF_ATTENTION");
        gDebugOptions.syncAfterOptimizerStep = GetEnvFlag("AI_DEBUG_SYNC_AFTER_OPT");
        gDebugOptions.logPatchNorms = GetEnvFlag("AI_DEBUG_PATCH_NORM");
        gDebugOptions.useBceWithLogits = GetEnvFlag("AI_DEBUG_BCE_WITH_LOGITS");
        if(const char* lossScale = std::getenv("AI_DEBUG_LOSS_SCALE")){
                gDebugOptions.staticLossScale = std::max(0.0f, static_cast<float>(atof(lossScale)));
                gDebugOptions.applyStaticLossScale = gDebugOptions.staticLossScale > 0.0f;
        }
}

unsigned char keyMap[] = {
	0x11, // W
	0x1E, // A
	0x1F, // S
	0x20, // D
	0x39, // Space
	0x1D, // Ctrl
	0x10, // Q
	0x13, // R
	0x12, // E
	0x02, // 1
	0x03, // 2
	0x0B, // Mouse button 1
	0x0C, // Mouse button 2
	0x0D  // Mouse button 3
};
void HalfToFloatAsm(float* dst, __half* src, int count){
	__asm {
		mov rsi, src
		mov rdi, dst
		mov ecx, count
		mov eax, ecx
		shr ecx, 7
		test ecx, ecx
		jz remainder

		loop_start :
		movdqa xmm0, [rsi]
			movdqa xmm1, [rsi+16]
			movdqa xmm2, [rsi+32]
			movdqa xmm3, [rsi+48]
			movdqa xmm4, [rsi+64]
			movdqa xmm5, [rsi+80]
			movdqa xmm6, [rsi+96]
			movdqa xmm7, [rsi+112]
			movdqa xmm8, [rsi+128]
			movdqa xmm9, [rsi+144]
			movdqa xmm10, [rsi+160]
			movdqa xmm11, [rsi+176]
			movdqa xmm12, [rsi+192]
			movdqa xmm13, [rsi+208]
			movdqa xmm14, [rsi+224]
			movdqa xmm15, [rsi+240]
			vcvtph2ps ymm0, xmm0
			vcvtph2ps ymm1, xmm1
			vcvtph2ps ymm2, xmm2
			vcvtph2ps ymm3, xmm3
			vcvtph2ps ymm4, xmm4
			vcvtph2ps ymm5, xmm5
			vcvtph2ps ymm6, xmm6
			vcvtph2ps ymm7, xmm7
			vcvtph2ps ymm8, xmm8
			vcvtph2ps ymm9, xmm9
			vcvtph2ps ymm10, xmm10
			vcvtph2ps ymm11, xmm11
			vcvtph2ps ymm12, xmm12
			vcvtph2ps ymm13, xmm13
			vcvtph2ps ymm14, xmm14
			vcvtph2ps ymm15, xmm15
			vmovdqa[rdi], ymm0
			vmovdqa[rdi+32], ymm1
			vmovdqa[rdi+64], ymm2
			vmovdqa[rdi+96], ymm3
			vmovdqa[rdi+128], ymm4
			vmovdqa[rdi+160], ymm5
			vmovdqa[rdi+192], ymm6
			vmovdqa[rdi+224], ymm7
			vmovdqa[rdi+256], ymm8
			vmovdqa[rdi+288], ymm9
			vmovdqa[rdi+320], ymm10
			vmovdqa[rdi+352], ymm11
			vmovdqa[rdi+384], ymm12
			vmovdqa[rdi+416], ymm13
			vmovdqa[rdi+448], ymm14
			vmovdqa[rdi+480], ymm15
			add rsi, 256
			add rdi, 512
			dec ecx
			jnz loop_start

			remainder :
		and eax, 127
			jz done

			remainder_loop :
		movdqa xmm0, [rsi]
			vcvtph2ps ymm0, xmm0
			vmovdqa[rdi], ymm0
			add rsi, 16
			add rdi, 32
			sub eax, 8
			cmp eax, 8
			jge remainder_loop
			test eax, eax
			jz done

			final_elements :
		movsd xmm0, [rsi]
			vcvtph2ps xmm0, xmm0
			movsd[rdi], xmm0
			add rsi, 4
			add rdi, 8
			sub eax, 2
			jg final_elements

			done :
		vzeroupper
	}
}
void FloatToHalfAsm(float* src, __half* dst, int count){
	__asm {
		mov rsi, src
		mov rdi, dst
		mov ecx, count
		mov eax, ecx
		shr ecx, 7
		test ecx, ecx
		jz remainder

		loop_start :
		vmovdqa ymm0, [rsi]
			vmovdqa ymm1, [rsi+32]
			vmovdqa ymm2, [rsi+64]
			vmovdqa ymm3, [rsi+96]
			vmovdqa ymm4, [rsi+128]
			vmovdqa ymm5, [rsi+160]
			vmovdqa ymm6, [rsi+192]
			vmovdqa ymm7, [rsi+224]
			vmovdqa ymm8, [rsi+256]
			vmovdqa ymm9, [rsi+288]
			vmovdqa ymm10, [rsi+320]
			vmovdqa ymm11, [rsi+352]
			vmovdqa ymm12, [rsi+384]
			vmovdqa ymm13, [rsi+416]
			vmovdqa ymm14, [rsi+448]
			vmovdqa ymm15, [rsi+480]
			vcvtps2ph xmm0, ymm0, 0
			vcvtps2ph xmm1, ymm1, 0
			vcvtps2ph xmm2, ymm2, 0
			vcvtps2ph xmm3, ymm3, 0
			vcvtps2ph xmm4, ymm4, 0
			vcvtps2ph xmm5, ymm5, 0
			vcvtps2ph xmm6, ymm6, 0
			vcvtps2ph xmm7, ymm7, 0
			vcvtps2ph xmm8, ymm8, 0
			vcvtps2ph xmm9, ymm9, 0
			vcvtps2ph xmm10, ymm10, 0
			vcvtps2ph xmm11, ymm11, 0
			vcvtps2ph xmm12, ymm12, 0
			vcvtps2ph xmm13, ymm13, 0
			vcvtps2ph xmm14, ymm14, 0
			vcvtps2ph xmm15, ymm15, 0
			movdqa[rdi], xmm0
			movdqa[rdi+16], xmm1
			movdqa[rdi+32], xmm2
			movdqa[rdi+48], xmm3
			movdqa[rdi+64], xmm4
			movdqa[rdi+80], xmm5
			movdqa[rdi+96], xmm6
			movdqa[rdi+112], xmm7
			movdqa[rdi+128], xmm8
			movdqa[rdi+144], xmm9
			movdqa[rdi+160], xmm10
			movdqa[rdi+176], xmm11
			movdqa[rdi+192], xmm12
			movdqa[rdi+208], xmm13
			movdqa[rdi+224], xmm14
			movdqa[rdi+240], xmm15
			add rsi, 512
			add rdi, 256
			dec ecx
			jnz loop_start

			remainder :
		and eax, 127
			jz done

			remainder_loop :
		vmovdqa ymm0, [rsi]
			vcvtps2ph xmm0, ymm0, 0
			movdqa[rdi], xmm0
			add rsi, 32
			add rdi, 16
			sub eax, 8
			cmp eax, 8
			jge remainder_loop
			test eax, eax
			jz done

			final_elements :
		movsd xmm0, [rsi]
			vcvtps2ph xmm0, xmm0, 0
			movsd[rdi], xmm0
			add rsi, 8
			add rdi, 4
			sub eax, 2
			jg final_elements

			done :
		vzeroupper
	}
}
size_t printDataCount = 0;
__half* hData = nullptr;
float* fData = nullptr;
void PrintDataHalfDevice(const __half* data, const size_t size, const char* label){
	if(printDataCount < size){
		if(hData)
			_mm_free(hData);
		if(fData)
			_mm_free(fData);
		hData = static_cast<__half*>(_mm_malloc(size*sizeof(__half), 64));
		fData = static_cast<float*>(_mm_malloc(size*sizeof(float), 64));
		printDataCount = size;
	}
	checkCUDA(cudaMemcpy(hData, data, size*sizeof(__half), cudaMemcpyDeviceToHost));
	HalfToFloatAsm(fData, hData, size);
	std::ostringstream output;
	output << label << ":\n";
	for(size_t i = 0; i < size; ++i){ output << fData[i] << " "; }
	output << "\n";
	std::cout << output.str();
}
void PrintDataFloatDevice(const float* data, const size_t size, const char* label){
	std::vector<float> hData(size);
	checkCUDA(cudaMemcpy(hData.data(), data, size*sizeof(float), cudaMemcpyDeviceToHost));
	std::ostringstream output;
	output << label << ":\n";
	for(size_t i = 0; i < hData.size(); ++i){ output << hData[i] << " "; }
	output << "\n";
	std::cout << output.str();
}
void PrintDataFloatHost(const float* data, const size_t size, const char* label){
	std::ostringstream output;
	output << label << ":\n";
	for(size_t i = 0; i < size; ++i){ output << data[i] << " "; }
	output << "\n";
	std::cout << output.str();
}
void PrintDataCharHost(const unsigned char* data, const size_t size, const char* label){
	std::ostringstream output;
	output << label << ":\n";
	for(size_t i = 0; i < size; ++i){ output << data[i] << " "; }
	output << "\n";
	std::cout << output.str();
}
void SummarizeHalfDevice(const __half* data, const size_t size, const char* label){
	if(printDataCount < size){
		if(hData)
			_mm_free(hData);
		if(fData)
			_mm_free(fData);
		hData = static_cast<__half*>(_mm_malloc(size*sizeof(__half), 64));
		fData = static_cast<float*>(_mm_malloc(size*sizeof(float), 64));
		printDataCount = size;
	}
	checkCUDA(cudaMemcpy(hData, data, size*sizeof(__half), cudaMemcpyDeviceToHost));
	HalfToFloatAsm(fData, hData, size);
	float minVal = std::numeric_limits<float>::infinity();
	float maxVal = -std::numeric_limits<float>::infinity();
	bool hasNaN = false;
	bool hasInf = false;
	for(size_t i = 0; i < size; ++i){
		const float v = fData[i];
		if(std::isnan(v)){ hasNaN = true; continue; }
		if(std::isinf(v)){ hasInf = true; continue; }
		if(v < minVal) minVal = v;
		if(v > maxVal) maxVal = v;
	}
	if(minVal == std::numeric_limits<float>::infinity()) minVal = 0.0f;
	if(maxVal == -std::numeric_limits<float>::infinity()) maxVal = 0.0f;
	std::ostringstream output;
	output << label << " summary: min=" << minVal << " max=" << maxVal
		<< " NaN=" << (hasNaN ? "true" : "false")
		<< " Inf=" << (hasInf ? "true" : "false") << "\n";
	std::cout << output.str();
}
void ClearScreen(char fill){
	const COORD tl = {0, 0};
	CONSOLE_SCREEN_BUFFER_INFO s;
	const HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(console, &s);
	DWORD written;
	const DWORD cells = s.dwSize.X*s.dwSize.Y;
	FillConsoleOutputCharacter(console, fill, cells, tl, &written);
	FillConsoleOutputAttribute(console, s.wAttributes, cells, tl, &written);
	SetConsoleCursorPosition(console, tl);
}
std::vector<std::string> trainDataFiles = {"E:\\TrainingData\\trainingData0.bin", "E:\\TrainingData\\trainingData1.bin"};
std::string valDataFile = "E:\\TrainingData\\validationData.bin";
std::string trainDataOutFileName = "E:\\TrainingData.bin";
std::string ckptFileName = "E:\\AIGamer.ckpt";
std::string optFileName = "E:\\AIGamer.opt";
std::vector<RecordIndex> trainRecordIndices;
std::vector<RecordIndex> valRecordIndices;
std::mutex recordIndicesMutex;
ThreadPool threadPool(8);
void ReportStreamState(std::ifstream& file){
	if(file.eof()){ std::cerr << "End of file reached prematurely\n"; } else if(file.fail()){ std::cerr << "Logical error on I/O operation\n"; } else if(file.bad()){ std::cerr << "Read/writing error on I/O operation\n"; } else{
		std::cerr << "Unknown error occurred\n";
	}
	std::cerr << "Current stream position: " << file.tellg() << "\n";
}
static std::ifstream& GetThreadFile(const std::string& fileName){
	using StreamPtr = std::unique_ptr<std::ifstream>;
	static thread_local std::unordered_map<std::string, StreamPtr> fileMap;
	auto it = fileMap.find(fileName);
	if(it == fileMap.end() || !it->second || !it->second->is_open()){
		auto stream = std::make_unique<std::ifstream>(fileName, std::ios::binary | std::ios::in);
		if(!stream->is_open()){
			std::cerr << "Failed to open file: " << fileName << "\n";
			throw std::runtime_error("Failed to open file");
		}
		const auto emplaceResult = fileMap.emplace(fileName, std::move(stream));
		it = emplaceResult.first;
	}
	it->second->clear();
	return *(it->second);
}
void LoadBatch(StateBatch* batch, const int batchSize, const int stateSize, const bool validation){
	const std::vector<RecordIndex>* recordIndices = validation ? &valRecordIndices : &trainRecordIndices;
	if(recordIndices->size() < batchSize){
		std::cerr << "Not enough records to fill the batch\n";
		return;
	}
	for(size_t i = 0; i < batchSize; ++i){
		threadPool.Enqueue([i, batch, stateSize, recordIndices]{
			const std::uniform_int_distribution<size_t> dist(0, recordIndices->size() - 1);
			const size_t randomIndex = dist(threadPool.GetThreadGenerator());
			const RecordIndex record = (*recordIndices)[randomIndex];
			try{
				auto& file = GetThreadFile(*record.fileName);
				file.seekg(record.position);
				if(file.fail()){
					std::cerr << "Failed to seek to position: " << record.position << " in file: " << *record.fileName << "\n";
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->inputStates[i]), sizeof(InputState))){
					std::cerr << "Failed to read input states at index " << i << " from file: " << *record.fileName << "\n";
					return;
				}
				if(!file.read(reinterpret_cast<char*>(batch->stateData + i*stateSize), stateSize)){ std::cerr << "Failed to read stateData at index " << i << " from file: " << *record.fileName << "\n"; }
			} catch(const std::exception&){ }
		});
	}
}
void LoadBatchLSTM(StateBatch* batch, const int batchSize, int seqLength, const int stateSize, const bool validation){
	const std::vector<RecordIndex>* recordIndices = validation ? &valRecordIndices : &trainRecordIndices;
	if(recordIndices->size() < batchSize*seqLength){
		std::cerr << "Not enough records in the index to load the batch\n";
		return;
	}
	for(size_t i = 0; i < batchSize; ++i){
		threadPool.Enqueue([i, batch, seqLength, stateSize, recordIndices]() mutable{
			const std::uniform_int_distribution<size_t> dist(0, recordIndices->size() - seqLength);
			const size_t randomIndex = dist(threadPool.GetThreadGenerator());
			for(int t = 0; t < seqLength; ++t){
				const size_t recordIndex = randomIndex + t;
				const RecordIndex record = (*recordIndices)[recordIndex];
				try{
					auto& file = GetThreadFile(*record.fileName);
					file.seekg(record.position);
					if(file.fail()){
						std::cerr << "Failed to seek to position:" << record.position << " in file: " << *record.fileName << "\n";
						return;
					}
					const auto index = i*seqLength + t;
					if(!file.read(reinterpret_cast<char*>(&batch->inputStates[index]), sizeof(InputState))){
						std::cerr << "Failed to read input states for sequence " << index << " from file: " << *record.fileName << "\n";
						return;
					}
					if(!file.read(reinterpret_cast<char*>(batch->stateData + index*stateSize), stateSize)){
						std::cerr << "Failed to read stateData at index " << index << " from file: " << *record.fileName << "\n";
						return;
					}
				} catch(const std::exception&){ return; }
			}
		});
	}
}
void LoadBatchFromVector(const std::vector<StateSingle*>& states, StateBatch* batch, int batchSize, const int stateSize){
	if(states.size() < batchSize){
		std::cerr << "Not enough RecordState instances to fill the batch\n";
		return;
	}
	std::uniform_int_distribution<size_t> dist(0, states.size() - 1);
	for(size_t i = 0; i < batchSize; ++i){
		threadPool.Enqueue([i, batch, stateSize, &states, dist]() mutable{
			const size_t randomIndex = dist(threadPool.GetThreadGenerator());
			const auto& record = states[randomIndex];
			batch->inputStates[i] = record->inputState;
			if(batch->stateData && record->stateData){ std::memcpy(batch->stateData + i*stateSize, record->stateData, stateSize); } else{ std::cerr << "Invalid stateData pointer for RecordState at index " << randomIndex << "\n"; }
		});
	}
}
ConvolutionAlgorithms GetConvolutionAlgorithms(cudnnHandle_t cudnnHandle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, bool isTraining){
	ConvolutionAlgorithms algorithms;
	algorithms.workspaceSize = 0;
	// Forward algorithm
	cudnnConvolutionFwdAlgoPerf_t fwdAlgoPerf[10];
	int returnedAlgoCount;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7( cudnnHandle, xDesc, wDesc, convDesc, yDesc, 10, &returnedAlgoCount, fwdAlgoPerf ));
	algorithms.fwdAlgo = fwdAlgoPerf[0].algo;
	algorithms.workspaceSize = max(algorithms.workspaceSize, fwdAlgoPerf[0].memory);
	if(isTraining){
		// Backward data algorithm
		cudnnConvolutionBwdDataAlgoPerf_t bwdDataAlgoPerf[10];
		checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7( cudnnHandle, wDesc, yDesc, convDesc, xDesc, 10, &returnedAlgoCount, bwdDataAlgoPerf ));
		algorithms.bwdDataAlgo = bwdDataAlgoPerf[0].algo;
		algorithms.workspaceSize = max(algorithms.workspaceSize, bwdDataAlgoPerf[0].memory);
		// Backward filter algorithm
		cudnnConvolutionBwdFilterAlgoPerf_t bwdFilterAlgoPerf[10];
		checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7( cudnnHandle, xDesc, yDesc, convDesc, wDesc, 10, &returnedAlgoCount, bwdFilterAlgoPerf ));
		algorithms.bwdFilterAlgo = bwdFilterAlgoPerf[0].algo;
		algorithms.workspaceSize = max(algorithms.workspaceSize, bwdFilterAlgoPerf[0].memory);
	} else{
		algorithms.bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		algorithms.bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
	}
	return algorithms;
}
#include <mkl_lapacke.h>
#include <mkl_vsl.h>
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
void InitializeStream(){ if(stream_ == nullptr){ vslNewStream(&stream_, VSL_BRNG_SFMT19937, time(nullptr)); } }
void OrthogonalInit(__half* output, const int rows, const int cols, WeightInitMethod method){
	bool transpose = false;
	int m = rows;
	int n = cols;
	int fan_in = cols;
	int fan_out = rows;
	if(rows < cols){
		transpose = true;
		m = cols;
		n = rows;
		fan_in = rows;
		fan_out = cols;
	}
	const size_t matrixSize = static_cast<size_t>(m)*n;
	if(matrixSize_ < matrixSize){
		if(matrixSize_ > 0){
			_mm_free(matrixF_);
			_mm_free(matrixH_);
		}
		matrixF_ = static_cast<float*>(_mm_malloc(matrixSize*sizeof(float), 64));
		matrixH_ = static_cast<__half*>(_mm_malloc(matrixSize*sizeof(__half), 64));
		matrixSize_ = matrixSize;
	}
	if(transpose && matrixTSize_ < matrixSize){
		if(matrixTSize_ > 0){ _mm_free(matrixT_); }
		matrixT_ = static_cast<float*>(_mm_malloc(matrixSize*sizeof(float), 64));
		matrixTSize_ = matrixSize;
	}
	const size_t tauSize = min(m, n);
	if(tauSize_ < tauSize){
		if(tauSize_ > 0){ _mm_free(tau_); }
		tau_ = static_cast<float*>(_mm_malloc(tauSize*sizeof(float), 64));
		tauSize_ = tauSize;
	}
	InitializeStream();
	vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream_, matrixSize, matrixF_, 0.0f, 1.0f);
	float workQueryQRF, workQueryORGQR;
	lapack_int info = LAPACKE_sgeqrf_work(LAPACK_COL_MAJOR, m, n, matrixF_, m, tau_, &workQueryQRF, -1);
	if(info != 0){
		return;
	}
	const auto tempMatrix = static_cast<float*>(_mm_malloc(matrixSize*sizeof(float), 64));
	const auto tempTau = static_cast<float*>(_mm_malloc(tauSize*sizeof(float), 64));
	memcpy(tempMatrix, matrixF_, matrixSize*sizeof(float));
	const auto tempWork = static_cast<float*>(_mm_malloc(static_cast<size_t>(workQueryQRF)*sizeof(float), 64));
	LAPACKE_sgeqrf_work(LAPACK_COL_MAJOR, m, n, tempMatrix, m, tempTau, tempWork, static_cast<lapack_int>(workQueryQRF));
	info = LAPACKE_sorgqr_work(LAPACK_COL_MAJOR, m, n, tauSize, tempMatrix, m, tempTau, &workQueryORGQR, -1);
	if(info != 0){
		_mm_free(tempMatrix);
		_mm_free(tempTau);
		_mm_free(tempWork);
		return;
	}
	_mm_free(tempMatrix);
	_mm_free(tempTau);
	_mm_free(tempWork);
	const size_t optimalWorkSize = max(static_cast<size_t>(workQueryQRF), static_cast<size_t>(workQueryORGQR));
	if(workSize_ < optimalWorkSize){
		if(workSize_ > 0){ _mm_free(work_); }
		work_ = static_cast<float*>(_mm_malloc(optimalWorkSize*sizeof(float), 64));
		workSize_ = optimalWorkSize;
	}
	info = LAPACKE_sgeqrf_work(LAPACK_COL_MAJOR, m, n, matrixF_, m, tau_, work_, workSize_);
	if(info != 0){
		return;
	}
	info = LAPACKE_sorgqr_work(LAPACK_COL_MAJOR, m, n, tauSize, matrixF_, m, tau_, work_, workSize_);
	if(info != 0){
		return;
	}
	float* outF = matrixF_;
	if(transpose){
		for(int r = 0; r < rows; ++r){
			for(int c = 0; c < cols; ++c){
				matrixT_[r*cols + c] = matrixF_[c*rows + r];
			}
		}
		outF = matrixT_;
	}
	auto scale = 1.0f;
	if(method == He) scale = sqrtf(2.0f / fan_in);
	if(method == Xavier) scale = sqrtf(1.0f / fan_in);
	for(int i = 0; i < rows*cols; ++i){
		outF[i] *= scale;
	}
	FloatToHalfAsm(outF, matrixH_, rows*cols);
	checkCUDA(cudaMemcpy(output, matrixH_, rows*cols*sizeof(__half), cudaMemcpyHostToDevice));
}