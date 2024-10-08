#include "common.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <windows.h>
const char* cublasGetErrorString(cublasStatus_t status){
	switch(status){
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
		default:
			return "Unknown cuBLAS error";
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
	11,  // Mouse button 1
	12,  // Mouse button 2
	13   // Mouse button 3
};
int ConvertSmVer2Cores(int major, int minor){
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct{
		int SM; // 0xMm (hexadecimal notation), M = SM Major version and m = SM minor version
		int Cores;
	} sSMtoCores;
	const sSMtoCores nGpuArchCoresPerSM[] = {
		{0x10, 8}, // Tesla Generation (SM 1.0) G80 class
		{0x11, 8}, // Tesla Generation (SM 1.1) G8x class
		{0x12, 8}, // Tesla Generation (SM 1.2) G9x class
		{0x13, 8}, // Tesla Generation (SM 1.3) GT200 class
		{0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
		{0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
		{0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
		{0x60, 64}, // Pascal Generation (SM 6.0) GP100 class
		{0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
		{0x70, 64}, // Volta Generation (SM 7.0) GV100 class
		{0x72, 64}, // Volta Generation (SM 7.2) GV10B class
		{0x75, 64}, // Turing Generation (SM 7.5) TU10x class
		{0x80, 64}, // Ampere Generation (SM 8.0) GA100 class
		{0x86, 128}, // Ampere Generation (SM 8.6) GA10x class
		{0x87, 128}, // Ampere Generation (SM 8.7) GA10x class
		{0x89, 128}, // Ada Lovelace Generation (SM 8.9) AD10x class
	};
	int index = 0;
	while(nGpuArchCoresPerSM[index].SM != -1){
		if(nGpuArchCoresPerSM[index].SM == (major << 4) + minor){
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	// If we don't find the values, we default to the last known architecture to run properly
	printf("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}
void H2F128Asm(float* dst, __half* src, int numElements){
	int num_iterations = numElements / 128;
	__asm {
		mov rsi, src
		mov rdi, dst
		mov ecx, num_iterations
		loop_start:
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
	}
}
void PrintDataHalf2(const __half* data, const size_t size, const char* label){
	std::vector<__half> h_data(size);
	checkCUDA(cudaMemcpy(h_data.data(), data, size*sizeof(__half), cudaMemcpyDeviceToHost));
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < h_data.size(); ++i){ std::cout << __half2float(h_data[i]) << " "; }
	std::cout << "\r\n";
}
void PrintDataHalf(const __half* data, const size_t size, const char* label){
	const size_t truncatedSize = size / 128*128;
	if(truncatedSize == 0){
		PrintDataHalf2(data, size, label);
		return;
	}
	const auto hData = static_cast<__half*>(_mm_malloc(truncatedSize*sizeof(__half), 32));
	const auto fData = static_cast<float*>(_mm_malloc(truncatedSize*sizeof(float), 32));
	checkCUDA(cudaMemcpy(hData, data, truncatedSize*sizeof(__half), cudaMemcpyDeviceToHost));
	H2F128Asm(fData, hData, truncatedSize);
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < truncatedSize; ++i){
		std::cout << fData[i] << " ";
	}
	std::cout << "\r\n";
	_mm_free(hData);
	_mm_free(fData);
}
void PrintDataFloat(const float* data, const size_t size, const char* label){
	std::vector<float> h_data(size);
	checkCUDA(cudaMemcpy(h_data.data(), data, size*sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < h_data.size(); ++i){ std::cout << h_data[i] << " "; }
	std::cout << "\r\n";
}
void PrintDataFloatHost(const float* data, const size_t size, const char* label){
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < size; ++i){ std::cout << data[i] << " "; }
	std::cout << "\r\n";
}
void PrintDataCharHost(const unsigned char* data, const size_t size, const char* label){
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < size; ++i){ std::cout << data[i] << " "; }
	std::cout << "\r\n";
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
std::vector<std::string> trainDataInFiles = {"E:\\TrainingData\\training_data1.bin", "E:\\TrainingData\\training_data2.bin"};
std::unordered_map<std::string, std::vector<std::streampos>> fileRecordIndex;
std::size_t stateSize_;
ThreadPool threadPool(8);
void ReportStreamState(std::ifstream& file){
	if(file.eof()){ std::cerr<<"End of file reached prematurely.\r\n"; } else if(file.fail()){ std::cerr<<"Logical error on I/O operation.\r\n"; } else if(file.bad()){ std::cerr<<"Read/writing error on I/O operation.\r\n"; } else{
		std::cerr<<"Unknown error occurred.\r\n";
	}
	std::cerr<<"Current stream position: "<<file.tellg()<<"\r\n";
}
void LoadBatch(StateBatch* batch, int batchSize){
	for(size_t i = 0; i<batchSize; ++i){
		threadPool.Enqueue([i, batch]{
			auto& gen = threadPool.GetThreadGenerator();
			const std::uniform_int_distribution<> fileDis(0, trainDataInFiles.size() - 1);
			const std::string selectedFile = trainDataInFiles[fileDis(gen)];
			const auto recordIndexIt = fileRecordIndex.find(selectedFile);
			if(recordIndexIt==fileRecordIndex.end()){
				std::cerr<<"No records for file: "<<selectedFile<<"\r\n";
				return;
			}
			std::ifstream file(selectedFile, std::ios::binary|std::ios::in);
			if(!file.is_open()){
				std::cerr<<"Failed to open training data file: "<<selectedFile<<"\r\n";
				return;
			}
			const std::uniform_int_distribution<> recordDis(0, recordIndexIt->second.size() - 1);
			const size_t recordIndex = recordDis(gen);
			if(recordIndex>=recordIndexIt->second.size()){
				std::cerr<<"Record index out of bounds in file: "<<selectedFile<<"\r\n";
				return;
			}
			file.seekg(recordIndexIt->second[recordIndex]);
			if(file.fail()){
				std::cerr<<"seekg failed for recordIndex: "<<recordIndex<<" in file: "<<selectedFile<<" at position: "<<recordIndexIt->second[recordIndex]<<"\r\n";
				return;
			}
			if(!file.read(reinterpret_cast<char*>(&batch->keyStates[i]), sizeof(unsigned short))){
				std::cerr<<"Failed to read keyStates at index "<<i<<" from file: "<<selectedFile<<"\r\n";
				ReportStreamState(file);
				return;
			}
			if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaX[i]), sizeof(int))){
				std::cerr<<"Failed to read mouseDeltaX at index "<<i<<" from file: "<<selectedFile<<"\r\n";
				ReportStreamState(file);
				return;
			}
			if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaY[i]), sizeof(int))){
				std::cerr<<"Failed to read mouseDeltaY at index "<<i<<" from file: "<<selectedFile<<"\r\n";
				ReportStreamState(file);
				return;
			}
			if(!file.read(reinterpret_cast<char*>(batch->stateData+i*stateSize_), stateSize_)){
				std::cerr<<"Failed to read stateData at index "<<i<<" from file: "<<selectedFile<<"\r\n";
				ReportStreamState(file);
			}
		});
	}
}
void LoadBatch3D(StateBatch* batch, int seqLength, int batchSize){
	for(size_t n = 0; n<batchSize; ++n){
		threadPool.Enqueue([n, batch, seqLength](){
			auto& gen = threadPool.GetThreadGenerator();
			const std::uniform_int_distribution<> fileDis(0, trainDataInFiles.size()-1);
			const std::string selectedFile = trainDataInFiles[fileDis(gen)];
			const auto recordIndexIt = fileRecordIndex.find(selectedFile);
			if(recordIndexIt==fileRecordIndex.end()){
				std::cerr<<"No records for file: "<<selectedFile<<"\r\n";
				return;
			}
			std::ifstream file(selectedFile, std::ios::binary|std::ios::in);
			if(!file.is_open()){
				std::cerr<<"Failed to open training data file: "<<selectedFile<<"\r\n";
				return;
			}
			const std::uniform_int_distribution<> recordDis(0, recordIndexIt->second.size()-seqLength);
			const size_t recordIndex = recordDis(gen);
			if(recordIndex+seqLength>=recordIndexIt->second.size()){
				std::cerr<<"Record index out of bounds in file: "<<selectedFile<<"\r\n";
				return;
			}
			for(int d = 0; d<seqLength; ++d){
				file.seekg(recordIndexIt->second[recordIndex+d]);
				if(file.fail()){
					std::cerr<<"seekg failed for recordIndex: "<<recordIndex+d<<" in file: "<<selectedFile<<" at position: "<<recordIndexIt->second[recordIndex+d]<<"\r\n";
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->keyStates[n]), sizeof(unsigned short))){
					std::cerr<<"Failed to read keyStates at index "<<n<<" from file: "<<selectedFile<<"\r\n";
					ReportStreamState(file);
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaX[n]), sizeof(int))){
					std::cerr<<"Failed to read mouseDeltaX at index "<<n<<" from file: "<<selectedFile<<"\r\n";
					ReportStreamState(file);
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaY[n]), sizeof(int))){
					std::cerr<<"Failed to read mouseDeltaY at index "<<n<<" from file: "<<selectedFile<<"\r\n";
					ReportStreamState(file);
					return;
				}
				const auto stateCSize = stateSize_/3;
				for(int c = 0; c<3; ++c){
					const auto dstIndex = seqLength*stateSize_*n + stateCSize*(c + 3*d);
					if(!file.read(reinterpret_cast<char*>(batch->stateData+dstIndex), stateCSize)){
						std::cerr<<"Failed to read stateData at index "<<dstIndex<<" from file: "<<selectedFile<<"\r\n";
						ReportStreamState(file);
						return;
					}
				}
			}
		});
	}
}
void LoadBatchLSTM(StateBatch* batch, int seqLength, int batchSize){
	for(size_t i = 0; i<batchSize; ++i){
		threadPool.Enqueue([i, batch, seqLength, batchSize](){
			auto& gen = threadPool.GetThreadGenerator();
			const std::uniform_int_distribution<> fileDis(0, trainDataInFiles.size()-1);
			const std::string selectedFile = trainDataInFiles[fileDis(gen)];
			const auto recordIndexIt = fileRecordIndex.find(selectedFile);
			if(recordIndexIt==fileRecordIndex.end()){
				std::cerr<<"No records for file: "<<selectedFile<<"\r\n";
				return;
			}
			std::ifstream file(selectedFile, std::ios::binary|std::ios::in);
			if(!file.is_open()){
				std::cerr<<"Failed to open training data file: "<<selectedFile<<"\r\n";
				return;
			}
			const std::uniform_int_distribution<> recordDis(0, recordIndexIt->second.size()-seqLength);
			const size_t recordIndex = recordDis(gen);
			if(recordIndex+seqLength>=recordIndexIt->second.size()){
				std::cerr<<"Record index out of bounds in file: "<<selectedFile<<"\r\n";
				return;
			}
			for(int j = 0; j<seqLength; ++j){
				file.seekg(recordIndexIt->second[recordIndex+j]);
				if(file.fail()){
					std::cerr<<"seekg failed for recordIndex: "<<recordIndex+j<<" in file: "<<selectedFile<<" at position: "<<recordIndexIt->second[recordIndex+j]<<"\r\n";
					return;
				}
				const auto index = j*batchSize+i;
				if(j == seqLength - 1){
					if(!file.read(reinterpret_cast<char*>(&batch->keyStates[i]), sizeof(unsigned short))){
						std::cerr<<"Failed to read keyStates for sequence "<<i<<" from file: "<<selectedFile<<"\r\n";
						ReportStreamState(file);
						return;
					}
					if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaX[i]), sizeof(int))){
						std::cerr<<"Failed to read mouseDeltaX for sequence "<<i<<" from file: "<<selectedFile<<"\r\n";
						ReportStreamState(file);
						return;
					}
					if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaY[i]), sizeof(int))){
						std::cerr<<"Failed to read mouseDeltaY for sequence "<<i<<" from file: "<<selectedFile<<"\r\n";
						ReportStreamState(file);
						return;
					}
				} else file.seekg(10, std::ios_base::cur);
				// Load stateData for all time steps
				if(!file.read(reinterpret_cast<char*>(batch->stateData+index*stateSize_), stateSize_)){
					std::cerr<<"Failed to read stateData at index "<<index<<" from file: "<<selectedFile<<"\r\n";
					ReportStreamState(file);
					return;
				}
			}
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