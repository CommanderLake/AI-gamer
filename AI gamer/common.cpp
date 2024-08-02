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
unsigned char keyReMap[] = {
	VK_UP,       // Up arrow key (replaces W)
	VK_LEFT,     // Left arrow key (replaces A)
	VK_DOWN,     // Down arrow key (replaces S)
	VK_RIGHT,    // Right arrow key (replaces D)
	VK_SPACE,    // Space
	VK_CONTROL,  // Ctrl
	0x10,        // Q
	0x13,        // R
	0x12,        // E
	0x02,        // 1
	0x03,        // 2
	VK_LBUTTON,  // Mouse button 1
	VK_RBUTTON,  // Mouse button 2
	VK_MBUTTON   // Mouse button 3
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
		if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)){
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
void printDataHalf2(const __half* data, const size_t size, const char* label){
	std::vector<__half> h_data(size);
	checkCUDA(cudaMemcpy(h_data.data(), data, size*sizeof(__half), cudaMemcpyDeviceToHost));
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < h_data.size(); ++i){ std::cout << __half2float(h_data[i]) << " "; }
	std::cout << "\r\n";
}
void PrintDataHalf(const __half* data, const size_t size, const char* label){
	const size_t truncatedSize = (size / 128)*128;
	if(truncatedSize == 0){
		printDataHalf2(data, size, label);
		return;
	}
	const auto h_data = static_cast<__half*>(_mm_malloc(truncatedSize*sizeof(__half), 32));
	const auto f_data = static_cast<float*>(_mm_malloc(truncatedSize*sizeof(float), 32));
	checkCUDA(cudaMemcpy(h_data, data, truncatedSize*sizeof(__half), cudaMemcpyDeviceToHost));
	H2F128Asm(f_data, h_data, truncatedSize);
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < truncatedSize; ++i){
		std::cout << f_data[i] << " ";
	}
	std::cout << "\r\n";
	_mm_free(h_data);
	_mm_free(f_data);
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
	for(size_t i = 0; i < size; ++i){ std::cout << data[i] << "\r\n"; }
	std::cout << "\r\n";
}
void PrintDataCharHost(const unsigned char* data, const size_t size, const char* label){
	std::cout << label << ":\r\n";
	for(size_t i = 0; i < size; ++i){ std::cout << data[i] << "\r\n"; }
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
std::vector<std::string> trainingDataFiles = {"E:\\TrainingData\\training_data1.bin", "E:\\TrainingData\\training_data2.bin", "E:\\TrainingData\\training_data3.bin", "E:\\TrainingData\\training_data4.bin", "E:\\TrainingData\\training_data5.bin"};
std::unordered_map<std::string, std::vector<std::streampos>> fileRecordIndex;
std::size_t stateSize;
ThreadPool threadPool(8);
void LoadBatch(StateBatch* batch, int seqLength, int numSequences){
	std::vector<std::future<void>> futures;
	for(size_t i = 0; i < numSequences; ++i){
		futures.push_back(threadPool.Enqueue([i, batch, seqLength](){
			std::mt19937& gen = threadPool.GetThreadGenerator();
			std::uniform_int_distribution<> fileDis(0, trainingDataFiles.size() - 1);
			const std::string selectedFile = trainingDataFiles[fileDis(gen)];
			const auto recordIndexIt = fileRecordIndex.find(selectedFile);
			if(recordIndexIt == fileRecordIndex.end()){
				std::cerr << "No records for file: " << selectedFile << std::endl;
				return;
			}
			std::ifstream file(selectedFile, std::ios::binary | std::ios::in);
			if(!file.is_open()){
				std::cerr << "Failed to open training data file: " << selectedFile << std::endl;
				return;
			}
			const std::uniform_int_distribution<> recordDis(0, recordIndexIt->second.size() - seqLength);
			const size_t recordIndex = recordDis(gen);
			for(int j = 0; j < seqLength; ++j){
				file.seekg(recordIndexIt->second[recordIndex + j]);
				if(!file.read(reinterpret_cast<char*>(&batch->keyStates[i*seqLength + j]), sizeof(unsigned short))){
					std::cerr << "Failed to read keyStates from file: " << selectedFile << std::endl;
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaX[i*seqLength + j]), sizeof(int))){
					std::cerr << "Failed to read mouseDeltaX from file: " << selectedFile << std::endl;
					return;
				}
				if(!file.read(reinterpret_cast<char*>(&batch->mouseDeltaY[i*seqLength + j]), sizeof(int))){
					std::cerr << "Failed to read mouseDeltaY from file: " << selectedFile << std::endl;
					return;
				}
				if(!file.read(reinterpret_cast<char*>(batch->stateData + (i*seqLength + j)*stateSize), stateSize)){
					std::cerr << "Failed to read stateData from file: " << selectedFile << std::endl;
					return;
				}
			}
			}));
	}
	for(auto &future : futures){
		future.get();
	}
}