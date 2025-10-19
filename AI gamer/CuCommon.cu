#include "CuCommon.cuh"
#include "WeightInitMethod.h"
#include <ctime>
curandGenerator_t generator_;
int GS, BS, RPB, CPB, TPG, maxTPB, smemPB;
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
int ConvertSmVer2Cores(int major, int minor){
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct{
		int SM; // 0xMm (hexadecimal notation), M = SM Major version and m = SM minor version
		int Cores;
	} sSMtoCores;
	constexpr sSMtoCores nGpuArchCoresPerSM[] = {
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
int DivCeil(const int a, const int b){ return a%b != 0 ? a/b + 1 : a/b; }
void GetLaunchConfigGridStride(int n, int& blocks, int& tpb){
	if(tpb <= 0 || tpb > 1024) tpb = BS;
	blocks = min(DivCeil(n, tpb*8), GS);
}
static bool inited = false;
void InitCUDA(){
	if(inited) return;
	const CUresult cudaRes = cuInit(0);
	if(cudaRes != CUDA_SUCCESS){
		const char* pStr = nullptr;
		cuGetErrorString(cudaRes, &pStr);
		throw std::runtime_error("CUDA Init failed, error string:\n\n" + std::string(pStr));
	}
	inited = true;
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
	TPB = TPB / warps*warps;
	TPG = warps;
	while(TPG*2 <= TPB && TPG < warps){ TPG *= 2; }
	const int groups = TPB / TPG;
	RPB = sqrt(groups);
	CPB = groups / RPB;
	while(RPB*CPB < groups){ if(RPB < CPB){ RPB++; } else{ CPB++; } }
	curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator_, static_cast<unsigned long long>(time(nullptr)));
}
void WeightInit(__half* weights, const int elementCount, const int fanIn, const WeightInitMethod method, const float scale){
	if(fanIn <= 0){
		throw std::invalid_argument("WeightInit fanIn must be positive");
	}
	float* weightFloat;
	checkCUDA(cudaMalloc(&weightFloat, elementCount*sizeof(float)));
	const float factor = method == Xavier ? 1.0f : 2.0f;
	curandGenerateNormal(generator_, weightFloat, elementCount, 0.0f, 1.0f);
	ConvertFloatToHalfScale(weights, weightFloat, elementCount, sqrtf(factor / fanIn)*scale);
	cudaFree(weightFloat);
}