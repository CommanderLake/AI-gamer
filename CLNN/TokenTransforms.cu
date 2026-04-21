#include "CuCommon.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
__global__ void PatchMergeKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const int outRows = patchRows/2;
	const int outCols = patchCols/2;
	const int outTokens = outRows*outCols;
	const int outChannels = embedDim*4;
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*outTokens*outChannels;
	if(idx >= total) return;
	const int channel = idx % outChannels;
	const int outToken = idx/outChannels % outTokens;
	const int batchIndex = idx/(static_cast<size_t>(outTokens)*outChannels);
	const int outRow = outToken/outCols;
	const int outCol = outToken % outCols;
	const int quadrant = channel/embedDim;
	const int inChannel = channel - quadrant*embedDim;
	const int qRow = quadrant/2;
	const int qCol = quadrant % 2;
	const int inRow = outRow*2 + qRow;
	const int inCol = outCol*2 + qCol;
	const int inToken = inRow*patchCols + inCol;
	const size_t inIdx = (static_cast<size_t>(batchIndex)*tokens + inToken)*embedDim + inChannel;
	output[idx] = input[inIdx];
}
void PatchMerge(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const int outRows = patchRows/2;
	const int outCols = patchCols/2;
	const int outTokens = outRows*outCols;
	const int outChannels = embedDim*4;
	const size_t total = static_cast<size_t>(batch)*outTokens*outChannels;
	constexpr int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	PatchMergeKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
__global__ void PatchUnmergeKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const int outRows = patchRows/2;
	const int outCols = patchCols/2;
	const int outTokens = outRows*outCols;
	const int outChannels = embedDim*4;
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*outTokens*outChannels;
	if(idx >= total) return;
	const int channel = idx % outChannels;
	const int outToken = idx/outChannels % outTokens;
	const int batchIndex = idx/(static_cast<size_t>(outTokens)*outChannels);
	const int outRow = outToken/outCols;
	const int outCol = outToken % outCols;
	const int quadrant = channel/embedDim;
	const int inChannel = channel - quadrant*embedDim;
	const int qRow = quadrant/2;
	const int qCol = quadrant % 2;
	const int inRow = outRow*2 + qRow;
	const int inCol = outCol*2 + qCol;
	const int inToken = inRow*patchCols + inCol;
	const size_t outIdx = (static_cast<size_t>(batchIndex)*tokens + inToken)*embedDim + inChannel;
	output[outIdx] = input[idx];
}
void PatchUnmerge(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const int outRows = patchRows/2;
	const int outCols = patchCols/2;
	const int outTokens = outRows*outCols;
	const int outChannels = embedDim*4;
	const size_t total = static_cast<size_t>(batch)*outTokens*outChannels;
	constexpr int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	PatchUnmergeKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
__global__ void TokensToSpatialKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	if(idx >= total) return;
	const int feature = idx % embedDim;
	const int tokenIndex = idx/embedDim % tokens;
	const int batchIndex = idx/(static_cast<size_t>(embedDim)*tokens);
	const int row = tokenIndex/patchCols;
	const int col = tokenIndex % patchCols;
	const size_t outIdx = ((static_cast<size_t>(batchIndex)*embedDim + feature)*patchRows + row)*patchCols + col;
	output[outIdx] = input[idx];
}
void TokensToSpatial(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	int bs = 256;
	const auto blocks = DivCeil(total, bs);
	TokensToSpatialKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
__global__ void SpatialToTokensKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*embedDim*patchRows*patchCols;
	if(idx >= total) return;
	const int col = idx % patchCols;
	const int row = idx/patchCols % patchRows;
	const int feature = idx/(patchCols*patchRows) % embedDim;
	const int batchIndex = idx/(static_cast<size_t>(embedDim)*patchRows*patchCols);
	const int tokenIndex = row*patchCols + col;
	const size_t outIdx = (static_cast<size_t>(batchIndex)*tokens + tokenIndex)*embedDim + feature;
	output[outIdx] = input[idx];
}
void SpatialToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch)*embedDim*patchRows*patchCols;
	int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	SpatialToTokensKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
__global__ void TokensToWindowsKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	if(idx >= total) return;
	const int feature = idx % embedDim;
	const int tokenIndex = idx/embedDim % tokens;
	const int batchIndex = idx/(static_cast<size_t>(embedDim)*tokens);
	const int row = tokenIndex/patchCols;
	const int col = tokenIndex % patchCols;
	const int shiftedRow = (row - shiftHeight + patchRows) % patchRows;
	const int shiftedCol = (col - shiftWidth + patchCols) % patchCols;
	const int windowRow = shiftedRow/windowHeight;
	const int windowCol = shiftedCol/windowWidth;
	const int windowsCols = patchCols/windowWidth;
	const int windowIndex = windowRow*windowsCols + windowCol;
	const int localRow = shiftedRow % windowHeight;
	const int localCol = shiftedCol % windowWidth;
	const int windowToken = localRow*windowWidth + localCol;
	const int windowTokens = windowHeight*windowWidth;
	const int windowCount = windowsCols*(patchRows/windowHeight);
	const size_t outColumn = (static_cast<size_t>(batchIndex)*windowCount + windowIndex)*windowTokens + windowToken;
	const size_t outIdx = outColumn*embedDim + feature;
	output[outIdx] = input[idx];
}
void TokensToWindows(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth){
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	TokensToWindowsKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols, windowHeight, windowWidth, shiftHeight, shiftWidth);
	checkCUDA(cudaGetLastError());
}
__global__ void WindowsToTokensKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	if(idx >= total) return;
	const int feature = idx % embedDim;
	const int tokenIndex = idx/embedDim % tokens;
	const int batchIndex = idx/(static_cast<size_t>(embedDim)*tokens);
	const int row = tokenIndex/patchCols;
	const int col = tokenIndex % patchCols;
	const int shiftedRow = (row + shiftHeight) % patchRows;
	const int shiftedCol = (col + shiftWidth) % patchCols;
	const int windowRow = shiftedRow/windowHeight;
	const int windowCol = shiftedCol/windowWidth;
	const int windowsCols = patchCols/windowWidth;
	const int windowIndex = windowRow*windowsCols + windowCol;
	const int localRow = shiftedRow % windowHeight;
	const int localCol = shiftedCol % windowWidth;
	const int windowToken = localRow*windowWidth + localCol;
	const int windowTokens = windowHeight*windowWidth;
	const int windowCount = windowsCols*(patchRows/windowHeight);
	const size_t inColumn = (static_cast<size_t>(batchIndex)*windowCount + windowIndex)*windowTokens + windowToken;
	const size_t inIdx = inColumn*embedDim + feature;
	output[idx] = input[inIdx];
}
void WindowsToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth){
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	WindowsToTokensKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols, windowHeight, windowWidth, shiftHeight, shiftWidth);
	checkCUDA(cudaGetLastError());
}