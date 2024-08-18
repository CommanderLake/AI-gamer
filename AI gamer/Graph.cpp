#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_backend.h>
#include <iostream>

#define CHECK_CUDNN(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << " - " << cudnnGetErrorString(status) << std::endl; \
            exit(1); \
        } \
    } while(0)

void createTensorDescriptor(cudnnBackendDescriptor_t &desc, cudnnDataType_t dtype, int64_t dims[], int64_t strides[], int64_t alignment, int64_t unique_id){
	CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &desc));
	CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
	CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims));
	CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides));
	CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &unique_id));
	CHECK_CUDNN(cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
	CHECK_CUDNN(cudnnBackendFinalize(desc));
}

void createConvLayer(cudnnHandle_t handle, cudnnBackendDescriptor_t &inputDesc, cudnnBackendDescriptor_t &filterDesc, cudnnBackendDescriptor_t &outputDesc, cudnnBackendDescriptor_t &convDesc, cudnnBackendDescriptor_t &fprop, int64_t inputDims[], int64_t filterDims[], int64_t outputDims[]){
	// Create convolution tensor descriptors
	int64_t inputStr[] = {inputDims[1] * inputDims[2] * inputDims[3], inputDims[2] * inputDims[3], inputDims[3], 1};
	int64_t filterStr[] = {filterDims[1] * filterDims[2] * filterDims[3], filterDims[2] * filterDims[3], filterDims[3], 1};
	int64_t outputStr[] = {outputDims[1] * outputDims[2] * outputDims[3], outputDims[2] * outputDims[3], outputDims[3], 1};

	createTensorDescriptor(inputDesc, CUDNN_DATA_HALF, inputDims, inputStr, 2, 'x');
	createTensorDescriptor(filterDesc, CUDNN_DATA_HALF, filterDims, filterStr, 2, 'w');
	createTensorDescriptor(outputDesc, CUDNN_DATA_HALF, outputDims, outputStr, 2, 'y');

	// Create and finalize convolution descriptor
	CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convDesc));
	const int64_t nbDims = 2;
	const int64_t pad[] = {1, 1};
	const int64_t stride[] = {1, 1};
	const int64_t dilation[] = {1, 1};
	const cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;
	const cudnnDataType_t dataType = CUDNN_DATA_HALF;

	CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &nbDims));
	CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dataType));
	CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
	CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
	CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
	CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, nbDims, stride));
	CHECK_CUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, nbDims, dilation));
	CHECK_CUDNN(cudnnBackendFinalize(convDesc));

	// Create and finalize convolution forward operation
	CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &fprop));
	CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &inputDesc));
	CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &filterDesc));
	CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outputDesc));
	CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convDesc));
	const float alpha = 1.0, beta = 0.0;
	CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
	CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
	CHECK_CUDNN(cudnnBackendFinalize(fprop));
}

void createFullyConnectedLayer(cudnnBackendDescriptor_t &inputDesc, cudnnBackendDescriptor_t &outputDesc, cudnnBackendDescriptor_t &fcDesc, int64_t inputDims[], int64_t outputDims[], int64_t inputStr[], int64_t outputStr[], int64_t alignment, int64_t inputUID, int64_t outputUID){
	createTensorDescriptor(inputDesc, CUDNN_DATA_HALF, inputDims, inputStr, alignment, inputUID);
	createTensorDescriptor(outputDesc, CUDNN_DATA_HALF, outputDims, outputStr, alignment, outputUID);
}

void graph(){
	cudnnHandle_t handle;
	CHECK_CUDNN(cudnnCreate(&handle));

	// Tensor dimensions for convolution layers
	const int64_t n = 1, c = 3, h = 224, w = 224;
	const int64_t k1 = 64, k2 = 128, k3 = 256, k4 = 512, k5 = 512;
	const int64_t convOutputH = 112, convOutputW = 112;

	int64_t conv1Dims[] = {n, c, h, w};
	int64_t conv1OutDims[] = {n, k1, convOutputH, convOutputW};

	int64_t conv2Dims[] = {n, k1, convOutputH, convOutputW};
	int64_t conv2OutDims[] = {n, k2, convOutputH, convOutputW};

	int64_t conv3Dims[] = {n, k2, convOutputH, convOutputW};
	int64_t conv3OutDims[] = {n, k3, convOutputH, convOutputW};

	int64_t conv4Dims[] = {n, k3, convOutputH, convOutputW};
	int64_t conv4OutDims[] = {n, k4, convOutputH, convOutputW};

	int64_t conv5Dims[] = {n, k4, convOutputH, convOutputW};
	int64_t conv5OutDims[] = {n, k5, convOutputH, convOutputW};

	// Tensor dimensions for fully connected layers
	int64_t fc1Dims[] = {n, k5 * convOutputH * convOutputW, 1, 1};
	int64_t fc1OutDims[] = {n, 4096, 1, 1};

	int64_t fc2Dims[] = {n, 4096, 1, 1};
	int64_t fc2OutDims[] = {n, 4096, 1, 1};

	int64_t outDims[] = {n, 16, 1, 1};

	// Descriptors for convolution layers
	cudnnBackendDescriptor_t inputDesc, filterDesc, outputDesc;
	cudnnBackendDescriptor_t convDesc, fprop;

	// Create 5 convolutional layers
	createConvLayer(handle, inputDesc, filterDesc, outputDesc, convDesc, fprop, conv1Dims, conv1OutDims, conv2Dims);
	createConvLayer(handle, inputDesc, filterDesc, outputDesc, convDesc, fprop, conv2Dims, conv2OutDims, conv3Dims);
	createConvLayer(handle, inputDesc, filterDesc, outputDesc, convDesc, fprop, conv3Dims, conv3OutDims, conv4Dims);
	createConvLayer(handle, inputDesc, filterDesc, outputDesc, convDesc, fprop, conv4Dims, conv4OutDims, conv5Dims);
	createConvLayer(handle, inputDesc, filterDesc, outputDesc, convDesc, fprop, conv5Dims, conv5OutDims, fc1Dims);

	// Descriptors for fully connected layers
	cudnnBackendDescriptor_t fc1Desc, fc1OutputDesc, fc2Desc, fc2OutputDesc, outputLayerDesc;
	createFullyConnectedLayer(fc1Desc, fc1OutputDesc, fc1Desc, fc1Dims, fc1OutDims, fc1Dims, fc1OutDims, 2, 'a', 'b');
	createFullyConnectedLayer(fc2Desc, fc2OutputDesc, fc2Desc, fc2Dims, fc2OutDims, fc2Dims, fc2OutDims, 2, 'c', 'd');
	createFullyConnectedLayer(outputLayerDesc, fc2OutputDesc, fc2Desc, fc2Dims, outDims, fc2Dims, outDims, 2, 'e', 'f');

	// Create and finalize operation graph
	cudnnBackendDescriptor_t opGraph;
	CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));
	const cudnnBackendDescriptor_t ops[] = {fprop, fc1Desc, fc2Desc, outputLayerDesc};
	CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 4, ops));
	CHECK_CUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
	CHECK_CUDNN(cudnnBackendFinalize(opGraph));

	// Clean up
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(inputDesc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(filterDesc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(outputDesc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(convDesc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(fprop));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(fc1Desc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(fc1OutputDesc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(fc2Desc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(fc2OutputDesc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(outputLayerDesc));
	CHECK_CUDNN(cudnnBackendDestroyDescriptor(opGraph));
	CHECK_CUDNN(cudnnDestroy(handle));
}