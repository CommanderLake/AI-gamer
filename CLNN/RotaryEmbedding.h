#pragma once
#include "Layer.h"
class __declspec(dllexport) RotaryEmbedding final : public Layer{
public:
	RotaryEmbedding(int batchSize, int tokens, int embedDim, int numHeads, int rotaryDim, std::string layerName, float theta = 10000.0f, bool interleaved = false);
	~RotaryEmbedding() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetPositionOffset(int offset);
	void SetPositionOffsetsDevice(const int* offsets);
	void SetPositionOffsetsHost(const int* offsets);
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
private:
	int batchSize_, tokens_, embedDim_, numHeads_, headDim_, rotaryDim_;
	float theta_;
	bool interleaved_;
	int basePosition_ = 0;
	const int* positionOffsets_ = nullptr;
	int* ownedPositionOffsets_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
};
