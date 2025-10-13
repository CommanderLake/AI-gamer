#pragma once
#include "Layer.h"
#include "Viewer.h"
class ViewerLayer final : public Layer{
public:
	ViewerLayer(int channels, int height, int width, int gridWidth, std::string windowTitle, bool backwardPass = false, __half* displayData = nullptr);
	~ViewerLayer() override;
	void DisplayData(const __half* data);
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	Viewer* viewer_ = nullptr;
	__half* displayData_;
	std::string windowTitle_;
	bool backwardPass_;
	unsigned char* mosaicD_ = nullptr;
	unsigned char* mosaicH_ = nullptr;
	int mosaicDimW_ = 0, mosaicDimH_ = 0;
	int inC_, inH_, inW_;
	int gridW_, gridH_;
};