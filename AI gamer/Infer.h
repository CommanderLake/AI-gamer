#pragma once
#include "common.h"
#include <windows.h>
#include <atomic>
class NN;
class Infer{
public:
	Infer(bool tune);
	~Infer();
	static void Run();
	void ListenForKey();
	void StartInfer();
	void PauseInfer();
	static void ProcessOutput(const float* predictions);
	void Inference();
	void FrameCaptureTimer();
	HWND hwnd_ = nullptr;
	bool tune_ = false;
	std::atomic<bool> stopInfer_ = false;
	std::atomic<bool> inferring_ = false;
	InferMode activeMode_ = InferMode::Off;
	std::vector<RecordState> states_;
	cudnnContext* cudnn_ = nullptr;
	cublasContext* cublas_ = nullptr;
	NN* nn_ = nullptr;
	std::thread inferThread_;
	float* hPredictionsF_ = nullptr;
	float* dPredictionsF_ = nullptr;
	__half* sequenceHalf_ = nullptr;
};