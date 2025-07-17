#pragma once
#include "common.h"
#include "Record.h"
#include "Train.h"
#include <windows.h>
#include <atomic>
struct cublasContext;
class NN;
class Infer{
public:
	explicit Infer(bool tune);
	~Infer();
	void Run();
	void Dispose();
	void ListenForKey();
	void StartInfer();
	void PauseInfer();
	static void ProcessOutput(const float* predictions);
	void Step(InferMode mode);
	Record* record_ = nullptr;
	Train* train_ = nullptr;
	HWND hwnd_ = nullptr;
	bool tune_ = false;
	std::atomic<bool> stop_ = false;
	InferMode activeMode_ = InferMode::Off;
	InferMode lastMode_ = InferMode::Off;
	std::vector<StateSingle*> states_;
	cudnnContext* cudnn_ = nullptr;
	cublasContext* cublas_ = nullptr;
	NN* nn_ = nullptr;
	std::thread inferThread_;
	std::thread listenThread_;
	float* predictionsF_ = nullptr;
	__half* sequenceHalf_ = nullptr;
	int scaleFactor_ = 2;
};