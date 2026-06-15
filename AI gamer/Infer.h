#pragma once
#include "common.h"
#include "Record.h"
#include "Train.h"
#include <windows.h>
#include <atomic>
class NN;
class Infer{
public:
	explicit Infer();
	~Infer();
	void Run();
	void Dispose();
	void ListenForKey();
	void StartInfer();
	void PauseInfer();
	static void ProcessOutput(const float* predictions);
	static void ReleaseOutputs();
	void Step();
	Record* record_ = nullptr;
	Train* train_ = nullptr;
	HWND hwnd_ = nullptr;
	std::atomic<bool> stop_ = false;
	std::atomic<bool> inferEnable_ = false;
	std::atomic<bool> disposed_ = false;
	cudnnContext* cudnn_ = nullptr;
	NN* nn_ = nullptr;
	std::thread inferThread_;
	std::thread listenThread_;
	float* predictionsF_ = nullptr;
	__half* frameHalf_ = nullptr;
	int scaleFactor_ = 2;
	bool inferLast_ = false;
};
