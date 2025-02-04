#pragma once
#include "common.h"
#include <atomic>
#include <windows.h>
#include <fstream>
#include <thread>
class Record{
public:
	explicit Record();
	~Record();
	void Run();
	void MassageLoop();
	void Init();
	void Dispose();
	void ListenForKey();
	void ProcessRawInput(LPARAM lParam);
	void StartCapture();
	void PauseCapture();
	void Step(InputState& inputState);
	InputState GetInputStates();
	cudnnContext* cudnn_ = nullptr;
	HWND hwnd_ = nullptr;
	std::ofstream outputFile_;
	std::atomic<bool> stop_ = false;
	std::atomic<bool> recording_ = false;
	int keyEvents_[256]{0};
	InputState inputState_{0};
	int frameSize_ = 0;
	int* keyCodeToBitPos;
	const int tgtStateWidth_ = 320;
	int scaleFactor_ = 2;
};