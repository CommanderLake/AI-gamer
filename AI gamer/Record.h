#pragma once
#include "common.h"
#include <atomic>
#include <windows.h>
#include <fstream>
#include <thread>
struct cudnnContext;
class Record{
public:
	explicit Record();
	~Record();
	void Run();
	void MassageLoop();
	void Init();
	void Dispose();
	void StopThreads();
	void ListenForKey();
	void ProcessRawInput(LPARAM lParam);
	void StartCapture();
	void PauseCapture();
	const unsigned char* CaptureFrame();
	void Step(const InputState& inputState, const unsigned char* frame);
	InputState GetInputStates();
	cudnnContext* cudnn_ = nullptr;
	std::atomic<HWND> hwnd_ = nullptr;
	std::atomic<DWORD> messageThreadId_ = 0;
	std::ofstream outputFile_;
	std::atomic<bool> stop_ = false;
	std::atomic<bool> recording_ = false;
	std::atomic<bool> disposed_ = false;
	std::mutex inputsMutex;
	std::thread messageThread_;
	std::thread listenThread_;
	int keyCodeToBitPos[256] = {};
	int keyEvents_[256] = {};
	bool mouseButtons_[3] = {};
	InputState inputState_ = {};
	int frameSize_ = 0;
	int scaleFactor_ = 2;
};
