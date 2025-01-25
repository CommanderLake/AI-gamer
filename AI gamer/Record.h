#pragma once
#include <atomic>
#include <windows.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstdint>
class Record{
public:
	static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	explicit Record();
	~Record();
	static void Run();
	void ListenForKey();
	void ProcessRawInput(LPARAM lParam);
	void StartCapture();
	void PauseCapture();
	void WriteFrameData();
	void ProcessKeyStates();
	void FrameCaptureTimer();
	HWND hwnd_ = nullptr;
	bool inited_ = false;
	std::ofstream outputFile_;
	std::atomic<bool> capturing_ = false;
	int keyEvents_[256] = {0};
	std::thread captureThread_;
	uint16_t keyStates_ = 0;
	int mouseDeltaX_ = 0;
	int mouseDeltaY_ = 0;
	int frameSize_ = 0;
	unsigned long long fbSize_ = 0;
	std::unordered_map<int, int> codeToBitPos_;
	int* keyCodeToBitPos;
};