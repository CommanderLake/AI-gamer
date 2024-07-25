#pragma once
#include <atomic>
#include <Windows.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include <cstdint>
#include <map>

class InputRecorder{
public:
	explicit InputRecorder(HWND hwnd);
	~InputRecorder();
	void StartCapture();
	void StopCapture();
	void ListenForKey();
	void ProcessRawInput(LPARAM lParam);
	void WriteFrameData();
private:
	void ProcessKeyStates();
	void FrameCaptureThread();
	std::ofstream outputFile;
	uint16_t keyStates;
	long mouseDeltaX;
	long mouseDeltaY;
	bool capturing = false;
	std::chrono::steady_clock::time_point lastPacketTime;
	unsigned long long fbSize = 0;
	int buf1Size = 0;
	HWND hwnd;
	bool inited = false;
	std::map<int, bool> keyEvents;
	std::map<int, bool> nextFrameKeyEvents;
	std::thread captureThread;
	int frameSize;
};
