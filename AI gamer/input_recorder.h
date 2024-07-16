#pragma once
#include <Windows.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include <cstdint>
#include <map>

#define WM_USER_STOP_CAPTURE (WM_USER + 1)
#define WM_USER_START_CAPTURE (WM_USER + 2)
#define WM_USER_CAPTURE_FRAME (WM_USER + 3)

class InputRecorder{
public:
	explicit InputRecorder(HWND hwnd);
	~InputRecorder();
	void StartCapture();
	void StopCapture();
	void ListenForKey() const;
	void ProcessRawInput(LPARAM lParam);
	void WriteFrameData();

private:
	void ProcessKeyStates();
	void FrameCaptureThread() const;
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
