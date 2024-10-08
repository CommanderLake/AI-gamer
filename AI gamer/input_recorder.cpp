#include "input_recorder.h"
#include "common.h"
#include "NvDisplayCap.h"
#include <iomanip>
#include <iostream>
InputRecorder::InputRecorder(const HWND hwnd) : keyStates(0), mouseDeltaX(0), mouseDeltaY(0), hwnd(hwnd), frameSize(0){}
InputRecorder::~InputRecorder(){
	if(outputFile.is_open()){
		outputFile.close();
		std::cout << "Output file closed." << std::endl;
	}
	if(captureThread.joinable()){ captureThread.join(); }
}
void InputRecorder::StartCapture(){
	if(!inited){
		InitCUDA();
		InitNvFBC();
		AllocGPU();
		AllocHost(fbSize);
		int width;
		int height;
		GrabFrameInt8(&width, &height, true, false);
		frameSize = width*height*3;
		outputFile.open(trainDataOutFileName, std::ios::binary);
		if(!outputFile.is_open()){ std::cerr << "Failed to open output file!" << std::endl; } else{ std::cout << "Output file opened successfully." << std::endl; }
		outputFile.write(reinterpret_cast<char*>(&width), sizeof width);
		outputFile.write(reinterpret_cast<char*>(&height), sizeof height);
		inited = true;
	}
	capturing = true;
	lastPacketTime = std::chrono::high_resolution_clock::now();
	std::cout << "Capture started" << std::endl;
	// Start the frame capture thread
	captureThread = std::thread(&InputRecorder::FrameCaptureThread, this);
}
void InputRecorder::StopCapture(){
	capturing = false;
	if(captureThread.joinable()){ captureThread.join(); }
	WriteFrameData();
	std::cout << "Capture paused" << std::endl;
}
void InputRecorder::ListenForKey(){
	std::cout << "Press F9 to start recording and Escape to stop.\r\n";
	while(true){
		if(GetAsyncKeyState(VK_F9) & 0x8000){
			if(!capturing){ PostMessage(hwnd, WM_USER_START_CAPTURE, 0, 0); }
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			if(capturing){ PostMessage(hwnd, WM_USER_STOP_CAPTURE, 0, 0); }
			while(GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void InputRecorder::ProcessRawInput(LPARAM lParam){
	if(!capturing) return;
	UINT dwSize;
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, nullptr, &dwSize, sizeof(RAWINPUTHEADER)) != 0){
		std::cerr << "Failed to get raw input data size." << std::endl;
		return;
	}
	const auto lpb = std::make_unique<BYTE[]>(dwSize);
	if(!lpb){
		std::cerr << "Failed to allocate memory for raw input data." << std::endl;
		return;
	}
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, lpb.get(), &dwSize, sizeof(RAWINPUTHEADER)) != dwSize){
		std::cerr << "GetRawInputData does not return correct size!" << std::endl;
		return;
	}
	const auto raw = reinterpret_cast<RAWINPUT*>(lpb.get());
	if(raw->header.dwType == RIM_TYPEKEYBOARD){ keyEvents[raw->data.keyboard.MakeCode] = !(raw->data.keyboard.Flags & 1); } else if(raw->header.dwType == RIM_TYPEMOUSE){
		mouseDeltaX += raw->data.mouse.lLastX;
		mouseDeltaY += raw->data.mouse.lLastY;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_DOWN) keyEvents[11] = true;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_UP) keyEvents[11] = false;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_DOWN) keyEvents[12] = true;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_UP) keyEvents[12] = false;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_DOWN) keyEvents[13] = true;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_UP) keyEvents[13] = false;
	}
}
void InputRecorder::ProcessKeyStates(){
	auto findBitPos = [](int keyCode) -> int{
		for(int i = 0; i < numButs_; ++i){
			if(keyMap[i] == keyCode){
				return i;
			}
		}
		return -1;
	};
	for(const auto& event : keyEvents){
		int keyCode = event.first;
		const bool isPressed = event.second;
		const int bitPos = findBitPos(keyCode);
		if(bitPos != -1){
			if(isPressed){
				keyStates |= 1 << bitPos;
				nextFrameKeyEvents[keyCode] = true;
			} else{
				keyStates &= ~(1 << bitPos);
				nextFrameKeyEvents[keyCode] = false;
			}
		}
	}
	for(const auto& event : nextFrameKeyEvents){
		const int keyCode = event.first;
		const bool isPressed = event.second;
		const int bitPos = findBitPos(keyCode);
		if(bitPos != -1){
			if(isPressed){
				keyStates |= 1 << bitPos;
			} else{
				keyStates &= ~(1 << bitPos);
			}
		}
	}
	keyEvents.clear();
	nextFrameKeyEvents.clear();
}
void InputRecorder::WriteFrameData(){
	ProcessKeyStates();
	outputFile.write(reinterpret_cast<char*>(&keyStates), sizeof keyStates);
	outputFile.write(reinterpret_cast<char*>(&mouseDeltaX), sizeof mouseDeltaX);
	outputFile.write(reinterpret_cast<char*>(&mouseDeltaY), sizeof mouseDeltaY);
	mouseDeltaX = 0;
	mouseDeltaY = 0;
	int width;
	int height;
	const auto buf = GrabFrameInt8(&width, &height, true, true);
	outputFile.write(reinterpret_cast<char*>(buf), frameSize);
}
void InputRecorder::FrameCaptureThread(){
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	while(capturing){
		nextFrameTime += frameDuration;
		std::this_thread::sleep_until(nextFrameTime);
		if(capturing){
			PostMessage(hwnd, WM_USER_CAPTURE_FRAME, 0, 0);
		}
	}
}