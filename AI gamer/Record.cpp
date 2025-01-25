#include "Record.h"
#include "common.h"
#include "NvDisplayCap.h"
#include <iomanip>
#include <iostream>
#define WM_USER_PAUSE_INFER (WM_USER + 1)
#define WM_USER_START_INFER (WM_USER + 2)
#define WM_USER_INFER_STEP (WM_USER + 3)
Record* this_ = nullptr;
LRESULT CALLBACK Record::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	switch(uMsg){
		case WM_INPUT: this_->ProcessRawInput(lParam);
			break;
		case WM_USER_START_INFER: this_->StartCapture();
			break;
		case WM_USER_PAUSE_INFER: this_->PauseCapture();
			break;
		case WM_USER_INFER_STEP: this_->WriteFrameData();
			break;
		case WM_DESTROY: PostQuitMessage(0);
			return 0;
		default: return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	return 0;
}
Record::Record(){
	this_ = this;
	const HINSTANCE hInstance = GetModuleHandle(nullptr);
	constexpr char className[] = "InputCaptureWindowClass";
	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = className;
	RegisterClass(&wc);
	hwnd_ = CreateWindowEx(0, className, "Input Capture", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, hInstance, nullptr);
	ShowWindow(hwnd_, SW_HIDE);
	RAWINPUTDEVICE rid[2];
	rid[0].usUsagePage = 0x01;
	rid[0].usUsage = 0x06;
	rid[0].dwFlags = RIDEV_INPUTSINK;
	rid[0].hwndTarget = hwnd_;
	rid[1].usUsagePage = 0x01;
	rid[1].usUsage = 0x02;
	rid[1].dwFlags = RIDEV_INPUTSINK;
	rid[1].hwndTarget = hwnd_;
	if(RegisterRawInputDevices(rid, 2, sizeof rid[0]) == FALSE){
		throw std::runtime_error("Failed to register input devices");
	}
	keyCodeToBitPos = static_cast<int*>(_mm_malloc(256*sizeof(int), 32));
	for(int i = 0; i<256; ++i){
		keyCodeToBitPos[i] = -1;
	}
	for(int i = 0; i<numButs_; ++i){ keyCodeToBitPos[keyMap[i]] = i; }
	InitCUDA();
}
Record::~Record(){
	PauseCapture();
	inited_ = false;
	if(captureThread_.joinable()) captureThread_.join();
	if(outputFile_ && outputFile_.is_open()){
		outputFile_.close();
		std::cout << "Output file closed." << std::endl;
	}
	FreeHost();
	FreeGPU();
	DisposeNvFBC();
}
void Record::Run(){
	std::thread listenThread(&Record::ListenForKey, this_);
	listenThread.detach();
	MSG msg = {};
	while(GetMessage(&msg, nullptr, 0, 0)){
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}
void Record::ListenForKey(){
	std::cout << "Press F9 to start recording and Escape to pause.\r\n";
	while(true){
		if(GetAsyncKeyState(VK_F9) & 0x8000){
			if(!capturing_){ PostMessage(hwnd_, WM_USER_START_INFER, 0, 0); }
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			if(capturing_){ PostMessage(hwnd_, WM_USER_PAUSE_INFER, 0, 0); }
			while(GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void Record::ProcessRawInput(LPARAM lParam){
	//if(!capturing_) return;
	unsigned dwSize;
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, nullptr, &dwSize, sizeof(RAWINPUTHEADER)) != 0){
		std::cerr << "Failed to get raw input data size." << std::endl;
		return;
	}
	//const auto lpb = std::make_unique<unsigned char[]>(dwSize);
	unsigned char lpb[128];
	//if(!lpb){
	//	std::cerr << "Failed to allocate memory for raw input data." << std::endl;
	//	return;
	//}
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER)) != dwSize){
		std::cerr << "GetRawInputData does not return correct size!" << std::endl;
		return;
	}
	const auto raw = reinterpret_cast<RAWINPUT*>(lpb);
	if(raw->header.dwType == RIM_TYPEKEYBOARD){
		keyEvents_[raw->data.keyboard.MakeCode] = !(raw->data.keyboard.Flags & 1);
	} else if(raw->header.dwType == RIM_TYPEMOUSE){
		mouseDeltaX_ += raw->data.mouse.lLastX;
		mouseDeltaY_ += raw->data.mouse.lLastY;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_DOWN) keyEvents_[11] = 1;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_UP) keyEvents_[11] = 0;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_DOWN) keyEvents_[12] = 1;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_UP) keyEvents_[12] = 0;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_DOWN) keyEvents_[13] = 1;
		if(raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_UP) keyEvents_[13] = 0;
	}
}
void Record::StartCapture(){
	if(!inited_){
		inited_ = true;
		InitNvFBC();
		AllocGPU();
		AllocHost(fbSize_);
		int width, height;
		GrabFrameInt8(&width, &height, true, false);
		frameSize_ = width*height*3;
		outputFile_.open(trainDataOutFileName, std::ios::binary);
		if(!outputFile_.is_open()){
			throw std::runtime_error("Failed to open output file");
		}
		outputFile_.write(reinterpret_cast<char*>(&width), sizeof width);
		outputFile_.write(reinterpret_cast<char*>(&height), sizeof height);
	}
	capturing_ = true;
	captureThread_ = std::thread(&Record::FrameCaptureTimer, this);
	captureThread_.detach();
	std::cout << "Capture started" << std::endl;
}
void Record::PauseCapture(){
	capturing_ = false;
	std::cout << "Capture paused" << std::endl;
}
void Record::ProcessKeyStates(){
	for(int keyCode = 0; keyCode<256; ++keyCode){
		const auto bitPos = keyCodeToBitPos[keyCode];
		if(bitPos!=-1){
			if(keyEvents_[keyCode]){
				keyStates_ |= 1<<bitPos;
			} else{
				keyStates_ &= ~(1<<bitPos);
			}
		}
	}
}
void Record::WriteFrameData(){
	auto mdx = mouseDeltaX_;
	mouseDeltaX_ = 0;
	auto mdy = mouseDeltaY_;
	mouseDeltaY_ = 0;
	ProcessKeyStates();
	outputFile_.write(reinterpret_cast<char*>(&keyStates_), sizeof keyStates_);
	outputFile_.write(reinterpret_cast<char*>(&mdx), sizeof mdx);
	outputFile_.write(reinterpret_cast<char*>(&mdy), sizeof mdy);
	int width, height;
	const auto buf = GrabFrameInt8(&width, &height, true, true);
	outputFile_.write(reinterpret_cast<char*>(buf), frameSize_);
}
void Record::FrameCaptureTimer(){
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	while(inited_){
		auto currentTime = std::chrono::high_resolution_clock::now();
		nextFrameTime += frameDuration;
		if(currentTime>nextFrameTime){ nextFrameTime = currentTime+frameDuration; }
		std::this_thread::sleep_until(nextFrameTime);
		if(capturing_){ PostMessage(hwnd_, WM_USER_INFER_STEP, 0, 0); }
	}
}