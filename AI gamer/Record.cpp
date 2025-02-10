#include "Record.h"
#include "common.h"
#include "NvDisplayCap.h"
#include <iomanip>
#include <iostream>
Record* this_ = nullptr;
static LRESULT CALLBACK WindowProcRecord(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	switch(uMsg){
		case WM_INPUT: this_->ProcessRawInput(lParam);
			break;
		case WM_DESTROY: PostQuitMessage(0);
			return 0;
		default: return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	return 0;
}
Record::Record(){
	this_ = this;
	for(int i = 0; i<256; ++i){ keyCodeToBitPos[i] = -1; }
	for(int i = 0; i<NUM_BUTS_; ++i){ keyCodeToBitPos[keyMap[i]] = i; }
	std::thread t1(&Record::MassageLoop, this);
	t1.detach();
}
Record::~Record(){
	stop_ = true;
	recording_ = false;
}
void Record::Init(){
	int width, height;
	GrabFrameUInt8(&width, &height, true, false);
	scaleFactor_ = width/TGT_STATE_WIDTH_;
	GrabFrameScaleUInt8(cudnn_, &width, &height, scaleFactor_, true, true);
	frameSize_ = width*height*3;
	outputFile_.open(trainDataOutFileName, std::ios::binary);
	if(!outputFile_.is_open()){ throw std::runtime_error("Failed to open output file"); }
	outputFile_.write(reinterpret_cast<char*>(&width), sizeof width);
	outputFile_.write(reinterpret_cast<char*>(&height), sizeof height);
}
void Record::Dispose(){
	if(outputFile_&&outputFile_.is_open()){
		outputFile_.close();
		std::cout<<"Output file closed\n";
	}
	cudnnDestroy(cudnn_);
	FreeHost();
	FreeGPU();
	DisposeNvFBC();
	cudaDeviceReset();
}
void Record::ListenForKey(){
	std::cout<<"Press F9 to start recording and Escape to pause\n";
	while(true){
		if(GetAsyncKeyState(VK_F9)&0x8000){
			if(!recording_) StartCapture();
			while(GetAsyncKeyState(VK_F9)&0x8000){ Sleep(10); }
		}
		if(GetAsyncKeyState(VK_ESCAPE)&0x8000){
			if(recording_) PauseCapture();
			while(GetAsyncKeyState(VK_ESCAPE)&0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void Record::MassageLoop(){
	const HINSTANCE hInstance = GetModuleHandle(nullptr);
	constexpr char className[] = "RecordCaptureWindowClass";
	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProcRecord;
	wc.hInstance = hInstance;
	wc.lpszClassName = className;
	RegisterClass(&wc);
	hwnd_ = CreateWindowEx(0, className, "Record Capture", WS_EX_TOOLWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, hInstance, nullptr);
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
	if(RegisterRawInputDevices(rid, 2, sizeof rid[0])==FALSE){ throw std::runtime_error("Failed to register input devices"); }
	MSG msg = {};
	while(GetMessage(&msg, nullptr, 0, 0)){
		if(msg.message==WM_QUIT) break;
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}
void Record::ProcessRawInput(LPARAM lParam){
	unsigned dwSize;
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, nullptr, &dwSize, sizeof(RAWINPUTHEADER))!=0){ throw std::runtime_error("Failed to get raw input data size"); }
	unsigned char lpb[1024];
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER))!=dwSize){ throw std::runtime_error("GetRawInputData does not return correct size"); }
	const auto raw = reinterpret_cast<RAWINPUT*>(lpb);
	if(raw->header.dwType==RIM_TYPEKEYBOARD){
		std::unique_lock<std::mutex> lock(inputsMutex);
		keyEvents_[raw->data.keyboard.MakeCode] = !(raw->data.keyboard.Flags&1);
	} else if(raw->header.dwType==RIM_TYPEMOUSE){
		std::unique_lock<std::mutex> lock(inputsMutex);
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_1_DOWN) keyEvents_[11] = 1;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_1_UP) keyEvents_[11] = 0;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_2_DOWN) keyEvents_[12] = 1;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_2_UP) keyEvents_[12] = 0;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_3_DOWN) keyEvents_[13] = 1;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_3_UP) keyEvents_[13] = 0;
		inputState_.deltaX += raw->data.mouse.lLastX;
		inputState_.deltaY += raw->data.mouse.lLastY;
	}
}
void Record::StartCapture(){
	recording_ = true;
	std::cout<<"Capture started\n";
}
void Record::PauseCapture(){
	recording_ = false;
	std::cout<<"Capture paused\n";
}
InputState Record::GetInputStates(){
	std::unique_lock<std::mutex> lock(inputsMutex);
	for(int keyCode = 0; keyCode<256; ++keyCode){
		const auto bitPos = keyCodeToBitPos[keyCode];
		if(bitPos!=-1){ if(keyEvents_[keyCode]){ inputState_.keyStates |= 1<<bitPos; } else{ inputState_.keyStates &= ~(1<<bitPos); } }
	}
	const auto inputState = inputState_;
	inputState_.deltaX = 0;
	inputState_.deltaY = 0;
	return inputState;
}
void Record::Step(InputState& inputState){
	int width, height;
	const auto buf = GrabFrameScaleUInt8(cudnn_, &width, &height, scaleFactor_, true, true);
	if(width*height*3!=frameSize_){
		std::cout<<"Resolution changed\n";
		PauseCapture();
		return;
	}
	outputFile_.write(reinterpret_cast<char*>(&inputState), sizeof inputState);
	outputFile_.write(reinterpret_cast<char*>(buf), frameSize_);
}
void Record::Run(){
	InitCUDA();
	InitNvFBC();
	cudnnCreate(&cudnn_);
	std::thread t0(&Record::ListenForKey, this);
	t0.detach();
	while(!recording_) Sleep(10);
	Init();
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	while(!stop_){
		auto currentTime = std::chrono::high_resolution_clock::now();
		nextFrameTime += frameDuration;
		if(currentTime>nextFrameTime){ nextFrameTime = currentTime+frameDuration; }
		std::this_thread::sleep_until(nextFrameTime);
		auto inputState = GetInputStates();
		if(recording_){ Step(inputState); }
	}
	PostQuitMessage(0);
	stop_ = true;
	Dispose();
	std::terminate();
}