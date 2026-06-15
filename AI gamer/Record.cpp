#include "Record.h"
#include "common.h"
#include "APICommon.h"
#include "NvDisplayCap.h"
#include <iomanip>
#include <iostream>
#include <csignal>
#include <vector>
static Record* this_ = nullptr;
static LRESULT CALLBACK WindowProcRecord(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	switch(uMsg){
		case WM_INPUT:
			if(this_){ this_->ProcessRawInput(lParam); }
			break;
		case WM_DESTROY: PostQuitMessage(0);
			return 0;
		default: return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	return 0;
}
void RecordSig(const int sig){
	if(sig == SIGINT && this_){
		this_->stop_ = true;
	}
}
Record::Record(){
	if(this_) throw std::runtime_error("Record class can only have one instance");
	this_ = this;
	for(int i = 0; i<256; ++i){ keyCodeToBitPos[i] = -1; }
	for(int i = 0; i<NUM_KBD_BUTS_; ++i){ keyCodeToBitPos[keyMap[i]] = i; }
	messageThread_ = std::thread(&Record::MassageLoop, this);
	signal(SIGINT, RecordSig);
}
Record::~Record(){
	StopThreads();
	Dispose();
	this_ = nullptr;
}
void Record::Init(){
	int width, height;
	GrabFrameUInt8(&width, &height, true, false);
	scaleFactor_ = width/TGT_STATE_WIDTH_;
	if(scaleFactor_ < 1){ scaleFactor_ = 1; }
	GrabFrameScaleUInt8(cudnn_, &width, &height, scaleFactor_, true, true);
	frameSize_ = width*height*3;
	outputFile_.open(trainDataOutFileName, std::ios::binary);
	if(!outputFile_.is_open()){ throw std::runtime_error("Failed to open output file"); }
	outputFile_.write(reinterpret_cast<char*>(&width), sizeof width);
	outputFile_.write(reinterpret_cast<char*>(&height), sizeof height);
}
void Record::Dispose(){
	if(disposed_.exchange(true)){ return; }
	if(outputFile_&&outputFile_.is_open()){
		outputFile_.close();
		std::cout<<"Output file closed\n";
	}
	if(cudnn_){
		checkCUDNN(cudnnDestroy(cudnn_));
		cudnn_ = nullptr;
	}
	DisposeNvFBC();
	cudaDeviceReset();
}
void Record::StopThreads(){
	stop_ = true;
	recording_ = false;
	const HWND hwnd = hwnd_.load();
	if(hwnd){ PostMessage(hwnd, WM_CLOSE, 0, 0); }
	else{
		const DWORD threadId = messageThreadId_.load();
		if(threadId){ PostThreadMessage(threadId, WM_QUIT, 0, 0); }
	}
	if(listenThread_.joinable() && listenThread_.get_id() != std::this_thread::get_id()){ listenThread_.join(); }
	if(messageThread_.joinable() && messageThread_.get_id() != std::this_thread::get_id()){ messageThread_.join(); }
}
void Record::ListenForKey(){
	std::cout<<"Press F9 to start recording and Escape to pause\n";
	while(!stop_){
		if(GetAsyncKeyState(VK_F9)&0x8000){
			if(!recording_) StartCapture();
			while(!stop_ && GetAsyncKeyState(VK_F9)&0x8000){ Sleep(10); }
		}
		if(GetAsyncKeyState(VK_ESCAPE)&0x8000){
			if(recording_) PauseCapture();
			while(!stop_ && GetAsyncKeyState(VK_ESCAPE)&0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void Record::MassageLoop(){
	messageThreadId_ = GetCurrentThreadId();
	try{
		const HINSTANCE hInstance = GetModuleHandle(nullptr);
		constexpr char className[] = "RecordCaptureWindowClass";
		WNDCLASS wc = {};
		wc.lpfnWndProc = WindowProcRecord;
		wc.hInstance = hInstance;
		wc.lpszClassName = className;
		if(!RegisterClass(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS){ throw std::runtime_error("Failed to register capture window class"); }
		const HWND hwnd = CreateWindowEx(0, className, "Record Capture", WS_EX_TOOLWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, hInstance, nullptr);
		if(!hwnd){ throw std::runtime_error("Failed to create capture window"); }
		hwnd_ = hwnd;
		ShowWindow(hwnd, SW_HIDE);
		RAWINPUTDEVICE rid[2];
		rid[0].usUsagePage = 0x01;
		rid[0].usUsage = 0x06;
		rid[0].dwFlags = RIDEV_INPUTSINK;
		rid[0].hwndTarget = hwnd;
		rid[1].usUsagePage = 0x01;
		rid[1].usUsage = 0x02;
		rid[1].dwFlags = RIDEV_INPUTSINK;
		rid[1].hwndTarget = hwnd;
		if(RegisterRawInputDevices(rid, 2, sizeof rid[0])==FALSE){ throw std::runtime_error("Failed to register input devices"); }
		if(stop_){
			DestroyWindow(hwnd);
		} else{
			MSG msg = {};
			while(GetMessage(&msg, nullptr, 0, 0)>0){
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
	} catch(const std::exception& e){
		std::cerr << "Raw input thread error: " << e.what() << "\n";
		stop_ = true;
	}
	hwnd_ = nullptr;
	messageThreadId_ = 0;
}
void Record::ProcessRawInput(LPARAM lParam){
	unsigned dwSize;
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, nullptr, &dwSize, sizeof(RAWINPUTHEADER))!=0){ throw std::runtime_error("Failed to get raw input data size"); }
	std::vector<unsigned char> inputBuffer(dwSize);
	if(GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, inputBuffer.data(), &dwSize, sizeof(RAWINPUTHEADER))!=dwSize){ throw std::runtime_error("GetRawInputData does not return correct size"); }
	const auto raw = reinterpret_cast<RAWINPUT*>(inputBuffer.data());
	if(raw->header.dwType==RIM_TYPEKEYBOARD){
		std::unique_lock<std::mutex> lock(inputsMutex);
		const unsigned makeCode = raw->data.keyboard.MakeCode;
		if(makeCode < 256){ keyEvents_[makeCode] = !(raw->data.keyboard.Flags&RI_KEY_BREAK); }
	} else if(raw->header.dwType==RIM_TYPEMOUSE){
		std::unique_lock<std::mutex> lock(inputsMutex);
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_1_DOWN) mouseButtons_[0] = true;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_1_UP) mouseButtons_[0] = false;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_2_DOWN) mouseButtons_[1] = true;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_2_UP) mouseButtons_[1] = false;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_3_DOWN) mouseButtons_[2] = true;
		if(raw->data.mouse.usButtonFlags&RI_MOUSE_BUTTON_3_UP) mouseButtons_[2] = false;
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
		if(bitPos!=-1){ if(keyEvents_[keyCode]){ inputState_.keyStates |= 1U<<bitPos; } else{ inputState_.keyStates &= ~(1U<<bitPos); } }
	}
	for(int button = 0; button < 3; ++button){
		const int bitPos = NUM_KBD_BUTS_ + button;
		if(mouseButtons_[button]){ inputState_.keyStates |= 1U<<bitPos; }
		else{ inputState_.keyStates &= ~(1U<<bitPos); }
	}
	const auto inputState = inputState_;
	inputState_.deltaX = 0;
	inputState_.deltaY = 0;
	return inputState;
}
const unsigned char* Record::CaptureFrame(){
	int width, height;
	const auto buf = GrabFrameScaleUInt8(cudnn_, &width, &height, scaleFactor_, true, true);
	if(width*height*3!=frameSize_){
		std::cout<<"Resolution changed\n";
		PauseCapture();
		return nullptr;
	}
	return buf;
}
void Record::Step(const InputState& inputState, const unsigned char* frame){
	outputFile_.write(reinterpret_cast<const char*>(&inputState), sizeof inputState);
	outputFile_.write(reinterpret_cast<const char*>(frame), frameSize_);
}
void Record::Run(){
	try{
		InitCUDA();
		InitNvFBC();
		checkCUDNN(cudnnCreate(&cudnn_));
		listenThread_ = std::thread(&Record::ListenForKey, this);
		while(!recording_ && !stop_){
			GetInputStates();
			Sleep(10);
		}
		if(!stop_){
			Init();
			constexpr std::chrono::microseconds frameDuration(33333);
			auto nextFrameTime = std::chrono::steady_clock::now();
			while(!stop_){
				if(!recording_){
					GetInputStates();
					nextFrameTime = std::chrono::steady_clock::now();
					Sleep(10);
					continue;
				}
				const unsigned char* frame = CaptureFrame();
				if(!frame){ continue; }
				auto currentTime = std::chrono::steady_clock::now();
				nextFrameTime += frameDuration;
				if(currentTime>nextFrameTime){ nextFrameTime = currentTime+frameDuration; }
				std::this_thread::sleep_until(nextFrameTime);
				const auto inputState = GetInputStates();
				if(recording_ && !stop_){ Step(inputState, frame); }
			}
		}
	} catch(...){
		StopThreads();
		Dispose();
		throw;
	}
	StopThreads();
	Dispose();
}
