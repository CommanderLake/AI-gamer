#include "Infer.h"
#include "common.h"
#include "NN.h"
#include "NvDisplayCap.h"
#define WM_USER_PAUSE_INFER (WM_USER + 1)
#define WM_USER_START_INFER (WM_USER + 2)
#define WM_USER_INFER_STEP (WM_USER + 3)
Infer* this_ = nullptr;
static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	switch(uMsg){
		case WM_USER_INFER_STEP:
			this_->Inference();
			break;
		case WM_USER_START_INFER:
			this_->StartInfer();
			break;
		case WM_USER_PAUSE_INFER:
			this_->PauseInfer();
			break;
		case WM_DESTROY: PostQuitMessage(0);
			return 0;
		default: return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	return 0;
}
Infer::Infer(const bool tune) : tune_(tune){
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
	RAWINPUTDEVICE rid[1];
	rid[0].usUsagePage = 0x01; // HID_USAGE_PAGE_GENERIC
	rid[0].usUsage = 0x06;     // HID_USAGE_GENERIC_KEYBOARD
	rid[0].dwFlags = RIDEV_INPUTSINK;
	rid[0].hwndTarget = hwnd_;
	if(!RegisterRawInputDevices(rid, 1, sizeof rid[0])){
		MessageBox(hwnd_, "Failed to register raw input device.", "Error", MB_OK);
	}
	try{
		InitCUDA();
		InitNvFBC();
		AllocGPU();
		cudnnCreate(&cudnn_);
		cublasCreate(&cublas_);
		nn_ = new NN(cudnn_, cublas_, 0, 0, tune_);
		checkCUDA(cudaMallocHost(&hPredictionsF_, numCtrls_*sizeof(float)));
		CUDAMallocZero(&dPredictionsF_, numCtrls_*sizeof(float));
		CUDAMallocZero(&sequenceHalf_, nn_->stateSize_*nn_->seqLength_*sizeof(__half));
	} catch(const std::exception& e){
		std::cerr << "Initialization error: " << e.what() << std::endl;
		return;
	}
	inferThread_ = std::thread(&Infer::FrameCaptureTimer, this);
	inferThread_.detach();
}
Infer::~Infer(){
	stopInfer_ = true;
	inferring_ = false;
	if(inferThread_.joinable()) inferThread_.join();
	delete nn_;
	cudaFree(sequenceHalf_);
	cudaFree(dPredictionsF_);
	cudaFreeHost(hPredictionsF_);
	cublasDestroy(cublas_);
	cudnnDestroy(cudnn_);
	DisposeNvFBC();
}
void Infer::Run(){
	std::thread listenThread(&Infer::ListenForKey, this_);
	listenThread.detach();
	MSG msg = {};
	while(GetMessage(&msg, nullptr, 0, 0)){
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}
void Infer::ListenForKey(){
	std::cout << "F9 to start Inference\r\nF10 to record a correction\r\nF11 to tune network with correction\r\nEscape to stop\r\n";
	while(true){
		if(activeMode_ == InferMode::Off && GetAsyncKeyState(VK_F9) & 0x8000){
			if(!inferring_) PostMessage(hwnd_, WM_USER_START_INFER, 0, 0);
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(activeMode_ != InferMode::Correct && GetAsyncKeyState(VK_F10) & 0x8000){
			activeMode_ = InferMode::Correct;
			while(GetAsyncKeyState(VK_F10) & 0x8000){ Sleep(10); }
		}
		if(activeMode_ == InferMode::Correct && GetAsyncKeyState(VK_F11) & 0x8000){
			activeMode_ = InferMode::Tune;
			while(GetAsyncKeyState(VK_F11) & 0x8000){ Sleep(10); }
		}
		if(activeMode_ != InferMode::Off && GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			if(inferring_) PostMessage(hwnd_, WM_USER_PAUSE_INFER, 0, 0);
			while(GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void Infer::StartInfer(){
	activeMode_ = InferMode::On;
	inferring_ = true;
	std::cout << "Inference started" << std::endl;
}
void Infer::PauseInfer(){
	activeMode_ = InferMode::Off;
	inferring_ = false;
	std::cout << "Inference paused" << std::endl;
}
void Infer::ProcessOutput(const float* predictions){
	INPUT inputs[20] = {};
	int inputIndex = 0;
	for(int i = 0; i < 11; ++i){
		inputs[inputIndex].type = INPUT_KEYBOARD;
		inputs[inputIndex].ki.wScan = keyMap[i];
		inputs[inputIndex].ki.dwFlags = KEYEVENTF_SCANCODE;
		if(predictions[i] <= 0.5){
			inputs[inputIndex].ki.dwFlags |= KEYEVENTF_KEYUP;
		}
		inputIndex++;
	}
	if(predictions[11] > 0.5){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_LEFTUP;
		inputIndex++;
	}
	if(predictions[12] > 0.5){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
		inputIndex++;
	}
	if(predictions[13] > 0.5){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
		inputIndex++;
	}
	const int mouseX = static_cast<int>(predictions[14]*128.0f);
	const int mouseY = static_cast<int>(predictions[15]*128.0f);
	if(mouseX != 0 || mouseY != 0){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dx = mouseX;
		inputs[inputIndex].mi.dy = mouseY;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MOVE;
		inputIndex++;
	}
	if(inputIndex > 0){ SendInput(inputIndex, inputs, sizeof(INPUT)); }
}
void Infer::Inference(){
	int capWidth = 0, capHeight = 0;
	const auto frame = GrabFrameInt8(&capWidth, &capHeight, true, false);
	if(capWidth != nn_->inWidth_ || capHeight != nn_->inHeight_){
		activeMode_ = InferMode::Off;
		PauseInfer();
		std::cerr << "Capture resolution mismatch.\r\n";
		return;
	}
	if(tune_ && activeMode_ == InferMode::Correct){
		//auto state = new RecordState(nn->stateSize_, );
	}
	//checkCUDA(cudaMemcpy(sequenceHalf, sequenceHalf + frameSize, (nn->seqLength_ - 1)*frameSize, cudaMemcpyDeviceToDevice));
	ConvertAndNormalize(sequenceHalf_/* + (nn->seqLength_ - 1)*frameSize*/, frame, nn_->stateSize_);
	const auto output = nn_->Forward(sequenceHalf_);
	ConvertHalfToFloat(output, dPredictionsF_, numCtrls_);
	checkCUDA(cudaMemcpy(hPredictionsF_, dPredictionsF_, numCtrls_*sizeof(float), cudaMemcpyDeviceToHost));
	ProcessOutput(hPredictionsF_);
}
void Infer::FrameCaptureTimer(){
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	while(!stopInfer_){
		auto currentTime = std::chrono::high_resolution_clock::now();
		nextFrameTime += frameDuration;
		if(currentTime > nextFrameTime){
			nextFrameTime = currentTime + frameDuration;
		}
		std::this_thread::sleep_until(nextFrameTime);
		if(inferring_){
			PostMessage(hwnd_, WM_USER_INFER_STEP, 0, 0);
		}
	}
}