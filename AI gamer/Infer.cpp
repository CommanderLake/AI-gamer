#include "Infer.h"
#include "common.h"
#include "CuCommon.cuh"
#include "NN.h"
#include "NvDisplayCap.h"
#include <csignal>
#undef min
#undef max
constexpr int kKbdButs = 11;
constexpr int kMouseLeft = kKbdButs;
constexpr int kMouseRight = kKbdButs + 1;
constexpr int kMouseMiddle = kKbdButs + 2;
static_assert(NUM_BUTS_ == 14, "Update inference output mapping constants for new NUM_BUTS_");
static_assert(kMouseMiddle == NUM_BUTS_ - 1, "Mouse button indices must match end of button outputs");
static Infer* this_ = nullptr;
void InferSig(const int sig){
	if(sig == SIGINT){
		this_->stop_ = true;
	}
}
Infer::Infer(){
	if(this_) throw std::runtime_error("Infer class can only have one instance");
	this_ = this;
	InitCUDA();
	InitNvFBC();
	cudnnCreate(&cudnn_);
	nn_ = new NN(cudnn_, 0, 0, false);
	cudaMallocHost(&predictionsF_, NUM_CTRLS_*sizeof(float));
	CUDAMallocZero(&frameHalf_, nn_->stateSize_*sizeof(__half));
	int width, height;
	GrabFrameUInt8(&width, &height, true, false);
	scaleFactor_ = width/TGT_STATE_WIDTH_;
	listenThread_ = std::thread(&Infer::ListenForKey, this);
	listenThread_.detach();
	signal(SIGINT, InferSig);
}
Infer::~Infer(){
	stop_ = true;
}
void Infer::Dispose(){
	delete nn_;
	cudaFree(frameHalf_);
	cudaFree(predictionsF_);
	cudnnDestroy(cudnn_);
	FreeHost();
	FreeGPU();
	DisposeNvFBC();
	cudaDeviceReset();
}
void Infer::ListenForKey(){
	std::cout << "\nF9 to start Inference\n";
	std::cout << "Escape to stop\n";
	while(!stop_){
		if(!inferEnable_ && GetAsyncKeyState(VK_F9) & 0x8000){
			StartInfer();
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(inferEnable_ && GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			PauseInfer();
			while(GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void Infer::StartInfer(){
	inferEnable_ = true;
	std::cout << "Inference started\n";
}
void Infer::PauseInfer(){
	inferEnable_ = false;
	std::cout << "Inference paused\n";
}
void Infer::ProcessOutput(const float* predictions){
	constexpr auto thr = 0.0f;
	INPUT inputs[20] = {};
	int inputIndex = 0;
	for(int i = 0; i < kKbdButs; ++i){
		const bool pressed = predictions[i] > thr;
		inputs[inputIndex].type = INPUT_KEYBOARD;
		inputs[inputIndex].ki.wScan = keyMap[i];
		inputs[inputIndex].ki.dwFlags = KEYEVENTF_SCANCODE;
		if(!pressed){
			inputs[inputIndex].ki.dwFlags |= KEYEVENTF_KEYUP;
		}
		inputIndex++;
	}
	if(predictions[kMouseLeft] > thr){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_LEFTUP;
		inputIndex++;
	}
	if(predictions[kMouseRight] > thr){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
		inputIndex++;
	}
	if(predictions[kMouseMiddle] > thr){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
		inputIndex++;
	} else{
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
		inputIndex++;
	}
	const int mouseX = static_cast<int>(predictions[NUM_BUTS_]*AXIS_SCALE_);
	const int mouseY = static_cast<int>(predictions[NUM_BUTS_ + 1]*AXIS_SCALE_);
	if(mouseX != 0 || mouseY != 0){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dx = mouseX;
		inputs[inputIndex].mi.dy = mouseY;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MOVE;
		inputIndex++;
	}
	if(inputIndex > 0){ SendInput(inputIndex, inputs, sizeof(INPUT)); }
}
void Infer::Step(){
	if(inferEnable_){
		int capWidth = 0, capHeight = 0;
		const auto* frame = GrabFrameScaleUInt8(cudnn_, &capWidth, &capHeight, scaleFactor_, true, false);
		if(capWidth != nn_->inWidth_ || capHeight != nn_->inHeight_){
			PauseInfer();
			std::cerr << "Capture resolution mismatch\n";
		}
		ConvertByteToHalf(frame, frameHalf_, nn_->stateSize_, true);
		const auto output = nn_->Forward(frameHalf_);
		GetPrediction(output, predictionsF_, NUM_CTRLS_, nn_->batchSize_);
		ProcessOutput(predictionsF_);
	}
	if(inferEnable_ != inferLast_){
		if(!inferEnable_){
			memset(predictionsF_, 0, NUM_CTRLS_*sizeof(float));
			ProcessOutput(predictionsF_);
		}
		inferLast_ = inferEnable_;
	}
}
void Infer::Run(){
	while(!inferEnable_){
		if(stop_) goto end;
		Sleep(10);
	}
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	try{
		while(!stop_){
			auto currentTime = std::chrono::high_resolution_clock::now();
			nextFrameTime += frameDuration;
			if(currentTime > nextFrameTime) nextFrameTime = currentTime + frameDuration;
			std::this_thread::sleep_until(nextFrameTime);
			Step();
		}
	} catch(const std::exception& e){
		stop_ = true;
		std::cerr << "Error: " << e.what() << "\n";
	}
	end:
	PostQuitMessage(0);
	Dispose();
}