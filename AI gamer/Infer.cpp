#include "Infer.h"
#include "common.h"
#include "NN.h"
#include "NvDisplayCap.h"
#include <csignal>
#include <algorithm>
#include <cmath>
#undef min
#undef max
constexpr int kMouseLeft = NUM_KBD_BUTS_;
constexpr int kMouseRight = NUM_KBD_BUTS_ + 1;
constexpr int kMouseMiddle = NUM_KBD_BUTS_ + 2;
static_assert(NUM_BUTS_ == 14, "Update inference output mapping constants for new NUM_BUTS_");
static_assert(kMouseMiddle == NUM_BUTS_ - 1, "Mouse button indices must match end of button outputs");
static Infer* this_ = nullptr;
void InferSig(const int sig){
	if(sig == SIGINT && this_){
		this_->stop_ = true;
	}
}
Infer::Infer(){
	if(this_) throw std::runtime_error("Infer class can only have one instance");
	this_ = this;
	InitCUDA();
	InitNvFBC();
	checkCUDNN(cudnnCreate(&cudnn_));
	int width, height;
	GrabFrameUInt8(&width, &height, true, false);
	scaleFactor_ = width/TGT_STATE_WIDTH_;
	if(scaleFactor_ < 1){ scaleFactor_ = 1; }
	int scaledWidth = 0, scaledHeight = 0;
	GrabFrameScaleUInt8(cudnn_, &scaledWidth, &scaledHeight, scaleFactor_, true, false);
	nn_ = new NN(scaledWidth, scaledHeight, false);
	cudaMallocHost(&predictionsF_, NUM_CTRLS_*sizeof(float));
	CUDAMallocZero(&frameHalf_, nn_->stateSize_*sizeof(__half));
	listenThread_ = std::thread(&Infer::ListenForKey, this);
	signal(SIGINT, InferSig);
}
Infer::~Infer(){
	Dispose();
}
void Infer::Dispose(){
	if(disposed_.exchange(true)){ return; }
	stop_ = true;
	inferEnable_ = false;
	if(listenThread_.joinable() && listenThread_.get_id() != std::this_thread::get_id()){ listenThread_.join(); }
	ReleaseOutputs();
	delete nn_;
	nn_ = nullptr;
	if(cudnn_){ checkCUDNN(cudnnDestroy(cudnn_)); }
	DisposeNvFBC();
	if(frameHalf_){
		cudaFree(frameHalf_);
		frameHalf_ = nullptr;
	}
	if(predictionsF_){
		cudaFreeHost(predictionsF_);
		predictionsF_ = nullptr;
	}
	cudnn_ = nullptr;
	this_ = nullptr;
	cudaDeviceReset();
}
void Infer::ListenForKey(){
	std::cout << "\nF9 to start Inference\n";
	std::cout << "Escape to stop\n";
	while(!stop_){
		if(!inferEnable_.load() && GetAsyncKeyState(VK_F9) & 0x8000){
			StartInfer();
			while(!stop_ && GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(inferEnable_.load() && GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			PauseInfer();
			while(!stop_ && GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
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
	constexpr float maxEncodedAxis = 4.0f;
	constexpr float maxMouseDelta = 32767.0f;
	INPUT inputs[20] = {};
	int inputIndex = 0;
	for(int i = 0; i < NUM_KBD_BUTS_; ++i){
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
	auto decodeAxis = [=](float encoded){
		if(!std::isfinite(encoded)){ return 0; }
		encoded = std::max(-maxEncodedAxis, std::min(maxEncodedAxis, encoded));
		float delta = std::sinh(encoded)*AXIS_SCALE_;
		delta = std::max(-maxMouseDelta, std::min(maxMouseDelta, delta));
		return static_cast<int>(delta);
	};
	const int mouseX = decodeAxis(predictions[NUM_BUTS_]);
	const int mouseY = decodeAxis(predictions[NUM_BUTS_ + 1]);
	//const int mouseX = static_cast<int>(DecompressAxisDelta(predictions[NUM_BUTS_]));
	//const int mouseY = static_cast<int>(DecompressAxisDelta(predictions[NUM_BUTS_ + 1]));
	if(mouseX != 0 || mouseY != 0){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dx = mouseX;
		inputs[inputIndex].mi.dy = mouseY;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MOVE;
		inputIndex++;
	}
	if(inputIndex > 0){ SendInput(inputIndex, inputs, sizeof(INPUT)); }
}
void Infer::ReleaseOutputs(){
	float neutral[NUM_CTRLS_] = {};
	ProcessOutput(neutral);
}
void Infer::Step(){
	const bool inferEnabled = inferEnable_.load();
	if(inferEnabled){
		int capWidth = 0, capHeight = 0;
		const auto* frame = GrabFrameScaleUInt8(cudnn_, &capWidth, &capHeight, scaleFactor_, true, false);
		if(capWidth != nn_->inWidth_ || capHeight != nn_->inHeight_){
			std::cerr << "Capture resolution changed to " << capWidth << "x" << capHeight << ". Reloading network input resolution...\n";
			delete nn_;
			nn_ = new NN(capWidth, capHeight, false);
			cudaFree(frameHalf_);
			CUDAMallocZero(&frameHalf_, nn_->stateSize_*sizeof(__half));
		}
		ConvertByteToHalf(frame, frameHalf_, nn_->stateSize_, true);
		const auto output = nn_->Forward(frameHalf_);
		GetPrediction(output, predictionsF_, NUM_CTRLS_, nn_->batchSize_);
		ProcessOutput(predictionsF_);
	}
	if(inferEnabled != inferLast_){
		if(!inferEnabled){ ReleaseOutputs(); }
		inferLast_ = inferEnabled;
	}
}
void Infer::Run(){
	while(!inferEnable_.load() && !stop_){
		Sleep(10);
	}
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::steady_clock::now();
	try{
		while(!stop_){
			auto currentTime = std::chrono::steady_clock::now();
			nextFrameTime += frameDuration;
			if(currentTime > nextFrameTime) nextFrameTime = currentTime + frameDuration;
			std::this_thread::sleep_until(nextFrameTime);
			Step();
		}
	} catch(const std::exception& e){
		stop_ = true;
		std::cerr << "Error: " << e.what() << "\n";
	}
	Dispose();
}
