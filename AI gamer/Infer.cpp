#include "Infer.h"
#include "common.h"
#include "NN.h"
#include "NvDisplayCap.h"
Infer::Infer(const bool tune) : tune_(tune){
	InitCUDA();
	InitNvFBC();
	cudnnCreate(&cudnn_);
	cublasCreate(&cublas_);
	nn_ = new NN(cudnn_, cublas_, 0, 0, tune_);
	cudaMallocHost(&predictionsF_, NUM_CTRLS_*sizeof(float));
	CUDAMallocZero(&sequenceHalf_, nn_->stateSize_*nn_->seqLength_*sizeof(__half));
	int width, height;
	GrabFrameUInt8(&width, &height, true, false);
	scaleFactor_ = width/TGT_STATE_WIDTH_;
	if(tune_){
		nn_->SetTrain(false);
		record_ = new Record();
		train_ = new Train();
	}
	listenThread_ = std::thread(&Infer::ListenForKey, this);
	listenThread_.detach();
}
Infer::~Infer(){
	stop_ = true;
}
void Infer::Dispose(){
	delete nn_;
	cudaFree(sequenceHalf_);
	cudaFree(predictionsF_);
	cublasDestroy(cublas_);
	cudnnDestroy(cudnn_);
	FreeHost();
	FreeGPU();
	DisposeNvFBC();
	cudaDeviceReset();
}
void Infer::ListenForKey(){
	std::cout << "\nF9 to start Inference\n";
	if(tune_) std::cout << "F10 to record a correction\nF11 to tune network with correction\n";
	std::cout << "Escape to stop\n";
	while(!stop_){
		if(activeMode_ == InferMode::Off && GetAsyncKeyState(VK_F9) & 0x8000){
			StartInfer();
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(tune_ && activeMode_ != InferMode::Correct && GetAsyncKeyState(VK_F10) & 0x8000){
			activeMode_ = InferMode::Correct;
			while(GetAsyncKeyState(VK_F10) & 0x8000){ Sleep(10); }
		}
		if(tune_ && activeMode_ == InferMode::Correct && GetAsyncKeyState(VK_F12) & 0x8000){
			activeMode_ = InferMode::Tune;
			while(GetAsyncKeyState(VK_F11) & 0x8000){ Sleep(10); }
		}
		if(activeMode_ != InferMode::Off && GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			PauseInfer();
			while(GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void Infer::StartInfer(){
	activeMode_ = InferMode::On;
	std::cout << "Inference started\n";
}
void Infer::PauseInfer(){
	activeMode_ = InferMode::Off;
	std::cout << "Inference paused\n";
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
	const int mouseX = static_cast<int>(predictions[14]*1024.0f);
	const int mouseY = static_cast<int>(predictions[15]*1024.0f);
	if(mouseX != 0 || mouseY != 0){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dx = mouseX;
		inputs[inputIndex].mi.dy = mouseY;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MOVE;
		inputIndex++;
	}
	if(inputIndex > 0){ SendInput(inputIndex, inputs, sizeof(INPUT)); }
}
void Infer::Step(InferMode mode){
	int capWidth = 0, capHeight = 0;
	InputState inputState;
	const unsigned char* frame = nullptr;
	if(mode != InferMode::Off){
		if(tune_) inputState = record_->GetInputStates();
		frame = GrabFrameScaleUInt8(cudnn_, &capWidth, &capHeight, scaleFactor_, true, false);
		if(capWidth != nn_->inWidth_ || capHeight != nn_->inHeight_){
			PauseInfer();
			std::cerr << "Capture resolution mismatch\n";
			return;
		}
	}
	if(mode != lastMode_){
		memset(predictionsF_, 0, NUM_CTRLS_*sizeof(float));
		ProcessOutput(predictionsF_);
		if(mode == InferMode::Tune && states_.size() >= nn_->batchSize_){
			nn_->SetTrain(true);
			train_->TuneModel(nn_, states_, 5, 0.000001);
			nn_->SetTrain(false);
			std::cout << "Tuned with " << states_.size() << " states\n";
			activeMode_ = InferMode::On;
		} else if(mode == InferMode::Tune && states_.size() < nn_->batchSize_){
			std::cout << "Sample size too small to tune\n";
			activeMode_ = InferMode::On;
		}
		if(mode == InferMode::Tune || mode == InferMode::Off || mode == InferMode::On){
			for(const StateSingle* state : states_){
				delete state;
			}
			states_.clear();
		}
		lastMode_ = mode;
	}
	if(mode == InferMode::Correct){
		const auto state = new StateSingle(inputState, frame, nn_->stateSize_, true);
		states_.push_back(state);
	} else if(mode == InferMode::On){
		//BlockShiftHalf(sequenceHalf_ + nn_->stateSize_, -nn_->stateSize_, nn_->seqLength_);
		//ConvertByteToHalf(frame, sequenceHalf_ + (nn_->seqLength_ - 1)*nn_->stateSize_, nn_->stateSize_, true);
		ConvertByteToHalf(frame, sequenceHalf_, nn_->stateSize_, true);
		const auto output = nn_->Forward(sequenceHalf_);
		GetPrediction(output, predictionsF_, NUM_CTRLS_, nn_->batchStateTotal_);
		ProcessOutput(predictionsF_);
	}
}
void Infer::Run(){
	while(activeMode_ == InferMode::Off) Sleep(10);
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	try{
		while(!stop_){
			auto currentTime = std::chrono::high_resolution_clock::now();
			nextFrameTime += frameDuration;
			if(currentTime > nextFrameTime) nextFrameTime = currentTime + frameDuration;
			std::this_thread::sleep_until(nextFrameTime);
			Step(activeMode_);
		}
		PostQuitMessage(0);
	} catch(const std::exception& e){
		std::cerr << "Error: " << e.what() << "\n";
		PostQuitMessage(1);
	}
	stop_ = true;
	Dispose();
	std::terminate();
}