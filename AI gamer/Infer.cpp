#include "Infer.h"
#include "common.h"
#include "NN.h"
#include "NvDisplayCap.h"
Infer::Infer(const bool tune) : tune_(tune){
	InitCUDA();
	InitNvFBC();
	AllocGPU();
	AllocHost(fbSize_);
	cudnnCreate(&cudnn_);
	cublasCreate(&cublas_);
	nn_ = new NN(cudnn_, cublas_, 0, 0, tune_, 0.000001f);
	checkCUDA(cudaMallocHost(&hPredictionsF_, numCtrls_*sizeof(float)));
	CUDAMallocZero(&dPredictionsF_, numCtrls_*sizeof(float));
	CUDAMallocZero(&sequenceHalf_, nn_->stateSize_*nn_->seqLength_*sizeof(__half));
	if(!tune_) return;
	nn_->SetTrain(false);
	record_ = new Record();
	train_ = new Train();
	listenThread_ = std::thread(&Infer::ListenForKey, this);
	listenThread_.detach();
}
Infer::~Infer(){
	stop_ = true;
}
void Infer::Dispose(){
	delete nn_;
	cudaFree(sequenceHalf_);
	cudaFree(dPredictionsF_);
	cudaFreeHost(hPredictionsF_);
	cublasDestroy(cublas_);
	cudnnDestroy(cudnn_);
	FreeHost();
	FreeGPU();
	DisposeNvFBC();
	cudaDeviceReset();
}
void Infer::ListenForKey(){
	std::cout << "\r\nF9 to start Inference\r\nF10 to record a correction\r\nF11 to tune network with correction\r\nEscape to stop\r\n";
	while(!stop_){
		if(activeMode_ == InferMode::Off && GetAsyncKeyState(VK_F9) & 0x8000){
			StartInfer();
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(activeMode_ != InferMode::Correct && GetAsyncKeyState(VK_F10) & 0x8000){
			activeMode_ = InferMode::Correct;
			while(GetAsyncKeyState(VK_F10) & 0x8000){ Sleep(10); }
		}
		if(activeMode_ == InferMode::Correct && GetAsyncKeyState(VK_F12) & 0x8000){
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
	std::cout << "Inference started" << std::endl;
}
void Infer::PauseInfer(){
	activeMode_ = InferMode::Off;
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
void Infer::Step(InferMode mode){
	int capWidth = 0, capHeight = 0;
	const InputState inputState = record_->GetInputStates();
	const auto frame = GrabFrameInt8(&capWidth, &capHeight, true, false);
	if(capWidth != nn_->inWidth_ || capHeight != nn_->inHeight_){
		PauseInfer();
		std::cerr << "Capture resolution mismatch\r\n";
		return;
	}
	if(mode != previousMode_){
		if(mode == InferMode::Tune && states_.size() >= nn_->batchSize_){
			nn_->SetTrain(true);
			train_->TuneModel(nn_, states_, 5);
			nn_->SetTrain(false);
			std::cout << "Tuned with " << states_.size() << " states\r\n";
			activeMode_ = InferMode::On;
		} else if(mode == InferMode::Tune && states_.size() < nn_->batchSize_){
			std::cout << "Sample size too small to tune\r\n";
			activeMode_ = InferMode::On;
		}
		if(mode == InferMode::Tune || mode == InferMode::Off || mode == InferMode::On){
			for(const StateSingle* state : states_){
				delete state;
			}
			states_.clear();
		}
		previousMode_ = mode;
	}
	if(mode == InferMode::Correct){
		const auto state = new StateSingle(inputState, frame, nn_->stateSize_, true);
		states_.push_back(state);
	} else if(mode == InferMode::On){
		//checkCUDA(cudaMemcpy(sequenceHalf, sequenceHalf + frameSize, (nn->seqLength_ - 1)*frameSize, cudaMemcpyDeviceToDevice));
		ConvertAndNormalize(sequenceHalf_/* + (nn->seqLength_ - 1)*frameSize*/, frame, nn_->stateSize_);
		const auto output = nn_->Forward(sequenceHalf_);
		ConvertHalfToFloat(output, dPredictionsF_, numCtrls_);
		checkCUDA(cudaMemcpy(hPredictionsF_, dPredictionsF_, numCtrls_*sizeof(float), cudaMemcpyDeviceToHost));
		ProcessOutput(hPredictionsF_);
	}
}
void Infer::Run(){
	while(activeMode_ == InferMode::Off) Sleep(10);
	MSG msg = {};
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	try{
		while(!stop_){
			auto currentTime = std::chrono::high_resolution_clock::now();
			nextFrameTime += frameDuration;
			if(currentTime > nextFrameTime) nextFrameTime = currentTime + frameDuration;
			std::this_thread::sleep_until(nextFrameTime);
			if(activeMode_ != InferMode::Off) Step(activeMode_);
		}
		PostQuitMessage(0);
	} catch(const std::exception& e){
		std::cerr << "Error: " << e.what() << std::endl;
		PostQuitMessage(1);
	}
	stop_ = true;
	Dispose();
	std::terminate();
}