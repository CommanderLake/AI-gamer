#include "Infer.h"
#include "common.h"
#include "NN.h"
#include "NvDisplayCap.h"
#include <atomic>
Infer::Infer(){}
Infer::~Infer(){}
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
	const int mouseX = static_cast<int>(predictions[14]*256.0f);
	const int mouseY = static_cast<int>(predictions[15]*256.0f);
	if(mouseX != 0 || mouseY != 0){
		inputs[inputIndex].type = INPUT_MOUSE;
		inputs[inputIndex].mi.dx = mouseX;
		inputs[inputIndex].mi.dy = mouseY;
		inputs[inputIndex].mi.dwFlags = MOUSEEVENTF_MOVE;
		inputIndex++;
	}
	if(inputIndex > 0){ SendInput(inputIndex, inputs, sizeof(INPUT)); }
}
void Infer::ListenForKey(){
	std::cout << "Press F9 to start AI input and Escape to pause.\r\n";
	while(true){
		if(GetAsyncKeyState(VK_F9) & 0x8000){
			simInput = true;
			while(GetAsyncKeyState(VK_F9) & 0x8000){ Sleep(10); }
		}
		if(GetAsyncKeyState(VK_ESCAPE) & 0x8000){
			simInput = false;
			while(GetAsyncKeyState(VK_ESCAPE) & 0x8000){ Sleep(10); }
		}
		Sleep(10);
	}
}
void Infer::Inference(){
	InitCUDA();
	InitNvFBC();
	AllocGPU();
	const auto nn = new NN(0, 0, false);
	std::thread listenKey(&Infer::ListenForKey, this);
	listenKey.detach();
	while(!simInput){
		Sleep(1);
	}
	int capWidth = 0, capHeight = 0;
	float* hPredictionsF;
	checkCUDA(cudaMallocHost(&hPredictionsF, numCtrls_*sizeof(float)));
	float* dPredictionsF;
	checkCUDA(cudaMalloc(&dPredictionsF, numCtrls_*sizeof(float)));
	const auto inputSize = nn->inWidth_*nn->inHeight_*3;
	__half* frameHalf = nullptr;
	checkCUDA(cudaMalloc(&frameHalf, inputSize*sizeof(__half)));
	checkCUDA(cudaMemset(frameHalf, 0, inputSize*sizeof(__half)));
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	while(!stopInfer){
		while(!simInput){
			Sleep(1);
		}
		nextFrameTime += frameDuration;
		std::this_thread::sleep_until(nextFrameTime);
		const auto frame = GrabFrameInt8(&capWidth, &capHeight, true, false);
		if(capWidth != nn->inWidth_ || capHeight != nn->inHeight_){
			simInput = false;
			std::cerr << "Capture resolution mismatch, pausing AI input.\r\n";
			continue;
		}
		ConvertAndNormalize(frameHalf, frame, inputSize);
		const auto output = nn->Forward(frameHalf);
		ConvertHalfToFloat(output, dPredictionsF, numCtrls_);
		checkCUDA(cudaMemcpy(hPredictionsF, dPredictionsF, numCtrls_*sizeof(float), cudaMemcpyDeviceToHost));
		ProcessOutput(hPredictionsF);
	}
	DisposeNvFBC();
	cudaFreeHost(hPredictionsF);
	cudaFree(dPredictionsF);
	cudaFree(frameHalf);
}