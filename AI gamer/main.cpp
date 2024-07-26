#include <cuda.h>
#include <iostream>
#include <mutex>
#include <Windows.h>
#include "display.h"
#include "input_recorder.h"
#include "training.h"
InputRecorder* recorder = nullptr;
NeuralNetwork* nn = nullptr;
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	switch(uMsg){
		case WM_INPUT: if(recorder != nullptr) recorder->ProcessRawInput(lParam);
			break;
		case WM_USER_START_CAPTURE: 
			if(recorder != nullptr){
				recorder->StartCapture();
				std::cout << "Capture enabled";
			}
			break;
		case WM_USER_STOP_CAPTURE: 
			if(recorder != nullptr){
				recorder->StopCapture();
				std::cout << "Capture disabled";
			}
			break;
		case WM_USER_CAPTURE_FRAME: 
			if(recorder != nullptr){
				recorder->WriteFrameData();
			}
			break;
		case WM_DESTROY: PostQuitMessage(0);
			return 0;
		default: return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	return 0;
}
int width = 640;
int height = 480;
InputRecord** trainingData = nullptr;
size_t trainingDataCount = 0;
std::mutex trainingDataMutex;
void ReadStateData(){
	std::ifstream file(trainDataFileName, std::ios::binary | std::ios::in);
	if(!file.is_open()){
		std::cerr << "Failed to open training data file!" << std::endl;
		return;
	}
	// Read width and height
	file.read(reinterpret_cast<char*>(&width), sizeof width);
	file.read(reinterpret_cast<char*>(&height), sizeof height);
	const std::size_t decompressedSize = width*height*3;
	// Step 1-4: Store file positions of each record
	std::vector<std::streampos> recordPositions;
	while(file.peek() != EOF){
		recordPositions.push_back(file.tellg());
		file.seekg(10 + decompressedSize, std::ios::cur);
		//if(recordPositions.size() >= 1024) break; // LIMIT FOR DEBUGGING!
	}
	file.close();
	// Step 5: Create trainingData array
	trainingDataCount = recordPositions.size();
	trainingData = new InputRecord*[trainingDataCount];
	// Step 6: Parallel processing with 8 threads
	auto processRecords = [&](size_t start, size_t end){
		std::ifstream threadFile(trainDataFileName, std::ios::binary | std::ios::in);
		if(!threadFile.is_open()){
			std::cerr << "Thread failed to open training data file!" << std::endl;
			return;
		}
		for(size_t i = start; i < end; ++i){
			threadFile.seekg(recordPositions[i]);
			const auto record = new InputRecord();
			threadFile.read(reinterpret_cast<char*>(&record->keyStates), sizeof record->keyStates);
			threadFile.read(reinterpret_cast<char*>(&record->mouseDeltaX), sizeof record->mouseDeltaX);
			threadFile.read(reinterpret_cast<char*>(&record->mouseDeltaY), sizeof record->mouseDeltaY);
			record->state_data = static_cast<unsigned char*>(_aligned_malloc(decompressedSize, 64));
			threadFile.read(reinterpret_cast<char*>(record->state_data), decompressedSize);
			std::lock_guard<std::mutex> guard(trainingDataMutex);
			trainingData[i] = record;
		}
	};
	// Create and join 8 threads
	std::vector<std::thread> threads;
	const size_t recordsPerThread = trainingDataCount / 8;
	for(size_t i = 0; i < 8; ++i){
		size_t start = i * recordsPerThread;
		size_t end = i == 7 ? trainingDataCount : start + recordsPerThread;
		threads.emplace_back(processRecords, start, end);
	}
	for(auto& thread : threads){ thread.join(); }
}
HWND MakeWindow(){
	const HINSTANCE hInstance = GetModuleHandle(nullptr);
	const char CLASS_NAME[] = "InputCaptureWindowClass";
	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASS_NAME;
	RegisterClass(&wc);
	const HWND hwnd = CreateWindowEx(0, // Optional window styles.
		CLASS_NAME, // Window class
		"Input Capture", // Window text
		WS_OVERLAPPEDWINDOW, // Window style
		// Size and position
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, nullptr, // Parent window    
		nullptr, // Menu
		hInstance, // Instance handle
		nullptr // Additional application data
	);
	ShowWindow(hwnd, SW_HIDE); // Hide the window
	return hwnd;
}
int main(){
	std::ios::sync_with_stdio(false);
	std::cout << "R for record mode, T for train mode, V for view mode, I for Infer mode... ";
	char mode;
	std::cin >> mode;
	std::cout << std::endl;
	if(mode == 'r' || mode == 'R'){
		const auto hwnd = MakeWindow();
		RAWINPUTDEVICE rid[2];
		rid[0].usUsagePage = 0x01;
		rid[0].usUsage = 0x06;
		rid[0].dwFlags = RIDEV_INPUTSINK;
		rid[0].hwndTarget = hwnd;
		rid[1].usUsagePage = 0x01;
		rid[1].usUsage = 0x02;
		rid[1].dwFlags = RIDEV_INPUTSINK;
		rid[1].hwndTarget = hwnd;
		if(RegisterRawInputDevices(rid, 2, sizeof rid[0]) == FALSE){
			return 0; // Registration failed
		}
		InitCUDA();
		recorder = new InputRecorder(hwnd);
		std::thread listenThread(&InputRecorder::ListenForKey, recorder);
		listenThread.detach();
		MSG msg = {};
		while(GetMessage(&msg, nullptr, 0, 0)){
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	} else if(mode == 't' || mode == 'T'){
		ReadStateData();
		nn = new NeuralNetwork();
		nn->Initialize(width, height, true);
		nn->Train(trainingData, trainingDataCount);
	} else if(mode == 'v' || mode == 'V'){
		ReadStateData();
		const auto display = new Display(width, height, WindowProc);
		//const auto frameDuration = std::chrono::milliseconds(1000 / 30);
		//auto nextFrameTime = std::chrono::steady_clock::now();
		for(auto i = 0; i < trainingDataCount; ++i){
			//nextFrameTime += frameDuration;
			// Process Windows messages to keep the window responsive
			MSG msg = {nullptr};
			while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)){
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			display->ShowKeyState(trainingData[i]->keyStates, trainingData[i]->mouseDeltaX, trainingData[i]->mouseDeltaY);
			display->ShowImage(trainingData[i]->state_data);
			//auto end = std::chrono::steady_clock::now();
			//auto sleep_duration = nextFrameTime - end;
			//if(sleep_duration.count() > 0){ std::this_thread::sleep_until(nextFrameTime); } else{ nextFrameTime = end; }
		}
	} else if(mode == 'i' || mode == 'I'){
		const auto hwnd = MakeWindow();
		RAWINPUTDEVICE rid[1];
		rid[0].usUsagePage = 0x01; // HID_USAGE_PAGE_GENERIC
		rid[0].usUsage = 0x06;     // HID_USAGE_GENERIC_KEYBOARD
		rid[0].dwFlags = RIDEV_INPUTSINK;
		rid[0].hwndTarget = hwnd;
		if(!RegisterRawInputDevices(rid, 1, sizeof(rid[0]))){
			MessageBox(hwnd, "Failed to register raw input device.", "Error", MB_OK);
		}
		nn = new NeuralNetwork();
		nn->Infer();
	}
	return 0;
}