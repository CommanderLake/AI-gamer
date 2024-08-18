#include "Train.h"
#include "Infer.h"
#include "Viewer.h"
#include "input_recorder.h"
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <Windows.h>
InputRecorder* recorder = nullptr;
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	switch(uMsg){
		case WM_INPUT: if(recorder != nullptr) recorder->ProcessRawInput(lParam);
			break;
		case WM_USER_START_CAPTURE: 
			if(recorder != nullptr){
				recorder->StartCapture();
				std::cout << "Capture enabled\r\n";
			}
			break;
		case WM_USER_STOP_CAPTURE: 
			if(recorder != nullptr){
				recorder->StopCapture();
				std::cout << "Capture disabled\r\n";
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
std::mutex fileRecordPositionsMutex;
size_t totalStateCount = 0;
void ReadStateData(int *width, int *height){
	for(const auto& fileName : trainDataInFiles){
		std::ifstream file(fileName, std::ios::binary | std::ios::in);
		if(!file.is_open()){
			std::cerr << "Failed to open training data file: " << fileName << std::endl;
			continue;
		}
		// Read width and height
		file.read(reinterpret_cast<char*>(width), sizeof *width);
		file.read(reinterpret_cast<char*>(height), sizeof *height);
		stateSize = *width**height*3;
		// Store file positions of each record
		std::vector<std::streampos> recordPositions;
		while(file.peek() != EOF){
			recordPositions.push_back(file.tellg());
			file.seekg(10 + stateSize, std::ios::cur);
		}
		file.close();
		std::lock_guard<std::mutex> lock(fileRecordPositionsMutex);
		totalStateCount += recordPositions.size();
		fileRecordIndex[fileName] = recordPositions;
	}
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
	std::cout << std::fixed << std::setprecision(8);
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
		recorder = new InputRecorder(hwnd);
		std::thread listenThread(&InputRecorder::ListenForKey, recorder);
		listenThread.detach();
		MSG msg = {};
		while(GetMessage(&msg, nullptr, 0, 0)){
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	} else if(mode == 't' || mode == 'T'){
		int width, height;
		ReadStateData(&width, &height);
		const auto nn = new Train();
		//auto viewer = new Viewer(WindowProc);
		nn->TrainModel(totalStateCount, width, height, nullptr);
	} else if(mode == 'v' || mode == 'V'){
		std::cout << "Training data file: ";
		std::string fileName;
		std::cin >> fileName;
		const auto viewer = new Viewer(WindowProc);
		viewer->Play(fileName);
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
		const auto nn = new Infer();
		nn->Inference();
	}
	return 0;
}