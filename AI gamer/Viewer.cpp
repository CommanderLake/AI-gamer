#include "Viewer.h"
#include <iostream>
#include <chrono>
Viewer::Viewer(WNDPROC windowProc) : windowProc(windowProc){}
Viewer::~Viewer(){
	Gdiplus::GdiplusShutdown(gdiplusToken);
	ReleaseDC(hwnd, hdc);
	DestroyWindow(hwnd);
}
void Viewer::InitializeWindow(int width, int height){
	const char CLASS_NAME[] = "ImageDisplayWindowClass";
	WNDCLASS wc = {};
	wc.lpfnWndProc = windowProc;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.lpszClassName = CLASS_NAME;
	RegisterClass(&wc);
	hwnd = CreateWindowEx(0, CLASS_NAME, "Image Viewer", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
	if(hwnd == nullptr){
		std::cerr << "Failed to create window!" << std::endl;
		exit(1);
	}
	ShowWindow(hwnd, SW_SHOW);
	hdc = GetDC(hwnd);
}
void Viewer::ShowImage(const unsigned char* imageData, int width, int height) const{
	Gdiplus::Bitmap bitmap(width, height, PixelFormat24bppRGB);
	Gdiplus::BitmapData bitmapData;
	const Gdiplus::Rect rect(0, 0, width, height);
	bitmap.LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat24bppRGB, &bitmapData);
	auto* pixels = static_cast<unsigned char*>(bitmapData.Scan0);
	//memcpy(pixels, imageData, width*height*3);
	const int planeSize = width * height; // Size of one color plane (R, G, or B)
	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
			const int index = y * width + x;
			pixels[(y * width + x) * 3] = imageData[index + 2 * planeSize]; // Blue
			pixels[(y * width + x) * 3 + 1] = imageData[index + planeSize]; // Green
			pixels[(y * width + x) * 3 + 2] = imageData[index]; // Red
		}
	}
	bitmap.UnlockBits(&bitmapData);
	Gdiplus::Graphics graphics(hdc);
	graphics.DrawImage(&bitmap, 0, 0, width, height);
}
void Viewer::ShowKeyState(uint16_t keyStates, int32_t mouseDeltaX, int32_t mouseDeltaY){
	// Clear the console
	ClearScreen();
	// Viewer the key states
	std::cout << "Key States:\n";
	std::cout << "Move forward (W): " << (keyStates & 1 ? "Pressed" : "Released") << "\n";
	std::cout << "Move left (A): " << (keyStates & 1 << 1 ? "Pressed" : "Released") << "\n";
	std::cout << "Move backward (S): " << (keyStates & 1 << 2 ? "Pressed" : "Released") << "\n";
	std::cout << "Move right (D): " << (keyStates & 1 << 3 ? "Pressed" : "Released") << "\n";
	std::cout << "Jump (Space): " << (keyStates & 1 << 4 ? "Pressed" : "Released") << "\n";
	std::cout << "Crouch (CTRL): " << (keyStates & 1 << 5 ? "Pressed" : "Released") << "\n";
	std::cout << "Melee (Q): " << (keyStates & 1 << 6 ? "Pressed" : "Released") << "\n";
	std::cout << "Reload (R): " << (keyStates & 1 << 7 ? "Pressed" : "Released") << "\n";
	std::cout << "Action (E): " << (keyStates & 1 << 8 ? "Pressed" : "Released") << "\n";
	std::cout << "Switch weapon (1): " << (keyStates & 1 << 9 ? "Pressed" : "Released") << "\n";
	std::cout << "Switch grenade (2): " << (keyStates & 1 << 10 ? "Pressed" : "Released") << "\n";
	std::cout << "Shoot (Mouse button 1): " << (keyStates & 1 << 11 ? "Pressed" : "Released") << "\n";
	std::cout << "Zoom in (Mouse button 2): " << (keyStates & 1 << 12 ? "Pressed" : "Released") << "\n";
	std::cout << "Throw grenade (Mouse button 3): " << (keyStates & 1 << 13 ? "Pressed" : "Released") << "\n";
	// Viewer the mouse movements
	std::cout << "Mouse Delta X: " << mouseDeltaX << "\n";
	std::cout << "Mouse Delta Y: " << mouseDeltaY << "\n";
}
void Viewer::Play(std::string fileName){
	std::ifstream file(fileName, std::ios::binary | std::ios::in);
	if(!file.is_open()){
		std::cerr << "Failed to open training data file!" << std::endl;
		return;
	}
	// Read width and height
	int width, height;
	file.read(reinterpret_cast<char*>(&width), sizeof width);
	file.read(reinterpret_cast<char*>(&height), sizeof height);
	const std::size_t stateSize = width*height*3;
	InitializeWindow(width, height);
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);
	uint16_t keyStates;
	int32_t mouseDeltaX;
	int32_t mouseDeltaY;
	const auto state_data = static_cast<unsigned char*>(_aligned_malloc(stateSize, 64));
	while(file.peek() != EOF){
		file.read(reinterpret_cast<char*>(&keyStates), sizeof keyStates);
		file.read(reinterpret_cast<char*>(&mouseDeltaX), sizeof mouseDeltaX);
		file.read(reinterpret_cast<char*>(&mouseDeltaY), sizeof mouseDeltaY);
		file.read(reinterpret_cast<char*>(state_data), stateSize);
		ShowKeyState(keyStates, mouseDeltaX, mouseDeltaY);
		ShowImage(state_data, width, height);
		MSG msg = {nullptr};
		while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)){
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
}