#pragma once
#include "ConvScale.h"
#include <windows.h>
#include <gdiplus.h>
#include <string>
#pragma comment(lib, "gdiplus.lib")
class Viewer{
public:
	Viewer();
	~Viewer();
	void ProcessMessages(int width, int height, const char* windowTitle);
	static LRESULT WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void InitializeWindow(int width, int height, const char* windowTitle);
	void ShowImageRGB(const unsigned char* imageData, int width, int height);
	void ShowImageGreyscale(const unsigned char* imageData, int width, int height);
	static void ShowKeyState(unsigned int keyStates, int mouseDeltaX, int mouseDeltaY);
	void Play(std::string fileName);
	cudnnHandle_t cudnn_;
	HWND hwnd_;
	HDC hdc_;
	Gdiplus::GdiplusStartupInput gdiplusStartupInput_;
	ULONG_PTR gdiplusToken_;
	Gdiplus::Bitmap* bitmap_ = nullptr;
	Gdiplus::BitmapData* bitmapData_ = nullptr;
};