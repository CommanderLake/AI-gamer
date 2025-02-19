#pragma once
#include "common.h"
#include <windows.h>
#include <gdiplus.h>
#include "ConvScale.h"
#pragma comment(lib, "gdiplus.lib")
class Viewer{
public:
	Viewer();
	~Viewer();
	void ProcessMessages(const int width, const int height, const char* windowTitle);
	static LRESULT WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void InitializeWindow(const int width, const int height, const char* windowTitle);
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