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
	static void ProcessMessages();
	static LRESULT WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void InitializeWindow(int width, int height);
	void ShowImage(const unsigned char* imageData, int width, int height) const;
	static void ShowKeyState(unsigned int keyStates, int mouseDeltaX, int mouseDeltaY);
	void Play(std::string fileName);
	HWND hwnd_;
	HDC hdc_;
	Gdiplus::GdiplusStartupInput gdiplusStartupInput_;
	ULONG_PTR gdiplusToken_;
	cudnnHandle_t cudnn_;
};