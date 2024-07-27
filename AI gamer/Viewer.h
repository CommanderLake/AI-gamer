#pragma once
#include "common.h"
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")

class Viewer{
public:
	Viewer(WNDPROC windowProc);
	~Viewer();

	void ShowImage(const unsigned char* imageData, int width, int height) const;
	static void ShowKeyState(uint16_t keyStates, int32_t mouseDeltaX, int32_t mouseDeltaY);
	void Play(std::string fileName);
private:
	HWND hwnd;
	HDC hdc;
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	WNDPROC windowProc;

	void InitializeWindow(int width, int height);
};
