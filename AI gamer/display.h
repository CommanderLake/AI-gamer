#pragma once
#include "common.h"
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")

class Display{
public:
	Display(int width, int height, WNDPROC windowProc);
	~Display();

	void ShowImage(const unsigned char* imageData) const;
	static void ShowKeyState(uint16_t keyStates, int32_t mouseDeltaX, int32_t mouseDeltaY);

private:
	int width, height;
	HWND hwnd;
	HDC hdc;
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	WNDPROC windowProc;

	void InitializeWindow();
	void InitializeGDIPlus();
};
