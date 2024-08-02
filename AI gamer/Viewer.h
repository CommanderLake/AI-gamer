#pragma once
#include "common.h"
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
class Viewer{
public:
	Viewer(WNDPROC windowProc);
	~Viewer();
	void InitializeWindow(int width, int height);
	void ShowImage(const unsigned char* imageData, int width, int height) const;
	static void ShowKeyState(unsigned short keyStates, int mouseDeltaX, int mouseDeltaY);
	void Play(std::string fileName);
	HWND hwnd;
	HDC hdc;
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	WNDPROC windowProc;
};