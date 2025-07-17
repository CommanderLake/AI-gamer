#include "Viewer.h"
#include "CuCommon.cuh"
#include <sstream>
#include <iostream>
#include <fstream>
LRESULT CALLBACK Viewer::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	switch(uMsg){
		case WM_DESTROY: PostQuitMessage(0);
			return 0;
		default: return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
}
Viewer::Viewer() : hwnd_(nullptr), hdc_(nullptr), gdiplusToken_(0){
	InitCUDA();
	cudnnCreate(&cudnn_);
}
Viewer::~Viewer(){
	if(bitmap_) delete bitmap_;
	if(bitmapData_) delete bitmapData_;
	Gdiplus::GdiplusShutdown(gdiplusToken_);
	ReleaseDC(hwnd_, hdc_);
	DestroyWindow(hwnd_);
}
void Viewer::ProcessMessages(const int width, const int height, const char* windowTitle){
	GdiplusStartup(&gdiplusToken_, &gdiplusStartupInput_, nullptr);
	constexpr char className[] = "ViewerWindowClass";
	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.lpszClassName = className;
	RegisterClass(&wc);
	constexpr DWORD windowStyle = WS_OVERLAPPEDWINDOW;
	RECT adjustedRect = {0, 0, width, height};
	AdjustWindowRect(&adjustedRect, windowStyle, FALSE);
	const int adjustedWidth = adjustedRect.right-adjustedRect.left;
	const int adjustedHeight = adjustedRect.bottom-adjustedRect.top;
	hwnd_ = CreateWindowEx(0, className, windowTitle, windowStyle, CW_USEDEFAULT, CW_USEDEFAULT, adjustedWidth, adjustedHeight, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
	if(hwnd_==nullptr){
		std::cerr<<"Failed to create window!\n";
		exit(1);
	}
	ShowWindow(hwnd_, SW_SHOW);
	hdc_ = GetDC(hwnd_);
	if(bitmap_) delete bitmap_;
	bitmap_ = new Gdiplus::Bitmap(width, height, PixelFormat32bppARGB);
	if(bitmapData_) delete bitmapData_;
	bitmapData_ = new Gdiplus::BitmapData();
	MSG msg;
	while(GetMessage(&msg, nullptr, 0, 0)){
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}
void Viewer::InitializeWindow(const int width, const int height, const char* windowTitle){
	std::thread messageLoop([&, this]{ ProcessMessages(width, height, windowTitle); });
	messageLoop.detach();
}
void Viewer::ShowImageRGB(const unsigned char* imageData, const int width, const int height){
	if(!bitmap_ || !bitmapData_ || !hdc_) return;
	const Gdiplus::Rect rect(0, 0, width, height);
	bitmap_->LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat32bppARGB, bitmapData_);
	auto* pixels = static_cast<unsigned char*>(bitmapData_->Scan0);
	const int planeSize = width*height;
	const int stride = bitmapData_->Stride;
	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
			const int index = y*width + x;
			const int pixelOffset = y*stride + x*4;
			pixels[pixelOffset] = imageData[index + 2*planeSize];
			pixels[pixelOffset + 1] = imageData[index + planeSize];
			pixels[pixelOffset + 2] = imageData[index];
			pixels[pixelOffset + 3] = 255;
		}
	}
	bitmap_->UnlockBits(bitmapData_);
	Gdiplus::Graphics graphics(hdc_);
	graphics.DrawImage(bitmap_, 0, 0, width, height);
}
void Viewer::ShowImageGreyscale(const unsigned char* imageData, const int width, const int height){
	if(!bitmap_||!bitmapData_||!hdc_) return;
	const Gdiplus::Rect rect(0, 0, width, height);
	if(bitmap_->LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat32bppARGB, bitmapData_)!=Gdiplus::Ok) return;
	auto* pixels = static_cast<unsigned char*>(bitmapData_->Scan0);
	const int stride = bitmapData_->Stride;
	for(int y = 0; y<height; ++y){
		unsigned char* row = pixels+y*stride;
		const unsigned char* src = imageData+y*width;
		for(int x = 0; x<width; ++x){
			const unsigned char value = src[x];
			row[x*4] = value;
			row[x*4+1] = value;
			row[x*4+2] = value;
			row[x*4+3] = 255;
		}
	}
	bitmap_->UnlockBits(bitmapData_);
	Gdiplus::Graphics graphics(hdc_);
	graphics.DrawImage(bitmap_, 0, 0, width, height);
}
const std::string DOWN = "1";
const std::string UP = "0";
std::ostringstream output;
void Viewer::ShowKeyState(const unsigned int keyStates, const int mouseDeltaX, const int mouseDeltaY){
	output.str("");
	//if(keyStates & 1) DebugBreak();
	output << "Move forward (W): " << (keyStates & 1 ? DOWN : UP) << "\n";
	output << "Move left (A): " << (keyStates & 1 << 1 ? DOWN : UP) << "\n";
	output << "Move backward (S): " << (keyStates & 1 << 2 ? DOWN : UP) << "\n";
	output << "Move right (D): " << (keyStates & 1 << 3 ? DOWN : UP) << "\n";
	output << "Jump (Space): " << (keyStates & 1 << 4 ? DOWN : UP) << "\n";
	output << "Crouch (CTRL): " << (keyStates & 1 << 5 ? DOWN : UP) << "\n";
	output << "Melee (Q): " << (keyStates & 1 << 6 ? DOWN : UP) << "\n";
	output << "Reload (R): " << (keyStates & 1 << 7 ? DOWN : UP) << "\n";
	output << "Action (E): " << (keyStates & 1 << 8 ? DOWN : UP) << "\n";
	output << "Switch weapon (1): " << (keyStates & 1 << 9 ? DOWN : UP) << "\n";
	output << "Switch grenade (2): " << (keyStates & 1 << 10 ? DOWN : UP) << "\n";
	output << "Shoot (Mouse button 1): " << (keyStates & 1 << 11 ? DOWN : UP) << "\n";
	output << "Zoom in (Mouse button 2): " << (keyStates & 1 << 12 ? DOWN : UP) << "\n";
	output << "Throw grenade (Mouse button 3): " << (keyStates & 1 << 13 ? DOWN : UP) << "\n";
	output << "Mouse Delta X: " << mouseDeltaX << "\n";
	output << "Mouse Delta Y: " << mouseDeltaY << "\n";
	ClearScreen();
	std::cout << output.str();
}
void Viewer::Play(std::string fileName){
	std::ifstream file(fileName, std::ios::binary | std::ios::in);
	if(!file.is_open()){
		std::cerr << "Failed to open training data file!" << std::endl;
		return;
	}
	int width, height;
	file.read(reinterpret_cast<char*>(&width), sizeof width);
	file.read(reinterpret_cast<char*>(&height), sizeof height);
	const std::size_t stateSize = width*height*3;
	InitializeWindow(width, height, fileName.c_str());
	unsigned int keyStates;
	int mouseDeltaX;
	int mouseDeltaY;
	const auto stateData = static_cast<unsigned char*>(_mm_malloc(stateSize, 64));
	constexpr std::chrono::microseconds frameDuration(33333);
	auto nextFrameTime = std::chrono::high_resolution_clock::now();
	while(file.peek() != EOF){
		auto currentTime = std::chrono::high_resolution_clock::now();
		nextFrameTime += frameDuration;
		if(currentTime > nextFrameTime) nextFrameTime = currentTime + frameDuration;
		std::this_thread::sleep_until(nextFrameTime);
		file.read(reinterpret_cast<char*>(&keyStates), sizeof keyStates);
		file.read(reinterpret_cast<char*>(&mouseDeltaX), sizeof mouseDeltaX);
		file.read(reinterpret_cast<char*>(&mouseDeltaY), sizeof mouseDeltaY);
		file.read(reinterpret_cast<char*>(stateData), stateSize);
		ShowKeyState(keyStates, mouseDeltaX, mouseDeltaY);
		ShowImageRGB(stateData, width, height);
	}
	_mm_free(stateData);
}
//void Viewer::Play(std::string fileName){
//	std::ifstream file(fileName, std::ios::binary|std::ios::in);
//	if(!file.is_open()){
//		std::cerr<<"Failed to open training data file!"<<std::endl;
//		return;
//	}
//	int width, height;
//	file.read(reinterpret_cast<char*>(&width), sizeof width);
//	file.read(reinterpret_cast<char*>(&height), sizeof height);
//	const std::size_t stateSize = width*height*3;
//	constexpr int batchSize = 80;
//	convScale = new ConvScale(cudnn_, 2, batchSize, 3, &height, &width);
//	const std::size_t newStateSize = width*height*3;
//	std::ofstream outputFile_;
//	outputFile_.open(trainDataOutFileName, std::ios::binary);
//	if(!outputFile_.is_open()){ throw std::runtime_error("Failed to open output file"); }
//	outputFile_.write(reinterpret_cast<char*>(&width), sizeof width);
//	outputFile_.write(reinterpret_cast<char*>(&height), sizeof height);
//	std::vector<uint16_t> keyStatesBatch(batchSize);
//	std::vector<int32_t> mouseDeltaXBatch(batchSize);
//	std::vector<int32_t> mouseDeltaYBatch(batchSize);
//	unsigned char* batchData = static_cast<unsigned char*>(_aligned_malloc(batchSize*stateSize, 64));
//	while(file.peek()!=EOF){
//		int actualBatchSize = 0;
//		// Read a batch of frames
//		for(int i = 0; i<batchSize; ++i){
//			if(file.peek()==EOF) break;
//			file.read(reinterpret_cast<char*>(&keyStatesBatch[i]), sizeof keyStatesBatch[i]);
//			file.read(reinterpret_cast<char*>(&mouseDeltaXBatch[i]), sizeof mouseDeltaXBatch[i]);
//			file.read(reinterpret_cast<char*>(&mouseDeltaYBatch[i]), sizeof mouseDeltaYBatch[i]);
//			file.read(reinterpret_cast<char*>(batchData+i*stateSize), stateSize);
//			actualBatchSize++;
//		}
//		// Process batch
//		if(actualBatchSize>0){
//			convScale->ScaleInPlace(batchData, true);
//			// Write processed batch to output file
//			for(int i = 0; i<actualBatchSize; ++i){
//				outputFile_.write(reinterpret_cast<char*>(&keyStatesBatch[i]), sizeof keyStatesBatch[i]);
//				outputFile_.write(reinterpret_cast<char*>(&mouseDeltaXBatch[i]), sizeof mouseDeltaXBatch[i]);
//				outputFile_.write(reinterpret_cast<char*>(&mouseDeltaYBatch[i]), sizeof mouseDeltaYBatch[i]);
//				outputFile_.write(reinterpret_cast<char*>(batchData+i*newStateSize), newStateSize);
//			}
//		}
//	}
//	_aligned_free(batchData);
//}