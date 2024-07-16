#pragma once
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <string>
inline std::string GetErrorMessage(HRESULT hr){
	LPVOID messageBuffer = nullptr;
	const size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		reinterpret_cast<LPSTR>(&messageBuffer), 0, nullptr);
	std::string message(static_cast<LPSTR>(messageBuffer), size);
	LocalFree(messageBuffer);
	return message;
}