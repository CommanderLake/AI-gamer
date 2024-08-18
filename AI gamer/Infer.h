#pragma once
#include <atomic>
class Infer{
public:
	Infer();
	~Infer();
	static void ProcessOutput(const float* predictions);
	void ListenForKey();
	void Inference();
	std::atomic<bool> simInput = false;
	std::atomic<bool> stopInfer = false;
};