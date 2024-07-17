#pragma once
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
struct Task{
	const unsigned char* data;
	size_t size;
	size_t index;
};
struct OutputChunk{
	__declspec(align(32)) unsigned char* data = nullptr;
	volatile size_t size = 0;
	volatile size_t capacity = 0;
	void resize(size_t newsize){
		if(newsize == capacity) return;
		if(data != nullptr) _mm_free(data);
		data = static_cast<unsigned char*>(_mm_malloc(newsize, 32));
		capacity = newsize;
	}
};
class ZstdThreadPool{
public:
	int compLevel = 5;
	explicit ZstdThreadPool(int numThreads);
	~ZstdThreadPool();
	void dispatchTask(const unsigned char* inputData, size_t inputSize, size_t numChunks);
	std::vector<OutputChunk> results;
private:
	int nt;
	std::vector<std::thread> workers;
	std::vector<std::queue<Task>> taskQueues;
	size_t outstandingTasks;
	std::mutex completionMutex;
	std::condition_variable completionCondition;
	std::mutex **queueMutexes;
	std::condition_variable **conditions;
	bool shutdown;
	void enqueueTask(const Task& task, size_t threadIndex);
	void waitForCompletion();
	void signalTaskCompletion();
};