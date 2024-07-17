#include "ZstdThreadPool.h"
#include "zstd.h"
ZstdThreadPool::ZstdThreadPool(int numThreads) : outstandingTasks(0), shutdown(false){
	nt = numThreads;
	taskQueues.resize(numThreads);
	results.resize(numThreads);
	queueMutexes = static_cast<std::mutex**>(malloc(numThreads * sizeof(std::mutex*)));
	conditions = static_cast<std::condition_variable**>(malloc(numThreads * sizeof(std::condition_variable*)));
	for(int i = 0; i < numThreads; ++i){
		queueMutexes[i] = new std::mutex();
		conditions[i] = new std::condition_variable();
	}
	for(size_t i = 0; i < numThreads; ++i){
		workers.emplace_back([this, i]{
			ZSTD_CCtx* cctx = ZSTD_createCCtx();
			while(true){
				Task task;
				{
					std::unique_lock<std::mutex> lock(*queueMutexes[i]);
					conditions[i]->wait(lock, [this, i]{ return !taskQueues[i].empty() || shutdown; });
					if(shutdown && taskQueues[i].empty()) break;
					task = taskQueues[i].front();
					taskQueues[i].pop();
				}
				results[task.index].size = ZSTD_compressCCtx(cctx, results[task.index].data, results[task.index].capacity, task.data, task.size, compLevel);
				signalTaskCompletion();
			}
			ZSTD_freeCCtx(cctx);
		});
	}
}
ZstdThreadPool::~ZstdThreadPool(){
	{
		for(size_t i = 0; i < workers.size(); ++i){
			std::unique_lock<std::mutex> lock(*queueMutexes[i]);
		}
		shutdown = true;
	}
	for(int i = 0; i < nt; ++i){
		conditions[i]->notify_all();
	}
	for(std::thread& worker : workers){
		if(worker.joinable()){
			worker.join();
		}
	}
	for(int i = 0; i < nt; ++i){
		delete queueMutexes[i];
		delete conditions[i];
	}
	free(queueMutexes);
	free(conditions);
	for(auto i = 0; i < results.size(); ++i){
		_mm_free(results[i].data);
	}
}
void ZstdThreadPool::dispatchTask(const unsigned char* inputData, size_t inputSize, size_t numChunks){
	// Reset buffers and result sizes for new task
	for(auto& buffer : results){
		buffer.resize(ZSTD_compressBound(inputSize/numChunks));
	}
	outstandingTasks = numChunks;
	const size_t baseChunkSize = inputSize / numChunks;
	const size_t remainingBytes = inputSize % numChunks;
	for(size_t i = 0; i < numChunks; ++i){
		size_t chunkSize = baseChunkSize;
		if(i < remainingBytes){ // Distribute the remainder among the first few chunks
			++chunkSize;
		}
		const size_t offset = i * baseChunkSize + std::min(i, remainingBytes);
		enqueueTask({inputData + offset, chunkSize, i}, i);
	}
	// Wait for all tasks to complete
	waitForCompletion();
}
void ZstdThreadPool::enqueueTask(const Task& task, size_t threadIndex){
	// This function should now accept a threadIndex parameter to specify which thread's queue to use
	std::unique_lock<std::mutex> lock(*queueMutexes[threadIndex]);
	taskQueues[threadIndex].push(task);
	conditions[threadIndex]->notify_one();
}
void ZstdThreadPool::waitForCompletion(){
	std::unique_lock<std::mutex> lock(completionMutex);
	completionCondition.wait(lock, [this]{ return outstandingTasks == 0; });
}
void ZstdThreadPool::signalTaskCompletion(){
	{
		std::unique_lock<std::mutex> lock(completionMutex);
		--outstandingTasks;
	}
	completionCondition.notify_one();
}