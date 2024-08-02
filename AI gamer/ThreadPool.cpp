#include "ThreadPool.h"
ThreadPool::ThreadPool(size_t numThreads) : stop(false), generators(numThreads){
	std::random_device rd;
	for(size_t i = 0; i < numThreads; ++i){
		generators[i].seed(rd());
		workers.emplace_back([this, i]{
			while(true){
				std::function<void()> task;
				{
					std::unique_lock<std::mutex> lock(this->queueMutex);
					this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
					if(this->stop && this->tasks.empty()) return;
					task = std::move(this->tasks.front());
					this->tasks.pop();
				}
				task();
			}
		});
	}
}
ThreadPool::~ThreadPool(){
	{
		std::unique_lock<std::mutex> lock(queueMutex);
		stop = true;
	}
	condition.notify_all();
	for(std::thread& worker : workers){ worker.join(); }
}
void ThreadPool::WaitAll(){
	for(auto& future : futures){ future.wait(); }
	futures.clear();
}
std::mt19937& ThreadPool::GetThreadGenerator(){
	// Get the thread index based on the current thread id
	auto it = std::find_if(workers.begin(), workers.end(), [](const std::thread& t){ return t.get_id() == std::this_thread::get_id(); });
	if(it != workers.end()){
		size_t index = std::distance(workers.begin(), it);
		return generators[index];
	}
	throw std::runtime_error("Thread not found in thread pool");
}