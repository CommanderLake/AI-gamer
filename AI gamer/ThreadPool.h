#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <functional>
#include <future>
#include <condition_variable>
#include <random>

class ThreadPool{
public:
	explicit ThreadPool(size_t numThreads);
	~ThreadPool();
	template <class F, class... Args> std::future<std::result_of_t<F(Args ...)>> Enqueue(F&& f, Args&&... args);
	void WaitAll();
	std::mt19937& GetThreadGenerator();
private:
	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;
	std::vector<std::shared_future<void>> futures;
	std::mutex queueMutex;
	std::condition_variable condition;
	bool stop;
	std::vector<std::mt19937> generators; // Vector of random number generators
};

template <class F, class... Args>
std::future<std::result_of_t<F(Args ...)>> ThreadPool::Enqueue(F&& f, Args&&... args){
	using return_type = std::result_of_t<F(Args ...)>;
	auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
	std::future<return_type> res = task->get_future();
	{
		std::unique_lock<std::mutex> lock(queueMutex);
		if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
		tasks.emplace([task](){ (*task)(); });
	}
	condition.notify_one();
	futures.emplace_back(res.share());
	return res;
}