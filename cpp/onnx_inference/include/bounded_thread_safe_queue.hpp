// BoundedThreadSafeQueue.hpp
#ifndef BOUNDED_THREAD_SAFE_QUEUE_HPP
#define BOUNDED_THREAD_SAFE_QUEUE_HPP

/**
 * @file BoundedThreadSafeQueue.hpp
 * @brief A thread-safe bounded queue implementation.
 * 
 * This class implements a bounded thread-safe queue with support for 
 * enqueueing and dequeueing items.
 */

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>


template<typename T>
class BoundedThreadSafeQueue {
public:
    BoundedThreadSafeQueue(size_t max_size) : _maxSize(max_size), _finished(false) {
        std::cout << "BoundedThreadSafeQueue initialized with max size: " << _maxSize << std::endl;
    }

    bool enqueue(T item) {
        std::unique_lock<std::mutex> lock(m);
        _cvNotFull.wait(lock, [this]() { return _queue.size() < _maxSize || _finished; });
        if (_finished) return false;

        _queue.push(std::move(item));
        // std::cout << "Enqueued item, current queue size: " << _queue.size() << std::endl;
        _cvNotEmpty.notify_one();
        return true;
    }

    bool dequeue(T& item) {
        std::unique_lock<std::mutex> lock(m);
        _cvNotEmpty.wait(lock, [this]() { return !_queue.empty() || _finished; });
        if (_queue.empty()) return false;

        item = std::move(_queue.front());
        _queue.pop();
        // std::cout << "Dequeued item, current queue size: " << _queue.size() << std::endl;
        _cvNotFull.notify_one();
        return true;
    }

    // 尝试将一个元素添加到队列中，如果队列已满，则移除最旧的元素
    bool enqueueWithEviction(T item) {
        std::unique_lock<std::mutex> lock(m);
        if (_queue.size() >= _maxSize) {
            _queue.pop();  // 移除最旧的元素
        }
        _queue.push(std::move(item));
        _cvNotEmpty.notify_one();
        return true;
    }

    // 读取队列中的元素但不移除它们
    bool peek(T& item) {
        std::unique_lock<std::mutex> lock(m);
        _cvNotEmpty.wait(lock, [this]() { return !_queue.empty() || _finished; });
        if (_queue.empty() || _finished) {
            std::cout << "Queue is empty." << std::endl;
            return false;
        }

        item = _queue.front();
        return true;
    }

    void setFinished() {
        std::unique_lock<std::mutex> lock(m);
        _finished = true;
        std::cout << "Queue marked as finished." << std::endl;
        _cvNotEmpty.notify_all();
        _cvNotFull.notify_all();
    }

private:
    std::queue<T> _queue;
    size_t _maxSize;
    std::mutex m;
    std::condition_variable _cvNotEmpty;
    std::condition_variable _cvNotFull;
    bool _finished;
};

#endif // BOUNDED_THREAD_SAFE_QUEUE_HPP
