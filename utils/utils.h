#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <queue>

typedef std::priority_queue<std::pair<float, unsigned>> candidate_pool;

struct Timer {
    std::chrono::_V2::system_clock::time_point s;
    std::chrono::_V2::system_clock::time_point e;
    std::chrono::duration<double> diff;

    void tick() {
        s = std::chrono::high_resolution_clock::now();
    }

    void tuck(std::string message) {
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        std::cout << "[" << diff.count() << " s] " << message << std::endl;
    }
};

template <class T>
T* read_fbin(const char* filename, unsigned& n, unsigned& d) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char*)&n, 4);
    in.read((char*)&d, 4);
    T* data = new T[n * d];
    in.read((char*)data, (size_t)n * (size_t)d * 4);
    in.close();
    return data;
}