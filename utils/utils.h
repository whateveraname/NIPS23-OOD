#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <queue>
#include <array>
#include <fcntl.h>
#include <filesystem>
#include <linux/mman.h>
#include <memory>

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
    auto data = new T[n * d];
    in.read((char*)data, (size_t)n * (size_t)d * 4);
    in.close();
    return data;
}

std::vector<float> read_vector(int fd, unsigned d, unsigned i) {
    std::vector<float> v(d);
    pread(fd, v.data(), d * 4, 8 + (size_t)i * (size_t)d * 4);
    return v;
}

size_t div_round_up(size_t x, size_t y) {
    return (x / y) + static_cast<size_t>((x % y) != 0);
}

struct HugepageX86Parameters {
    constexpr HugepageX86Parameters(size_t pagesize, int mmap_flags)
        : pagesize{pagesize}
        , mmap_flags{mmap_flags} {};

    size_t pagesize;
    int mmap_flags;

    friend bool operator==(HugepageX86Parameters l, HugepageX86Parameters r) {
        return l.pagesize == r.pagesize && l.mmap_flags == r.mmap_flags;
    }
};

static constexpr std::array<HugepageX86Parameters, 2> hugepage_x86_options{
    // HugepageX86Parameters{1 << 30, MAP_HUGETLB | MAP_HUGE_1GB},
    HugepageX86Parameters{1 << 21, MAP_HUGETLB | MAP_HUGE_2MB},
    HugepageX86Parameters{1 << 12, 0},
};
struct HugepageAllocation {
    void* ptr;
    size_t sz;
};

[[nodiscard]] inline HugepageAllocation hugepage_mmap(size_t bytes, bool force = false) {
    assert(bytes != 0);
    void* ptr = MAP_FAILED;
    size_t sz = 0;
    for (auto params : hugepage_x86_options) {
        auto pagesize = params.pagesize;
        auto flags = params.mmap_flags;
        sz = pagesize * div_round_up(bytes, pagesize);
        ptr = mmap(
            nullptr,
            sz,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | flags,
            -1,
            0
        );

        if (ptr != MAP_FAILED) {
            std::cout << "mapped " << pagesize / 1024 << "Kb page\n";
            break;
        }
    }

    if (ptr == MAP_FAILED) {
        abort();
    }
    return HugepageAllocation{.ptr = ptr, .sz = sz};
}

[[nodiscard]] inline bool hugepage_unmap(void* ptr, size_t sz) {
    return (munmap(ptr, sz) == 0);
}