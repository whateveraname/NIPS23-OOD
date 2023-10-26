#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>

#include "../utils/utils.h"
#include "../utils/dist_func.h"

struct Neighbor {
  unsigned id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(unsigned id, float distance, bool f)
      : id{id}, distance{distance}, flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance)
      right = mid;
    else
      left = mid;
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return K + 1;
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
  memmove((char *)&addr[right + 1], &addr[right],
          (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

struct IndexGraph {
    using DISTFUNC = float (*)(const void*, const void*, const void*);

    size_t dimension_;
    size_t nd_;
    unsigned width;
    size_t data_len;
    size_t neighbor_len;
    size_t node_size;
    float* data_;
    char* opt_graph_;
    size_t page_num;
    std::vector<std::vector<unsigned>> final_graph_;
    DISTFUNC distance_ = utils::InverseInnerProduct;

    IndexGraph(const size_t dimension, const size_t n): dimension_(dimension), nd_(n) {
        if (dimension == 200) {
            distance_ = utils::InnerProductFloatAVX512;
        } else {
            distance_ = utils::InnerProductFloatAVX512Dim20;
        }
    }

    ~IndexGraph() { delete[] opt_graph_; }

    void load_graph(const char* filename) {
        std::ifstream in(filename, std::ios::binary);
        in.ignore(4);
        in.read((char*)&width, sizeof(unsigned));
        width -= 1;
        for (size_t i = 0; i < nd_; i++) {
            unsigned k;
            in.read((char*)&k, sizeof(unsigned));
            if (k > width) {
                std::cout << k << "\n";
                abort();
            }
            std::vector<unsigned> tmp(width);
            in.read((char*)tmp.data(), width * sizeof(unsigned));
            tmp.resize(k);
            final_graph_.push_back(tmp);
        }
        in.close();
    }

    void optimizeGraph(int fd) {
        data_len = (dimension_) * sizeof(float);
        neighbor_len = (width + 1) * sizeof(unsigned);
        node_size = data_len + neighbor_len;
        opt_graph_ = (char*)malloc(node_size * nd_);
        for (unsigned i = 0; i < nd_; i++) {
            auto v = read_vector(fd, dimension_, i);
            char* cur_node_offset = opt_graph_ + i * node_size;
            std::memcpy(cur_node_offset, v.data(), data_len);
            cur_node_offset += data_len;
            unsigned k = final_graph_[i].size();
            std::memcpy(cur_node_offset, &k, sizeof(unsigned));
            std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                        k * sizeof(unsigned));
            std::vector<unsigned>().swap(final_graph_[i]);
        }
    }

    void save(const char* fn) {
        std::ofstream out(fn, std::ios::binary);
        out.write((char*)&data_len, 8);
        out.write((char*)&neighbor_len, 8);
        out.write((char*)&node_size, 8);
        out.write((char*)&page_num, 8);
        out.write(opt_graph_, node_size * nd_);
        out.close();
    }

    void load(const char* fn) {
        std::ifstream in(fn, std::ios::binary);
        in.read((char*)&data_len, 8);
        in.read((char*)&neighbor_len, 8);
        in.read((char*)&node_size, 8);
        in.read((char*)&page_num, 8);
        opt_graph_ = (char*)malloc(node_size * nd_);
        in.read(opt_graph_, node_size * nd_);
        in.close();
    }

    void searchWithOptGraph(const float* query, size_t K, unsigned L, unsigned* indices, int64_t* eps, unsigned nprobe) {
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0;

        for (; tmp_l < L && tmp_l < nprobe; tmp_l++) {
            init_ids[tmp_l] = eps[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id])
                continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_)
                continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_)
                continue;
            float* x = (float*)(opt_graph_ + node_size * id);
            float dist = distance_(x, query, NULL);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned* neighbors = (unsigned*)(opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id])
                        continue;
                    flags[id] = 1;
                    float* data = (float*)(opt_graph_ + node_size * id);
                    float dist = distance_(query, data, NULL);
                    if (dist >= retset[L - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    void searchWithOptGraphRestart(const float* query, size_t K, unsigned L, unsigned* indices, int64_t* eps, unsigned nprobe) {
        std::vector<Neighbor> retset(L + 1);
        std::vector<unsigned> init_ids(L);
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned tmp_l = 0, start_i = 0;

        init_ids[tmp_l] = eps[tmp_l];
        tmp_l++;
        start_i++;
        flags[init_ids[tmp_l]] = true;

        while (tmp_l < L) {
            unsigned id = rand() % nd_;
            if (flags[id])
                continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_)
                continue;
            _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_)
                continue;
            float* x = (float*)(opt_graph_ + node_size * id);
            float dist = distance_(x, query, NULL);
            retset[i] = Neighbor(id, dist, true);
            flags[id] = true;
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L && start_i < nprobe) {
            int nk = L;
            bool proceed = true;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                float lowerbound = retset[k].distance;

                proceed = false;

                _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
                unsigned* neighbors = (unsigned*)(opt_graph_ + node_size * n + data_len);
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id])
                        continue;
                    flags[id] = 1;
                    float* data = (float*)(opt_graph_ + node_size * id);
                    float dist = distance_(query, data, NULL);
                    if (dist < lowerbound) proceed = true;
                    if (dist >= retset[L - 1].distance)
                        continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
            if (!proceed) {
                for (size_t m = 0; m < L + 1; m++) {
                    retset[m].flag = false;
                }
                unsigned id;
                while (start_i < nprobe && flags[id = eps[start_i++]]) {}
                if (start_i == nprobe && flags[eps[nprobe - 1]]) break;
                flags[id] = 1;
                float* data = (float*)(opt_graph_ + node_size * id);
                float dist = distance_(query, data, NULL);
                k = InsertIntoPool(retset.data(), L, Neighbor(id, dist, true));
            }
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }
};