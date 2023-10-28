#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>

#include "../utils/utils.h"
#include "../utils/dist_func.h"

#include <faiss/impl/DistanceComputer.h>

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

struct IndexGraphSQ {
    using DISTFUNC = float (*)(const void*, const void*, const void*);

    size_t nd_;
    size_t dimension_;
    unsigned neighbor_len;
    std::vector<unsigned> graph_;
    uint8_t* codes_;
    size_t code_size_;
    DISTFUNC dist_ = utils::InnerProductFloatAVX512HpDim200;

    IndexGraphSQ(const size_t n): nd_(n) {}

    void load_graph(const char* filename) {
        std::ifstream in(filename, std::ios::binary);
        in.ignore(4);
        in.read((char*)&neighbor_len, sizeof(unsigned));
        graph_.resize(nd_ * neighbor_len);
        unsigned width = neighbor_len - 1;
        for (size_t i = 0; i < nd_; i++) {
            unsigned k;
            in.read((char*)&k, sizeof(unsigned));
            if (k > width) {
                std::cout << k << "\n";
                abort();
            }
            graph_[i * neighbor_len] = k;
            std::vector<unsigned> tmp(width);
            in.read((char*)tmp.data(), width * sizeof(unsigned));
            memcpy(graph_.data() + i * neighbor_len + 1, tmp.data(), width * 4);
        }
        in.close();
    }

    void set_storage(size_t d, uint8_t* codes, size_t code_size) {
        dimension_ = d;
        codes_ = codes;
        code_size_ = code_size;
        dist_ = (d == 200 ? utils::InnerProductFloatAVX512HpDim200 : utils::InnerProductFloatAVX512Hp);
    }

    void searchWithOptGraph(float* query, size_t K, unsigned L, unsigned* indices, int64_t* eps, unsigned nprobe) {
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
            _mm_prefetch(codes_ + code_size_ * id, _MM_HINT_T0);
        }
        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_)
                continue;
            float dist = dist_(query, codes_ + code_size_ * id, &dimension_);
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

                _mm_prefetch(graph_.data() + neighbor_len * n, _MM_HINT_T0);
                unsigned* neighbors = graph_.data() + neighbor_len * n;
                unsigned MaxM = *neighbors;
                neighbors++;
                for (unsigned m = 0; m < MaxM; ++m)
                    _mm_prefetch(codes_ + code_size_ * neighbors[m], _MM_HINT_T0);
                for (unsigned m = 0; m < MaxM; ++m) {
                    unsigned id = neighbors[m];
                    if (flags[id])
                        continue;
                    flags[id] = 1;
                    float dist = dist_(query, codes_ + code_size_ * id, &dimension_);
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
};