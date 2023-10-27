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
    HugepageAllocation alloc;

    IndexGraph(const size_t dimension, const size_t n): dimension_(dimension), nd_(n) {
        if (dimension == 200) {
            distance_ = utils::InnerProductFloatAVX512;
        } else {
            distance_ = utils::InnerProductFloatAVX512Dim20;
        }
    }

    ~IndexGraph() { 
        // delete[] opt_graph_; 
        hugepage_unmap(alloc.ptr, alloc.sz);
    }

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
        // opt_graph_ = (char*)malloc(node_size * nd_);
        alloc = hugepage_mmap(node_size * nd_);
        opt_graph_ = (char*)alloc.ptr;
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

    void searchWithOptGraph1(const float* query, size_t K, unsigned L, unsigned* indices, int64_t* eps, unsigned nprobe) {
        boost::dynamic_bitset<> visited{nd_, 0};
        candidate_pool top_candidates;
        candidate_pool candidate_set;

        for (size_t i = 0; i < nprobe; i++) {
            auto ep = eps[i];
            float dist = distance_(query, opt_graph_ + node_size * ep, NULL);
            top_candidates.emplace(dist, ep);
            candidate_set.emplace(-dist, ep);
            visited[ep] = true;
        }

        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            candidate_set.pop();
            if ((-current_node_pair.first) > top_candidates.top().first && top_candidates.size() == L)
                break;
            _mm_prefetch(opt_graph_ + node_size * current_node_pair.second + data_len, _MM_HINT_T0);
            unsigned* neighbors = (unsigned*)(opt_graph_ + node_size * current_node_pair.second + data_len);
            unsigned MaxM = *neighbors;
            neighbors++;
            for (size_t m = 0; m < MaxM; m++) {
                _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
            }
            for (size_t m = 0; m < MaxM; m++) {
                unsigned candidate_id = neighbors[m];
                if (visited[candidate_id])
                    continue;
                visited[candidate_id] = true;
                float dist = distance_(query, opt_graph_ + node_size * candidate_id, NULL);
                if (top_candidates.size() < L || top_candidates.top().first > dist) {
                    candidate_set.emplace(-dist, candidate_id);
                    top_candidates.emplace(dist, candidate_id);
                    if (top_candidates.size() > L)
                        top_candidates.pop();
                }
            }
        }
        while (top_candidates.size() > K) {
            top_candidates.pop();
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = top_candidates.top().second;
            top_candidates.pop();
        }
    }

    void searchWithOptGraphRestart(const float* query, size_t K, unsigned L, unsigned* indices, int64_t* eps, unsigned nprobe) {
        boost::dynamic_bitset<> visited{nd_, 0};
        candidate_pool top_candidates;
        candidate_pool candidate_set;

        unsigned start_i = nprobe;

        auto ep = eps[start_i--];
        float dist = distance_(query, opt_graph_ + node_size * ep, NULL);
        top_candidates.emplace(dist, ep);
        candidate_set.emplace(-dist, ep);
        visited[ep] = true;

        while (!candidate_set.empty() && start_i >= 0) {
            auto current_node_pair = candidate_set.top();
            candidate_set.pop();
            if ((-current_node_pair.first) > top_candidates.top().first && top_candidates.size() == L)
                break;
            _mm_prefetch(opt_graph_ + node_size * current_node_pair.second + data_len, _MM_HINT_T0);
            unsigned* neighbors = (unsigned*)(opt_graph_ + node_size * current_node_pair.second + data_len);
            unsigned MaxM = *neighbors;
            neighbors++;
            for (size_t m = 0; m < MaxM; m++) {
                _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
            }
            bool proceed = false;
            for (size_t m = 0; m < MaxM; m++) {
                unsigned candidate_id = neighbors[m];
                if (visited[candidate_id])
                    continue;
                visited[candidate_id] = true;
                float dist = distance_(query, opt_graph_ + node_size * candidate_id, NULL);
                if (dist < -current_node_pair.first) proceed = true;
                if (top_candidates.size() < L || top_candidates.top().first > dist) {
                    candidate_set.emplace(-dist, candidate_id);
                    top_candidates.emplace(dist, candidate_id);
                    if (top_candidates.size() > L)
                        top_candidates.pop();
                }
            }
            if (!proceed) {
                candidate_set = candidate_pool();
                unsigned ep;
                while (start_i >= 0 && visited[eps[start_i]]) start_i--;
                if (start_i >= nprobe) {
                    candidate_set.emplace(-distance_(query, opt_graph_ + node_size * eps[start_i], NULL), eps[start_i]);
                    start_i--;
                } else {
                    break;
                }
            }
        }

        while (top_candidates.size() > K) {
            top_candidates.pop();
        }
        for (size_t i = 0; i < K; i++) {
            indices[i] = top_candidates.top().second;
            top_candidates.pop();
        }
    }
};

struct IndexGraphSQ {
    using DISTFUNC = float (*)(const void*, const void*, const void*);

    size_t nd_;
    size_t dimension_;
    unsigned neighbor_len;
    std::vector<unsigned> graph_;
    uint8_t* codes_;
    size_t code_size_;
    DISTFUNC dist_;

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