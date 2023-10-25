#include <iostream>
#include <vector>
#include <limits>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <queue>
// #include <faiss/utils/Heap.h>
// #include <faiss/utils/distances.h>
// #include <boost/dynamic_bitset.hpp>
#include "../utils/utils.h"
#include "../utils/dist_func.h"
#include <omp.h>

using DISTFUNC = float (*)(const void *, const void *, const void *);

struct IndexIVF {
    size_t d;
    unsigned cluster_num;
    std::vector<std::vector<unsigned>> inverted_list_;
    std::vector<std::vector<float>> centroids_;
    unsigned n_;
    float* centroids;
    DISTFUNC dist_;
    std::vector<unsigned> represent_ids;

    IndexIVF(unsigned d, unsigned cluster_num): d(d), cluster_num(cluster_num), inverted_list_(cluster_num), centroids_(cluster_num), represent_ids(cluster_num) {
        // dist_ = utils::L2SqrFloatAVX512;
        centroids = new float[cluster_num * d];
    }

    ~IndexIVF() {
        delete[] centroids;
    }

    void add(unsigned n, float* data) {
        n_ = n;
        kmeans(n, data, 2);
    }

    void kmeans(unsigned n, float* data, unsigned kmeans_iter = 10) {
        unsigned bucket_size = n / cluster_num;
        std::cout << "Bucket size: " << bucket_size << std::endl;
        std::vector<std::vector<float> > t_l2_centroid(cluster_num);
        std::vector<std::vector<size_t>> t_ivf(cluster_num);


#pragma omp parallel for
        for (std::size_t i = 0; i < cluster_num; ++i) {
            float *data_ptr = &data[i * bucket_size * d];
            t_l2_centroid[i].assign(data_ptr, data_ptr+d);
        }

        std::vector<bool> centroid_empty(cluster_num, false);
        float err = std::numeric_limits<float>::max();
        while (kmeans_iter) {
            std::cout << "Iter: " << kmeans_iter << std::endl;
            std::vector<unsigned> cluster_id(n);

#pragma omp parallel for
            for (std::size_t i = 0; i < n; ++i) {
                cluster_id[i] = 0;
                float *data_ptr = &data[i*d];
                float min_l2 = dist_(data_ptr, t_l2_centroid[0].data(), &d) + 0 * t_ivf[0].size();
                for (std::size_t j = 1; j < cluster_num; ++j) {
                    if(centroid_empty[j]) continue;
                    float l2 = dist_(data_ptr, t_l2_centroid[j].data(), &d) + 0 * t_ivf[j].size();
                    if (l2 < min_l2) {
                        cluster_id[i] = j;
                        min_l2 = l2;
                    }
                }
            }
            std::cout << "Iter: " << kmeans_iter << ", cluster assign" << std::endl;

#pragma omp parallel for
            for (std::size_t i = 0; i < cluster_num; ++i) {
                t_ivf[i].clear();
            }

            for (std::size_t i = 0; i < n; ++i) {
                t_ivf[cluster_id[i]].push_back(i);
            }

            std::cout << t_ivf[0].size() << "\n";

#pragma omp parallel for
            for (std::size_t i = 0; i < cluster_num; ++i) {
                if (t_ivf[i].size()) {
                    for (std::size_t j = 0; j < d; ++j) {
                        t_l2_centroid[i][j] = data[t_ivf[i][0] * d + j];
                    }
                    for (std::size_t j = 1; j < t_ivf[i].size(); ++j) {
                        for (std::size_t k = 0; k < d; ++k) {
                            t_l2_centroid[i][k] += data[t_ivf[i][j] * d + k];
                        }
                    }
                    for (std::size_t j = 0; j < d; ++j) {
                        t_l2_centroid[i][j] /= t_ivf[i].size();
                    }
                } else {
                    // std::cout << "!!!!!!   Empty cluster: " << i  << "  !!!!!!!!!!!!!!!!!!"<< std::endl;
                    centroid_empty[i] = true;
                }
            }
            std::cout << "Recompute centroid" << std::endl;

            std::vector<float> err_clusters(cluster_num, 0);
#pragma omp parallel for
            for (std::size_t i = 0; i < cluster_num; ++i) {
                if (t_ivf[i].size()) {
                    float cluster_err = 0;
                    for (std::size_t j = 0; j < t_ivf[i].size(); ++j) {
                        cluster_err += dist_(&data[t_ivf[i][j]*d], t_l2_centroid[i].data(), &d);
                    }
                    err_clusters[i] = cluster_err / t_ivf[i].size();
                }
            }

            float avg_err = 0;
            for (std::size_t i = 0; i < cluster_num; ++i) {
                avg_err += err_clusters[i];
            }
            avg_err /= cluster_num;
            std::cout << "iter: " << kmeans_iter-- << ", avg err: " << avg_err << std::endl;
            // if (avg_err < err) {
            //     err = avg_err;
            // } else {
            //     break;
            // }
        }

        for (std::size_t i=0; i<cluster_num; ++i) {
            if(centroid_empty[i] && t_ivf[i].size()!=0) {
                throw std::runtime_error("cluster: "+std::to_string(i)+" flag is empty, inverted_list is not empty");
            }
        }

        unsigned cluster_num_ = 0;
        unsigned id_ = 0;
        unsigned min = n, max = 0;
        for (std::size_t i=0; i<cluster_num; ++i) {
            if (t_ivf[i].size() < min) min = t_ivf[i].size();
            if (t_ivf[i].size() > max) max = t_ivf[i].size();
            if (t_ivf[i].size() == 0) continue;
            for (const auto &id: t_ivf[i]) {
                inverted_list_[cluster_num_].push_back(id);
                ++id_;
            }
            centroids_[cluster_num_].assign(t_l2_centroid[i].begin(), t_l2_centroid[i].end());
            ++cluster_num_;
        }
        assert(id_ == n);
        cluster_num = cluster_num_;
        inverted_list_.resize(cluster_num);
        centroids_.resize(cluster_num);

        for (size_t i = 0; i < cluster_num; i++) {
            unsigned closest_id;
            float min_dist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < inverted_list_[i].size(); j++) {
                float dist = dist_(centroids_[i].data(), data + inverted_list_[i][j] * d, &d);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_id = inverted_list_[i][j];
                }
            }
            represent_ids[i] = closest_id;
        }

        std::cout << "cluster_num = " << cluster_num << "\n";
        std::cout << "min = " << min << "\n";
        std::cout << "max = " << max << "\n";
    }

    void save(const char* filename) {
        std::ofstream out(filename, std::ios::binary);
        out.write((char*)&d, sizeof(size_t));
        out.write((char*)&cluster_num, sizeof(unsigned));
        for (size_t i = 0; i < cluster_num; i++) {
            out.write((char*)centroids_[i].data(), d * sizeof(float));
        }
        out.write((char*)represent_ids.data(), cluster_num * sizeof(unsigned));
        out.close();
    }
};

IndexIVF* load_ivf(const char* filename) {
    std::ifstream in(filename, std::ios::binary);
    size_t d;
    unsigned cluster_num;
    in.read((char*)&d, sizeof(size_t));
    in.read((char*)&cluster_num, sizeof(unsigned));
    IndexIVF* index = new IndexIVF(d, cluster_num);
    for (size_t i = 0; i < cluster_num; i++) {
        index->centroids_[i].resize(d);
        in.read((char*)index->centroids_[i].data(), d * sizeof(float));
    }
    index->represent_ids.resize(cluster_num);
    in.read((char*)index->represent_ids.data(), cluster_num * sizeof(unsigned));
    in.close();
    for (size_t i = 0; i < cluster_num; i++) {
        memcpy(index->centroids + i * d, index->centroids_[i].data(), d * sizeof(float));
    }
    return index;
}

struct IndexIVF2Level {
    unsigned n;
    size_t d;
    unsigned ctl_factor = 1000;
    // level1
    unsigned l1_cluster_num = 100;
    std::vector<std::vector<float>> l1_centroids; // 100 * d
    // level2
    std::vector<unsigned> l2_cluster_nums;
    std::vector<std::vector<std::vector<float>>> l2_centroids; // 100 * l2_cluster_nums * d
    std::vector<std::vector<unsigned>> represent_ids;
    DISTFUNC dist_ = utils::L2SqrSIMD;

    IndexIVF2Level() {}

    void save(const char* fn) {
        std::ofstream out(fn, std::ios::binary);
        out.write((char*)&n, 4);
        out.write((char*)&d, 8);
        for (size_t i = 0; i < l1_cluster_num; i++) {
            out.write((char*)l1_centroids.data(), d * 4);
        }
        out.write((char*)l2_cluster_nums.data(), l1_cluster_num * 4);
        for (size_t i = 0; i < l1_cluster_num; i++) {
            for (size_t j = 0; j < l2_cluster_nums[i]; j++) {
                out.write((char*)l2_centroids[i][j].data(), d * 4);
            }
        }
        for (size_t i = 0; i < l1_cluster_num; i++) {
            out.write((char*)represent_ids[i].data(), l2_cluster_nums[i] * 4);
        }
        out.close();
    }

    void load(const char* fn) {
        std::ifstream in(fn, std::ios::binary);
        in.read((char*)&n, 4);
        in.read((char*)&d, 8);
        l1_centroids = std::vector<std::vector<float>>(l1_cluster_num, std::vector<float>(d));
        l2_cluster_nums.resize(l1_cluster_num);
        l2_centroids.resize(l1_cluster_num);
        represent_ids.resize(l1_cluster_num);
        for (size_t i = 0; i < l1_cluster_num; i++) {
            in.read((char*)l1_centroids.data(), d * 4);
        }
        in.read((char*)l2_cluster_nums.data(), l1_cluster_num * 4);
        for (size_t i = 0; i < l1_cluster_num; i++) {
            l2_centroids[i] = std::vector<std::vector<float>>(l2_cluster_nums[i], std::vector<float>(d));
            for (size_t j = 0; j < l2_cluster_nums[i]; j++) {
                in.read((char*)l2_centroids[i][j].data(), d * 4);
            }
        }
        for (size_t i = 0; i < l1_cluster_num; i++) {
            represent_ids[i].resize(l2_cluster_nums[i]);
            in.read((char*)represent_ids[i].data(), l2_cluster_nums[i] * 4);
        }
    }

    void search(float* query, unsigned nq, std::vector<int64_t>& eps, unsigned nprobe) {
        auto dfunc = (d == 200 ? utils::InnerProductFloatAVX512 : utils::InnerProductFloatAVX512Dim20);
#pragma omp parallel for
        for (size_t i = 0; i < nq; i++) {
            std::priority_queue<std::pair<float, unsigned>> queue;
            std::vector<std::pair<float, unsigned>> result;
            for (size_t j = 0; j < l1_cluster_num; j++) {
                queue.emplace(-dfunc(query + i * d, l1_centroids[j].data(), NULL), j);
            }
            for (size_t j = 0; j < 3; j++) {
                auto cid = queue.top().second;
                queue.pop();
                for (size_t k = 0; k < l2_cluster_nums[cid]; k++) {
                    result.emplace_back(dfunc(query + i * d, l2_centroids[cid][k].data(), NULL), represent_ids[cid][k]);
                }
            }
            std::sort(result.begin(), result.end());
            for (size_t j = 0; j < nprobe; j++) {
                eps[i * nprobe + j] = result[j].second;
            }
        }
    }

    void add(unsigned nd, size_t dim, float* data) {
        n = nd;
        d = dim;
        l1_centroids = std::vector<std::vector<float>>(l1_cluster_num, std::vector<float>(d));
        l2_cluster_nums.resize(l1_cluster_num);
        l2_centroids.resize(l1_cluster_num);
        represent_ids.resize(l1_cluster_num);
        std::vector<std::vector<unsigned>> l1_ivf(l1_cluster_num);
        kmeans(n, data, l1_cluster_num, l1_ivf, l1_centroids, 20);
        for (size_t i = 0; i < l1_cluster_num; i++) {
            unsigned size = l1_ivf[i].size();
            l2_cluster_nums[i] = div_round_up(size, ctl_factor);
            l2_centroids[i] = std::vector<std::vector<float>>(l2_cluster_nums[i], std::vector<float>(d));
            represent_ids[i].resize(l2_cluster_nums[i]);
            std::vector<float> tmp(size * d);
            for (size_t j = 0; j < size; j++) {
                for (size_t k = 0; k < d; k++) {
                    tmp[j * d + k] = data[l1_ivf[i][j] * d + k];
                }
            }
            std::vector<std::vector<unsigned>> l2_ivf(l2_cluster_nums[i]);
            kmeans(size, tmp.data(), l2_cluster_nums[i], l2_ivf, l2_centroids[i], 20);
            for (size_t j = 0; j < l2_cluster_nums[i]; j++) {
                unsigned closest_id;
                float min_dist = std::numeric_limits<float>::max();
                for (size_t k = 0; k < l2_ivf[j].size(); k++) {
                    float dist = dist_(l2_centroids[i][j].data(), data + l2_ivf[j][k] * d, &d);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_id = l2_ivf[j][k];
                    }
                }
                represent_ids[i][j] = closest_id;
            }
        }
    }

    void kmeans(unsigned n, float* data, unsigned cluster_num, std::vector<std::vector<unsigned>>& inverted_list, std::vector<std::vector<float>>& centroids, unsigned kmeans_iter = 10) {
        unsigned bucket_size = n / cluster_num;
        std::cout << "Bucket size: " << bucket_size << std::endl;
        std::vector<std::vector<float> > t_l2_centroid(cluster_num);
        std::vector<std::vector<size_t>> t_ivf(cluster_num);

#pragma omp parallel for
        for (std::size_t i = 0; i < cluster_num; ++i) {
            float *data_ptr = &data[i * bucket_size * d];
            t_l2_centroid[i].assign(data_ptr, data_ptr+d);
        }

        std::vector<bool> centroid_empty(cluster_num, false);
        float err = std::numeric_limits<float>::max();
        while (kmeans_iter) {
            std::cout << "Iter: " << kmeans_iter << std::endl;
            std::vector<unsigned> cluster_id(n);

#pragma omp parallel for
            for (std::size_t i = 0; i < n; ++i) {
                cluster_id[i] = 0;
                float *data_ptr = &data[i*d];
                float min_l2 = dist_(data_ptr, t_l2_centroid[0].data(), &d) + 0 * t_ivf[0].size();
                for (std::size_t j = 1; j < cluster_num; ++j) {
                    if(centroid_empty[j]) continue;
                    float l2 = dist_(data_ptr, t_l2_centroid[j].data(), &d) + 0 * t_ivf[j].size();
                    if (l2 < min_l2) {
                        cluster_id[i] = j;
                        min_l2 = l2;
                    }
                }
            }
            std::cout << "Iter: " << kmeans_iter << ", cluster assign" << std::endl;

#pragma omp parallel for
            for (std::size_t i = 0; i < cluster_num; ++i) {
                t_ivf[i].clear();
            }

            for (std::size_t i = 0; i < n; ++i) {
                t_ivf[cluster_id[i]].push_back(i);
            }

            std::cout << t_ivf[0].size() << "\n";

#pragma omp parallel for
            for (std::size_t i = 0; i < cluster_num; ++i) {
                if (t_ivf[i].size()) {
                    for (std::size_t j = 0; j < d; ++j) {
                        t_l2_centroid[i][j] = data[t_ivf[i][0] * d + j];
                    }
                    for (std::size_t j = 1; j < t_ivf[i].size(); ++j) {
                        for (std::size_t k = 0; k < d; ++k) {
                            t_l2_centroid[i][k] += data[t_ivf[i][j] * d + k];
                        }
                    }
                    for (std::size_t j = 0; j < d; ++j) {
                        t_l2_centroid[i][j] /= t_ivf[i].size();
                    }
                } else {
                    // std::cout << "!!!!!!   Empty cluster: " << i  << "  !!!!!!!!!!!!!!!!!!"<< std::endl;
                    centroid_empty[i] = true;
                }
            }
            std::cout << "Recompute centroid" << std::endl;

            std::vector<float> err_clusters(cluster_num, 0);
#pragma omp parallel for
            for (std::size_t i = 0; i < cluster_num; ++i) {
                if (t_ivf[i].size()) {
                    float cluster_err = 0;
                    for (std::size_t j = 0; j < t_ivf[i].size(); ++j) {
                        cluster_err += dist_(&data[t_ivf[i][j]*d], t_l2_centroid[i].data(), &d);
                    }
                    err_clusters[i] = cluster_err / t_ivf[i].size();
                }
            }

            float avg_err = 0;
            for (std::size_t i = 0; i < cluster_num; ++i) {
                avg_err += err_clusters[i];
            }
            avg_err /= cluster_num;
            std::cout << "iter: " << kmeans_iter-- << ", avg err: " << avg_err << std::endl;
        }

        for (std::size_t i=0; i<cluster_num; ++i) {
            if(centroid_empty[i] && t_ivf[i].size()!=0) {
                throw std::runtime_error("cluster: "+std::to_string(i)+" flag is empty, inverted_list is not empty");
            }
        }

        for (std::size_t i=0; i<cluster_num; ++i) {
            for (const auto &id: t_ivf[i]) {
                inverted_list[i].push_back(id);
            }
            centroids[i].assign(t_l2_centroid[i].begin(), t_l2_centroid[i].end());
        }
    }
};