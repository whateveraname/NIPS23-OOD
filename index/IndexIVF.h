#include <iostream>
#include <vector>
#include <limits>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <queue>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <boost/dynamic_bitset.hpp>
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
        dist_ = utils::L2Sqr;
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

        std::cout << "Bucket size: " << bucket_size << std::endl;

// #pragma omp parallel for
        for (std::size_t i = 0; i < cluster_num; ++i) {
            std::cout << "1 " << bucket_size << std::endl;
            float *data_ptr = &data[i * bucket_size * d];
            std::cout << "2 " << bucket_size << std::endl;
            t_l2_centroid[i].assign(data_ptr, data_ptr+d);
        }

        std::cout << "Bucket size: " << bucket_size << std::endl;

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