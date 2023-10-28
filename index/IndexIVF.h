#include <iostream>
#include <vector>
#include <limits>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <queue>
#include "../utils/utils.h"
#include "../utils/dist_func.h"
#include <omp.h>

using DISTFUNC = float (*)(const void *, const void *, const void *);

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

    IndexIVF2Level(unsigned ctl_factor): ctl_factor(ctl_factor) {} 

    void save(const char* fn) {
        std::ofstream out(fn, std::ios::binary);
        out.write((char*)&l1_cluster_num, 4);
        out.write((char*)&n, 4);
        out.write((char*)&d, 8);
        for (size_t i = 0; i < l1_cluster_num; i++) {
            out.write((char*)l1_centroids[i].data(), d * 4);
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
        in.read((char*)&l1_cluster_num, 4);
        in.read((char*)&n, 4);
        in.read((char*)&d, 8);
        l1_centroids = std::vector<std::vector<float>>(l1_cluster_num, std::vector<float>(d));
        l2_cluster_nums.resize(l1_cluster_num);
        l2_centroids.resize(l1_cluster_num);
        represent_ids.resize(l1_cluster_num);
        for (size_t i = 0; i < l1_cluster_num; i++) {
            in.read((char*)l1_centroids[i].data(), d * 4);
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
        in.close();
    }

    void search(float* query, unsigned nq, std::vector<int64_t>& eps, unsigned nprobe) {
        auto dfunc = (d == 200 ? utils::InnerProductFloatAVX512 : utils::InnerProductFloatAVX512Dim20);
#pragma omp parallel for num_threads(8)
        for (size_t i = 0; i < nq; i++) {
            std::priority_queue<std::pair<float, unsigned>> queue;
            std::vector<std::pair<float, unsigned>> result;
            for (size_t j = 0; j < l1_cluster_num; j++) {
                queue.emplace(-dfunc(query + i * d, l1_centroids[j].data(), NULL), j);
            }
            auto sum = 0, j = 0;
            while (sum < nprobe || j < 3) {
                auto cid = queue.top().second;
                queue.pop();
                for (size_t k = 0; k < l2_cluster_nums[cid]; k++) {
                    result.emplace_back(dfunc(query + i * d, l2_centroids[cid][k].data(), NULL), represent_ids[cid][k]);
                }
                sum += l2_cluster_nums[cid];
                j++;
            }
            std::sort(result.begin(), result.end());
            for (size_t j = 0; j < nprobe; j++) {
                eps[i * nprobe + j] = result[j].second;
            }
        }
    }

    void add(unsigned nd, size_t dim, float* data) {
        l1_cluster_num = sqrt(nd / ctl_factor) + 1;
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
            std::vector<unsigned> id_map(size);
            for (size_t j = 0; j < size; j++) {
                id_map[j] = l1_ivf[i][j];
                for (size_t k = 0; k < d; k++) {
                    tmp[j * d + k] = data[l1_ivf[i][j] * d + k];
                }
            }
            std::vector<std::vector<unsigned>> l2_ivf(l2_cluster_nums[i]);
            kmeans(size, tmp.data(), l2_cluster_nums[i], l2_ivf, l2_centroids[i], 20);
            for (size_t j = 0; j < l2_cluster_nums[i]; j++) {
                unsigned closest_id = -1;
                float min_dist = std::numeric_limits<float>::max();
                for (size_t k = 0; k < l2_ivf[j].size(); k++) {
                    unsigned id = id_map[l2_ivf[j][k]];
                    float dist = dist_(l2_centroids[i][j].data(), data + id * d, &d);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_id = id;
                    }
                }
                represent_ids[i][j] = closest_id;
            }
        }
    }

    void kmeans(unsigned n, float* data, unsigned cluster_num, std::vector<std::vector<unsigned>>& inverted_list, std::vector<std::vector<float>>& centroids, unsigned kmeans_iter = 10) {
        unsigned bucket_size = n / cluster_num;
        // std::cout << "Bucket size: " << bucket_size << std::endl;
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
            // std::cout << "Iter: " << kmeans_iter << std::endl;
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
            // std::cout << "Iter: " << kmeans_iter << ", cluster assign" << std::endl;

#pragma omp parallel for
            for (std::size_t i = 0; i < cluster_num; ++i) {
                t_ivf[i].clear();
            }

            for (std::size_t i = 0; i < n; ++i) {
                t_ivf[cluster_id[i]].push_back(i);
            }

            // std::cout << t_ivf[0].size() << "\n";

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
                    std::cout << "!!!!!!   Empty cluster: " << i  << "  !!!!!!!!!!!!!!!!!!"<< std::endl;
                    centroid_empty[i] = true;
                }
            }
            // std::cout << "Recompute centroid" << std::endl;

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
            kmeans_iter--;
            // std::cout << "iter: " << kmeans_iter-- << ", avg err: " << avg_err << std::endl;
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