#include <boost/dynamic_bitset.hpp>
#include <cmath>
#include "../utils/utils.h"
#include "../utils/dist_func.h"
#include "../index/IndexIVF.h"
#include "../index/IndexGraph.h"

void testOptSearch(unsigned nq, unsigned d, unsigned k, IndexGraph& nsg, float* query, unsigned* gt, unsigned topK, unsigned L) {
    std::vector<std::vector<unsigned>> res(nq);
    for (unsigned i = 0; i < nq; i++) res[i].resize(topK);

    auto index = load_ivf("8192.10M.ivf");
    auto centroids = index->centroids;
    auto cluster_num = index->cluster_num;
    auto represent_ids = index->represent_ids;
    delete index;

    omp_set_num_threads(8);
    Timer time;
    time.tick();
    size_t nprobe = 30;
    std::vector<int64_t> labels(nq * nprobe);
    std::vector<float> distances(nq * nprobe);
    faiss::float_minheap_array_t result = {size_t(nq), size_t(nprobe), labels.data(), distances.data()};
    faiss::knn_inner_product(query, centroids, d, nq, cluster_num, &result, nullptr);
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < nprobe; j++) {
            labels[i * nprobe + j] = represent_ids[labels[i * nprobe + j]];
        }
    }
    time.tuck("");
#pragma omp parallel for
    for (unsigned i = 0; i < nq; i++) {
        nsg.searchWithOptGraphRestart(query + i * d, topK, L, res[i].data(), labels.data() + i * nprobe, nprobe);
    }
    time.tuck("optsearch done");
    float optrecall = 0;
    for (size_t i = 0; i < nq; i++) {
        float num = 0;
        for (size_t j = 0; j < topK; j++) {
            for (size_t m = 0; m < topK; m++) {
                if (gt[i * k + j] == res[i][m]) {
                    num++;
                    break;
                }
            }
        }
        optrecall += num / topK;
    }
    std::cout << "optrecall: " << optrecall / nq << std::endl;
    std::cout << "optqps: " << nq / time.diff.count() << std::endl;
}

int main() {
    unsigned n, d, nq, k;
    float* data = read_fbin<float>("/home/yuxiang/NeurIPS23/big-ann-benchmarks-main/data/text2image1B/base.1B.fbin.crop_nb_10000000", n, d);
    float* query = read_fbin<float>("/home/yuxiang/NeurIPS23/big-ann-benchmarks-main/data/text2image1B/query.public.100K.fbin", nq, d);
    unsigned* gt = read_fbin<unsigned>("/home/yuxiang/NeurIPS23/big-ann-benchmarks-main/data/text2image1B/text2image-10M", nq, k);
    IndexGraph nsg(d, n);
    nsg.load_graph("hnsw.graph");
    nsg.optimizeGraph(data);
    unsigned topK = 10, L = 121;
    testOptSearch(nq, d, k, nsg, query, gt, topK, L);
}