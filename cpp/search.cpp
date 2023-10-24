#include <boost/dynamic_bitset.hpp>
#include <cmath>
#include "../utils/utils.h"
#include "../utils/dist_func.h"
#include "../index/IndexIVF.h"
#include "../index/IndexGraph.h"

#include <faiss/IndexPQ.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>

#include <mkl.h>

void knn_inner_product1(const float *x, const float *y, size_t d, size_t nx, size_t ny, faiss::float_minheap_array_t *result) {
    using RH = faiss::HeapResultHandler<faiss::CMin<float, int64_t>>;
    RH res(nx, result->val, result->ids, result->k);
    int bs_x = 4096;
    int bs_y = 1024;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
// #pragma omp parallel for
    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                int nyi = j1 - j0, nxi = i1 - i0, di = d;
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans,
                            CblasTrans,
                            nxi,
                            nyi,
                            di,
                            one,
                            x + i0 * d,
                            di,
                            y + j0 * d,
                            di,
                            zero,
                            ip_block.get(),
                            nyi);
            }

            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
    }
}

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
    // faiss::knn_inner_product(query, centroids, d, nq, cluster_num, &result, nullptr);
    knn_inner_product1(query, centroids, d, nq, cluster_num, &result);
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < nprobe; j++) {
            labels[i * nprobe + j] = represent_ids[labels[i * nprobe + j]];
        }
    }
    time.tuck("");
#pragma omp parallel for
    for (unsigned i = 0; i < nq; i++) {
        nsg.searchWithOptGraph(query + i * d, topK, L, res[i].data(), labels.data() + i * nprobe, nprobe);
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

    hnswlib::InnerProductSpace space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, 20, 1200);
    alg_hnsw->addPoint(data, 0);
#pragma omp parallel for
    for (size_t i = 1; i < n; i++) {
        alg_hnsw->addPoint(data + i * d, i);
    }

    IndexGraph nsg(d, n);
    nsg.load_graph("hnsw.graph");
    nsg.optimizeGraph(data);
    unsigned topK = 10, L = 121;
    testOptSearch(nq, d, k, nsg, query, gt, topK, L);
}