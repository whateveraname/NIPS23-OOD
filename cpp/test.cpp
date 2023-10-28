#include "../index/IndexGraph.h"
#include "../index/IndexIVF.h"
#include "../hnswlib/hnswlib.h"
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/NSG.h>

#include <omp.h>

class IndexGraphOOD {
public:
    IndexGraphOOD(unsigned d, unsigned n, const char* graph_fn, const char* ivf_fn, const char* sq_fn): d(d) {
        graph = new IndexGraphSQ(n);
        graph->load_graph(graph_fn);
        sq = faiss::read_index(sq_fn);
        ivf.load(ivf_fn);
        graph->set_storage(d, ((faiss::IndexScalarQuantizer*)sq)->codes.data(), ((faiss::IndexScalarQuantizer*)sq)->sq.code_size);
    }

    unsigned* batch_search(unsigned nq, float* query, unsigned k, unsigned ef, unsigned nprobe) {
        omp_set_num_threads(8);
        unsigned* I = new unsigned[nq * k];
        std::vector<int64_t> labels(nq * nprobe);
        Timer timer;
        timer.tick();
        ivf.search(query, nq, labels, nprobe);
        timer.tuck("");
#pragma omp parallel for
        for (size_t i = 0; i < nq; i++) {
            graph->searchWithOptGraph(query + i * d, k, ef, I + i * k, labels.data() + i * nprobe, nprobe);
        }
// #pragma omp parallel
//         {
//             faiss::FlatCodesDistanceComputer* dis = ((faiss::IndexScalarQuantizer*)sq)->get_FlatCodesDistanceComputer();
//             faiss::ScopeDeleter1<faiss::DistanceComputer> del(dis);
// #pragma omp for
//             for (size_t i = 0; i < nq; i++) {
//                 dis->set_query(query + i * d);
//                 graph->searchWithOptGraph(*dis, k, ef, I + i * k, labels.data() + i * nprobe, nprobe);
//             }
//         }
        timer.tuck("");
        return I;
    }

private:
    unsigned d;
    IndexGraphSQ* graph;
    faiss::Index* sq;
    IndexIVF2Level ivf;
};

void build_index(const char* dataset_fn, const char* hnsw_fn, const char* ivf_fn, const char* index_fn, const char* sq_fn, unsigned M, unsigned ef, unsigned cluster_num) {
    int fd = open(dataset_fn, O_RDONLY);
    unsigned n, d, nq;
    auto data = read_fbin<float>(dataset_fn, n, d);
    IndexIVF2Level index(cluster_num);
    index.add(n, d, data);
    index.save(ivf_fn);
    auto sq = faiss::index_factory(d, "SQfp16", faiss::METRIC_INNER_PRODUCT);
    sq->train(n, data);
    sq->add(n, data);
    faiss::write_index(sq, sq_fn);
    delete sq;
    delete[] data;
    close(fd);
}

int main() {
    // build_index("/home/yanqi/NIPS2023/ood/data/base.1B.fbin.crop_nb_10000000", "hnsw", "ivf", "graph", "sq", 20, 1200, 1000);
    IndexGraphOOD index(200, 10000000, "hnsw", "ivf", "sq");
    unsigned nq, d, k;
    float* query = read_fbin<float>("/home/yuxiang/NeurIPS23/big-ann-benchmarks-main/data/text2image1B/query.public.100K.fbin", nq, d);
    unsigned* gt = read_fbin<unsigned>("/home/yuxiang/NeurIPS23/big-ann-benchmarks-main/data/text2image1B/text2image-10M", nq, k);
    auto I = index.batch_search(100000, query, 10, 136, 30);
    float optrecall = 0;
    for (size_t i = 0; i < nq; i++) {
        float num = 0;
        for (size_t j = 0; j < 10; j++) {
            for (size_t m = 0; m < 10; m++) {
                if (gt[i * k + j] == I[i*10+m]) {
                    num++;
                    break;
                }
            }
        }
        optrecall += num / 10;
    }
    std::cout << "optrecall: " << optrecall / nq << std::endl;
}