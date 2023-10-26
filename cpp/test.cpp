#include "../index/IndexGraph.h"
#include "../index/IndexIVF.h"
#include "../hnswlib/hnswlib.h"

#include <omp.h>

class IndexGraphOOD {
public:
    IndexGraphOOD(unsigned d, unsigned n, const char* index_fn, const char* ivf_fn): d(d) {
        graph = new IndexGraph(d, n);
        graph->load(index_fn);
        ivf.load(ivf_fn);
    }

    void batch_search(unsigned nq, float* query, unsigned k, unsigned ef, unsigned nprobe) {
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
        timer.tuck("");
    }

private:
    unsigned d;
    IndexGraph *graph;
    IndexIVF2Level ivf;
};

void build_index(const char* dataset_fn, const char* hnsw_fn, const char* ivf_fn, const char* index_fn, unsigned M, unsigned ef, unsigned cluster_num) {
    int fd = open(dataset_fn, O_RDONLY);
    unsigned n, d;
    auto data = read_fbin<float>(dataset_fn, n, d);
    IndexIVF2Level index;
    index.add(n, d, data);
    index.save(ivf_fn);
    std::cout << "built ivf\n";
    delete[] data;
    hnswlib::InnerProductSpace space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef);
    alg_hnsw->addPoint(read_vector(fd, d, 0), 0);
#pragma omp parallel for
    for (size_t i = 1; i < n; i++) {
        alg_hnsw->addPoint(read_vector(fd, d, i), i);
    }
    alg_hnsw->save_graph(hnsw_fn);
    std::cout << "built hnsw\n";
    delete alg_hnsw;
    IndexGraph graph(d, n);
    graph.load_graph(hnsw_fn);
    graph.optimizeGraph(fd);
    graph.save(index_fn);
    close(fd);
}

int main() {
    // build_index("/home/yanqi/NIPS2023/ood/data/base.1B.fbin.crop_nb_1000000", "hnsw", "ivf", "graph", 20, 1200, 8192);
    IndexGraphOOD index(200, 1000000, "graph", "ivf");
    unsigned nq, d;
    float* query = read_fbin<float>("/home/yuxiang/NeurIPS23/big-ann-benchmarks-main/data/text2image1B/query.public.100K.fbin", nq, d);
    index.batch_search(100000, query, 10, 120, 30);
}