#include "../index/IndexGraph.h"
#include "../index/IndexIVF.h"
#include "../hnswlib/hnswlib.h"

#include <faiss/IndexPQ.h>
#include <faiss/impl/ResultHandler.h>
#include <omp.h>

#undef max
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

class IndexGraphOOD {
public:
    IndexGraphOOD(unsigned d, unsigned n, const char* index_fn, const char* ivf_fn): d(d) {
        graph = new IndexGraph(d, n);
        graph->load(index_fn);
        ivf = load_ivf(ivf_fn);
        centroids = ivf->centroids;
        cluster_num = ivf->cluster_num;
        represent_ids = ivf->represent_ids;
    }

    py::array_t<unsigned> batch_search(unsigned nq, py::array_t<float> query_, unsigned k, unsigned ef, unsigned nprobe) {
        omp_set_num_threads(8);
        py::buffer_info buf_info = query_.request();
        float* query = (float*)buf_info.ptr;
        auto py_I = py::array_t<unsigned>(nq * k);
        py::buffer_info buf = py_I.request();
        unsigned* I = (unsigned*)buf.ptr;
        std::vector<int64_t> labels(nq * nprobe);
        std::vector<float> distances(nq * nprobe);
        faiss::float_minheap_array_t res = {size_t(nq), size_t(nprobe), labels.data(), distances.data()};
        faiss::knn_inner_product(query, centroids, d, nq, cluster_num, &res, nullptr);
        for (size_t i = 0; i < nq; i++) {
            for (size_t j = 0; j < nprobe; j++) {
                labels[i * nprobe + j] = represent_ids[labels[i * nprobe + j]];
            }
        }
#pragma omp parallel for
        for (size_t i = 0; i < nq; i++) {
            graph->searchWithOptGraphRestart(query + i * d, k, ef, I + i * k, labels.data() + i * nprobe, nprobe);
        }
        py_I.resize({nq, k});
        return py_I;
    }

private:
    unsigned d;
    IndexGraph *graph;
    IndexIVF *ivf;
    float* centroids;
    unsigned cluster_num;
    std::vector<unsigned> represent_ids;
};

void build_index(const char* dataset_fn, const char* hnsw_fn, const char* ivf_fn, const char* index_fn, unsigned M, unsigned ef, unsigned cluster_num) {
    unsigned n, d;
    float* data = read_fbin<float>(dataset_fn, n, d);
    hnswlib::InnerProductSpace space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef);
    alg_hnsw->addPoint(data, 0);
#pragma omp parallel for
    for (size_t i = 1; i < n; i++) {
        alg_hnsw->addPoint(data + i * d, i);
    }
    alg_hnsw->save_graph(hnsw_fn);
    IndexIVF index(d, cluster_num);
    index.add(n, data);
    index.save(ivf_fn);
    IndexGraph graph(d, n);
    graph.load_graph(hnsw_fn);
    graph.optimizeGraph(data);
    std::cout << "optimize done\n";
    graph.save(index_fn);
    std::cout << "save done\n";
}

PYBIND11_MODULE(graphood, m) {
    m.def("build_index", &build_index, "Build the index");
    py::class_<IndexGraphOOD>(m, "IndexGraphOOD")
        .def(py::init<unsigned, unsigned, const char*, const char*>())
        .def("batch_search", &IndexGraphOOD::batch_search, "Perform batch search");
}