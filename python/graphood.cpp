#include "../index/IndexGraph.h"
#include "../index/IndexIVF.h"
#include "../hnswlib/hnswlib.h"

// #include <faiss/IndexPQ.h>
// #include <faiss/impl/ResultHandler.h>
// #include <faiss/utils/Heap.h>
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
        ivf = new IndexIVF2Level();
        ivf->load(ivf_fn);
    }

    py::array_t<unsigned> batch_search(unsigned nq, py::array_t<float> query_, unsigned k, unsigned ef, unsigned nprobe) {
        omp_set_num_threads(8);
        py::buffer_info buf_info = query_.request();
        float* query = (float*)buf_info.ptr;
        auto py_I = py::array_t<unsigned>(nq * k);
        py::buffer_info buf = py_I.request();
        unsigned* I = (unsigned*)buf.ptr;
        std::vector<int64_t> labels(nq * nprobe);
        Timer timer;
        timer.tick();
        ivf->search(query, nq, labels, nprobe);
        timer.tuck("");
#pragma omp parallel for
        for (size_t i = 0; i < nq; i++) {
            graph->searchWithOptGraph(query + i * d, k, ef, I + i * k, labels.data() + i * nprobe, nprobe);
        }
        timer.tuck("");
        py_I.resize({nq, k});
        return py_I;
    }

private:
    unsigned d;
    IndexGraph *graph;
    IndexIVF2Level *ivf;
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

PYBIND11_MODULE(graphood, m) {
    m.def("build_index", &build_index, "Build the index");
    py::class_<IndexGraphOOD>(m, "IndexGraphOOD")
        .def(py::init<unsigned, unsigned, const char*, const char*>())
        .def("batch_search", &IndexGraphOOD::batch_search, "Perform batch search");
}