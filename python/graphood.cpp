#include "../index/IndexGraph.h"
#include "../index/IndexIVF.h"
#include "../hnswlib/hnswlib.h"

#include <faiss/IndexPQ.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>
#include <omp.h>

#include <mkl.h>

#undef max
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

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

class IndexGraphOOD {
public:
    IndexGraphOOD(unsigned d, unsigned n, const char* index_fn, const char* ivf_fn): d(d) {
        std::ofstream log("/home/app/data/indices/ood/final/Text2Image1B-10000000/log1");
        log << "into constructor\n";
        log.flush();
        graph = new IndexGraph(d, n);
        graph->load(index_fn);
        log << "load graph\n";
        log.flush();
        ivf = load_ivf(ivf_fn);
        centroids = ivf->centroids;
        cluster_num = ivf->cluster_num;
        represent_ids = ivf->represent_ids;
        log << "load ivf\n";
        log.flush();
        log.close();
    }

    py::array_t<unsigned> batch_search(unsigned nq, py::array_t<float> query_, unsigned k, unsigned ef, unsigned nprobe) {
        // omp_set_num_threads(8);
        py::buffer_info buf_info = query_.request();
        float* query = (float*)buf_info.ptr;
        auto py_I = py::array_t<unsigned>(nq * k);
        py::buffer_info buf = py_I.request();
        unsigned* I = (unsigned*)buf.ptr;
        std::vector<int64_t> labels(nq * nprobe);
        std::vector<float> distances(nq * nprobe);
        faiss::float_minheap_array_t res = {size_t(nq), size_t(nprobe), labels.data(), distances.data()};
        // faiss::knn_inner_product(query, centroids, d, nq, cluster_num, &res, nullptr);
        Timer timer;
        timer.tick();
        knn_inner_product1(query, centroids, d, nq, cluster_num, &res);
        for (size_t i = 0; i < nq; i++) {
            for (size_t j = 0; j < nprobe; j++) {
                labels[i * nprobe + j] = represent_ids[labels[i * nprobe + j]];
            }
        }
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
    IndexIVF *ivf;
    float* centroids;
    unsigned cluster_num;
    std::vector<unsigned> represent_ids;
};

void build_index(const char* dataset_fn, const char* hnsw_fn, const char* ivf_fn, const char* index_fn, unsigned M, unsigned ef, unsigned cluster_num) {
    std::ofstream log("/home/app/data/indices/ood/final/Text2Image1B-10000000/log");
    log << "into build\n";
    log.flush();
    unsigned n, d;
    std::ifstream in(dataset_fn, std::ios::binary);
    in.read((char*)&n, 4);
    in.read((char*)&d, 4);
    in.close();
    int fd = open(dataset_fn, O_RDONLY);
    int len = lseek(fd,0,SEEK_END);
    auto data = read_fbin<float>(dataset_fn, n, d);
    IndexIVF index(d, cluster_num);
    index.add(n, data);
    index.save(ivf_fn);
    log << "ivf build done\n";
    log.flush();
    // munmap(data, len - 8);
    delete[] data;
    hnswlib::InnerProductSpace space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef);
    alg_hnsw->addPoint(read_vector(fd, d, 0), 0);
#pragma omp parallel for
    for (size_t i = 1; i < n; i++) {
        alg_hnsw->addPoint(read_vector(fd, d, i), i);
    }
    alg_hnsw->save_graph(hnsw_fn);
    log << "hnsw build done\n";
    log.flush();
    delete alg_hnsw;
    IndexGraph graph(d, n);
    graph.load_graph(hnsw_fn);
    graph.optimizeGraph(fd);
    std::cout << "optimize done\n";
    graph.save(index_fn);
    log << "graph build done\n";
    std::cout << "save done\n";
    log.close();
    close(fd);
}

PYBIND11_MODULE(graphood, m) {
    m.def("build_index", &build_index, "Build the index");
    py::class_<IndexGraphOOD>(m, "IndexGraphOOD")
        .def(py::init<unsigned, unsigned, const char*, const char*>())
        .def("batch_search", &IndexGraphOOD::batch_search, "Perform batch search");
}