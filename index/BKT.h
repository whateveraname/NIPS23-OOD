#include <stack>
#include <string>
#include <vector>
#include <shared_mutex>

#include "../utils/utils.h"
#include "../utils/dist_func.h"

#define ALIGN_ALLOC(size) _mm_malloc(size, 32)
#define ALIGN_FREE(ptr) _mm_free(ptr)

const float MaxDist = (std::numeric_limits<float>::max)() / 10;

static unsigned rand(unsigned high = RAND_MAX, unsigned low = 0)   // Generates a random int value.
{
    return low + (unsigned)(float(high - low)*(std::rand() / (RAND_MAX + 1.0)));
}

template <class T>
inline T min(T a, T b) {
    return a < b ? a : b;
}

// node type for storing BKT
struct BKTNode
{
    unsigned centerid;
    unsigned childStart;
    unsigned childEnd;

    BKTNode(unsigned cid = -1) : centerid(cid), childStart(-1), childEnd(-1) {}
};

using DISTFUNC = float (*)(const void *, const void *, const void *);

struct KmeansArgs {
    int _K;
    int _DK;
    size_t _D;
    size_t _RD;
    int _T;
    float* centers;
    float* newTCenters;
    unsigned* counts;
    float* newCenters;
    unsigned* newCounts;
    int* label;
    unsigned* clusterIdx;
    float* clusterDist;
    float* weightedCounts;
    float* newWeightedCounts;
    DISTFUNC fComputeDistance;

    KmeansArgs(int k, size_t dim, unsigned datasize, int threadnum) : _K(k), _DK(k), _D(dim), _RD(dim), _T(threadnum) {
        centers = (float*)ALIGN_ALLOC(sizeof(float) * _K * _D);
        newTCenters = (float*)ALIGN_ALLOC(sizeof(float) * _K * _D);
        counts = new unsigned[_K];
        newCenters = new float[_T * _K * _RD];
        newCounts = new unsigned[_T * _K];
        label = new int[datasize];
        clusterIdx = new unsigned[_T * _K];
        clusterDist = new float[_T * _K];
        weightedCounts = new float[_K];
        newWeightedCounts = new float[_T * _K];
    }

    ~KmeansArgs() {
        ALIGN_FREE(centers);
        ALIGN_FREE(newTCenters);
        delete[] counts;
        delete[] newCenters;
        delete[] newCounts;
        delete[] label;
        delete[] clusterIdx;
        delete[] clusterDist;
        delete[] weightedCounts;
        delete[] newWeightedCounts;
    }

    inline void ClearCounts() {
        memset(newCounts, 0, sizeof(unsigned) * _T * _K);
        memset(newWeightedCounts, 0, sizeof(float) * _T * _K);
    }

    inline void ClearCenters() {
        memset(newCenters, 0, sizeof(float) * _T * _K * _RD);
    }

    inline void ClearDists(float dist) {
        for (int i = 0; i < _T * _K; i++) {
            clusterIdx[i] = -1;
            clusterDist[i] = dist;
        }
    }

    void Shuffle(std::vector<unsigned>& indices, unsigned first, unsigned last) {
        unsigned* pos = new unsigned[_K];
        pos[0] = first;
        for (int k = 1; k < _K; k++) pos[k] = pos[k - 1] + newCounts[k - 1];

        for (int k = 0; k < _K; k++) {
            if (counts[k] == 0) continue;
            unsigned i = pos[k];
            while (newCounts[k] > 0) {
                unsigned swapid = pos[label[i]] + newCounts[label[i]] - 1;
                newCounts[label[i]]--;
                std::swap(indices[i], indices[swapid]);
                std::swap(label[i], label[swapid]);
            }
            while (indices[i] != clusterIdx[k]) i++;
            std::swap(indices[i], indices[pos[k] + counts[k] - 1]);
        }
        delete[] pos;
    }
};

void RefineLambda(KmeansArgs& args, float& lambda, int size)
{
    int maxcluster = -1;
    unsigned maxCount = 0;
    for (int k = 0; k < args._DK; k++) {
        if (args.counts[k] > maxCount && args.newCounts[k] > 0)
        {
            maxcluster = k;
            maxCount = args.counts[k];
        }
    }

    float avgDist = args.newWeightedCounts[maxcluster] / args.newCounts[maxcluster];
    //lambda = avgDist / 10 / args.counts[maxcluster];
    //lambda = (args.clusterDist[maxcluster] - avgDist) / args.newCounts[maxcluster];
    lambda = (args.clusterDist[maxcluster] - avgDist) / size;
    if (lambda < 0) lambda = 0;
}

float RefineCenters(const float* data, KmeansArgs& args)
{
    int maxcluster = -1;
    unsigned maxCount = 0;
    for (int k = 0; k < args._DK; k++) {
        if (args.counts[k] > maxCount && args.newCounts[k] > 0 && utils::L2SqrSIMD(data + args.clusterIdx[k] * args._D, args.centers + k * args._D, &args._D) > 1e-6) // TODO
        {
            maxcluster = k;
            maxCount = args.counts[k];
        }
    }

    float diff = 0;
    std::vector<float> reconstructVector(args._RD, 0);
    for (int k = 0; k < args._DK; k++) {
        float* TCenter = args.newTCenters + k * args._D;
        if (args.counts[k] == 0) {
            if (maxcluster != -1) {
                //int nextid = Utils::rand_int(last, first);
                //while (args.label[nextid] != maxcluster) nextid = Utils::rand_int(last, first);
                unsigned nextid = args.clusterIdx[maxcluster];
                std::memcpy(TCenter, data + nextid * args._D, sizeof(float)*args._D);
            }
            else {
                std::memcpy(TCenter, args.centers + k * args._D, sizeof(float)*args._D);
            }
        }
        else {
            float* currCenters = args.newCenters + k * args._RD;
            for (size_t j = 0; j < args._RD; j++) {
                currCenters[j] /= args.counts[k];
            }
            for (size_t j = 0; j < args._D; j++) TCenter[j] = (float)(currCenters[j]);
        }
        diff += utils::L2SqrSIMD(TCenter, args.centers + k * args._D, &args._D);
    }
    return diff;
}

inline float KmeansAssign(const float* data,
    std::vector<unsigned>& indices,
    const unsigned first, const unsigned last, KmeansArgs& args, 
    const bool updateCenters, float lambda) {
    float currDist = 0;
    unsigned subsize = (last - first - 1) / args._T + 1;

#pragma omp parallel for num_threads(args._T) shared(data, indices) reduction(+:currDist)
    for (int tid = 0; tid < args._T; tid++)
    {
        unsigned istart = first + tid * subsize;
        unsigned iend = min(first + (tid + 1) * subsize, last);
        unsigned *inewCounts = args.newCounts + tid * args._K;
        float *inewCenters = args.newCenters + tid * args._K * args._RD;
        unsigned * iclusterIdx = args.clusterIdx + tid * args._K;
        float * iclusterDist = args.clusterDist + tid * args._K;
        float * iweightedCounts = args.newWeightedCounts + tid * args._K;
        float idist = 0;
        float* reconstructVector = nullptr;

        for (unsigned i = istart; i < iend; i++) {
            int clusterid = 0;
            float smallestDist = MaxDist;
            for (int k = 0; k < args._DK; k++) {
                float dist = args.fComputeDistance(data + indices[i] * args._D, args.centers + k*args._D, &args._D) + lambda*args.counts[k];
                if (dist > -MaxDist && dist < smallestDist) {
                    clusterid = k; smallestDist = dist;
                }
            }
            args.label[i] = clusterid;
            inewCounts[clusterid]++;
            iweightedCounts[clusterid] += smallestDist;
            idist += smallestDist;
            if (updateCenters) {
                reconstructVector = (float*)data + indices[i] * args._D;
                float* center = inewCenters + clusterid*args._RD;
                for (size_t j = 0; j < args._RD; j++) center[j] += reconstructVector[j];

                if (smallestDist > iclusterDist[clusterid]) {
                    iclusterDist[clusterid] = smallestDist;
                    iclusterIdx[clusterid] = indices[i];
                }
            }
            else {
                if (smallestDist <= iclusterDist[clusterid]) {
                    iclusterDist[clusterid] = smallestDist;
                    iclusterIdx[clusterid] = indices[i];
                }
            }
        }
        currDist += idist;
    }

    for (int i = 1; i < args._T; i++) {
        for (int k = 0; k < args._DK; k++) {
            args.newCounts[k] += args.newCounts[i * args._K + k];
            args.newWeightedCounts[k] += args.newWeightedCounts[i * args._K + k];
        }
    }

    if (updateCenters) {
        for (int i = 1; i < args._T; i++) {
            float* currCenter = args.newCenters + i*args._K*args._RD;
            for (size_t j = 0; j < ((size_t)args._DK) * args._RD; j++) args.newCenters[j] += currCenter[j];

            for (int k = 0; k < args._DK; k++) {
                if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] > args.clusterDist[k]) {
                    args.clusterDist[k] = args.clusterDist[i*args._K + k];
                    args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                }
            }
        }
    }
    else {
        for (int i = 1; i < args._T; i++) {
            for (int k = 0; k < args._DK; k++) {
                if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] <= args.clusterDist[k]) {
                    args.clusterDist[k] = args.clusterDist[i*args._K + k];
                    args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                }
            }
        }
    }
    return currDist;
}

inline float InitCenters(const float* data, 
    std::vector<unsigned>& indices, const unsigned first, const unsigned last, 
    KmeansArgs& args, int samples, int tryIters) {
    unsigned batchEnd = min(first + samples, last);
    float lambda = 0, currDist, minClusterDist = MaxDist;
    for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
        for (int k = 0; k < args._DK; k++) {
            unsigned randid = rand(last, first);
            std::memcpy(args.centers + k*args._D, data + indices[randid] * args._D, sizeof(float)*args._D);
        }
        args.ClearCounts();
        args.ClearDists(-MaxDist);
        currDist = KmeansAssign(data, indices, first, batchEnd, args, true, 0);
        if (currDist < minClusterDist) {
            minClusterDist = currDist;
            memcpy(args.newTCenters, args.centers, sizeof(float)*args._K*args._D);
            memcpy(args.counts, args.newCounts, sizeof(unsigned) * args._K);

            RefineLambda(args, lambda, batchEnd - first);
        }
    }
    return lambda;
}

float TryClustering(const float* data,
    std::vector<unsigned>& indices, const unsigned first, const unsigned last,
    KmeansArgs& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false) {

    float adjustedLambda = InitCenters(data, indices, first, last, args, samples, 3);

    unsigned batchEnd = min(first + samples, last);
    float currDiff, currDist, minClusterDist = MaxDist;
    int noImprovement = 0;
    float originalLambda = 1 / lambdaFactor / (batchEnd - first);
    for (int iter = 0; iter < 100; iter++) {
        std::memcpy(args.centers, args.newTCenters, sizeof(float)*args._K*args._D);
        std::shuffle(indices.begin() + first, indices.begin() + last, std::mt19937());

        args.ClearCenters();
        args.ClearCounts();
        args.ClearDists(-MaxDist);
        currDist = KmeansAssign(data, indices, first, batchEnd, args, true, min(adjustedLambda, originalLambda));
        std::memcpy(args.counts, args.newCounts, sizeof(unsigned) * args._K);

        if (currDist < minClusterDist) {
            noImprovement = 0;
            minClusterDist = currDist;
        }
        else {
            noImprovement++;
        }

        /*
        if (debug) {
            std::string log = "";
            for (int k = 0; k < args._DK; k++) {
                log += std::to_string(args.counts[k]) + " ";
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f lambda:(%f,%f) counts:%s\n", iter, currDist, originalLambda, adjustedLambda, log.c_str());
        }
        */

        currDiff = RefineCenters(data, args);
        //if (debug) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iter, currDist, currDiff);

        if (currDiff < 1e-3 || noImprovement >= 5) break;
    }

    args.ClearCounts();
    args.ClearDists(MaxDist);
    currDist = KmeansAssign(data, indices, first, last, args, false, 0);
    for (int k = 0; k < args._DK; k++) {
        if (args.clusterIdx[k] != -1) std::memcpy(args.centers + k * args._D, data + args.clusterIdx[k] * args._D, sizeof(float) * args._D);
    }

    args.ClearCounts();
    args.ClearDists(MaxDist);
    currDist = KmeansAssign(data, indices, first, last, args, false, 0);
    std::memcpy(args.counts, args.newCounts, sizeof(unsigned) * args._K);

    unsigned maxCount = 0, minCount = (std::numeric_limits<unsigned>::max)(), availableClusters = 0;
    float CountStd = 0.0, CountAvg = (last - first) * 1.0f / args._DK;
    for (int i = 0; i < args._DK; i++) {
        if (args.counts[i] > maxCount) maxCount = args.counts[i];
        if (args.counts[i] < minCount) minCount = args.counts[i];
        CountStd += (args.counts[i] - CountAvg) * (args.counts[i] - CountAvg);
        if (args.counts[i] > 0) availableClusters++;
    }
    CountStd = sqrt(CountStd / args._DK) / CountAvg;

    return CountStd;
}

float DynamicFactorSelect(const float* data,
    std::vector<unsigned> & indices, const unsigned first, const unsigned last,
    KmeansArgs & args, int samples = 1000) {

    float bestLambdaFactor = 100.0f, bestCountStd = (std::numeric_limits<float>::max)();
    for (float lambdaFactor = 0.001f; lambdaFactor <= 1000.0f + 1e-3; lambdaFactor *= 10) {
        float CountStd = 0.0;
        CountStd = TryClustering(data, indices, first, last, args, samples, lambdaFactor, true);

        if (CountStd < bestCountStd) {
            bestLambdaFactor = lambdaFactor;
            bestCountStd = CountStd;
        }
    }
    /*
    std::vector<float> tries(16, 0);
    for (int i = 0; i < 8; i++) {
        tries[i] = bestLambdaFactor * (i + 2) / 10;
        tries[8 + i] = bestLambdaFactor * (i + 2);
    }
    for (float lambdaFactor : tries) {
        float CountStd = TryClustering(data, indices, first, last, args, samples, lambdaFactor, true);
        if (CountStd < bestCountStd) {
            bestLambdaFactor = lambdaFactor;
            bestCountStd = CountStd;
        }
    }
    */
    return bestLambdaFactor;
}

int KmeansClustering(const float* data,
    std::vector<unsigned>& indices, const unsigned first, const unsigned last, 
    KmeansArgs& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false) {
    
    TryClustering(data, indices, first, last, args, samples, lambdaFactor, debug);

    int numClusters = 0;
    for (int i = 0; i < args._K; i++) if (args.counts[i] > 0) numClusters++;

    if (numClusters <= 1) return numClusters;

    args.Shuffle(indices, first, last);
    return numClusters;
}

class BKTree
{
public:
    BKTree(): m_iTreeNumber(1), m_iBKTKmeansK(32), m_iBKTLeafSize(8), m_iSamples(1000), m_fBalanceFactor(-1.0f), m_bfs(0), m_lock(new std::shared_timed_mutex) {}
    
    BKTree(const BKTree& other): m_iTreeNumber(other.m_iTreeNumber), 
                            m_iBKTKmeansK(other.m_iBKTKmeansK), 
                            m_iBKTLeafSize(other.m_iBKTLeafSize),
                            m_iSamples(other.m_iSamples),
                            m_fBalanceFactor(other.m_fBalanceFactor),
                            m_lock(new std::shared_timed_mutex) {}
    ~BKTree() {}

    inline const BKTNode& operator[](unsigned index) const { return m_pTreeRoots[index]; }
    inline BKTNode& operator[](unsigned index) { return m_pTreeRoots[index]; }

    inline unsigned size() const { return (unsigned)m_pTreeRoots.size(); }
    
    inline unsigned sizePerTree() const {
        std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
        return (unsigned)m_pTreeRoots.size() - m_pTreeStart.back(); 
    }

    inline const std::unordered_map<unsigned, unsigned>& GetSampleMap() const { return m_pSampleCenterMap; }

    inline void SwapTree(BKTree& newTrees)
    {
        m_pTreeRoots.swap(newTrees.m_pTreeRoots);
        m_pTreeStart.swap(newTrees.m_pTreeStart);
        m_pSampleCenterMap.swap(newTrees.m_pSampleCenterMap);
    }

    template <typename T>
    void BuildTrees(const float* data, int numOfThreads, 
        std::vector<unsigned>* indices = nullptr, std::vector<unsigned>* reverseIndices = nullptr, 
        bool dynamicK = false)
    {
        struct  BKTStackItem {
            unsigned index, first, last;
            bool debug;
            BKTStackItem(unsigned index_, unsigned first_, unsigned last_, bool debug_ = false) : index(index_), first(first_), last(last_), debug(debug_) {}
        };
        std::stack<BKTStackItem> ss;

        std::vector<unsigned> localindices;
        if (indices == nullptr) {
            localindices.resize(data.R());
            for (unsigned i = 0; i < localindices.size(); i++) localindices[i] = i;
        }
        else {
            localindices.assign(indices->begin(), indices->end());
        }
        KmeansArgs args(m_iBKTKmeansK, data.C(), (unsigned)localindices.size(), numOfThreads);

        if (m_fBalanceFactor < 0) m_fBalanceFactor = DynamicFactorSelect(data, localindices, 0, (unsigned)localindices.size(), args, m_iSamples);

        m_pSampleCenterMap.clear();
        for (char i = 0; i < m_iTreeNumber; i++)
        {
            std::shuffle(localindices.begin(), localindices.end(), std::mt19937());

            m_pTreeStart.push_back((unsigned)m_pTreeRoots.size());
            m_pTreeRoots.emplace_back((unsigned)localindices.size());

            ss.push(BKTStackItem(m_pTreeStart[i], 0, (unsigned)localindices.size(), true));
            while (!ss.empty()) {
                BKTStackItem item = ss.top(); ss.pop();
                m_pTreeRoots[item.index].childStart = (unsigned)m_pTreeRoots.size();
                if (item.last - item.first <= m_iBKTLeafSize) {
                    for (unsigned j = item.first; j < item.last; j++) {
                        unsigned cid = (reverseIndices == nullptr)? localindices[j]: reverseIndices->at(localindices[j]);
                        m_pTreeRoots.emplace_back(cid);
                    }
                }
                else { // clustering the data into BKTKmeansK clusters
                    if (dynamicK) {
                        args._DK = std::min<int>((item.last - item.first) / m_iBKTLeafSize + 1, m_iBKTKmeansK);
                        args._DK = std::max<int>(args._DK, 2);
                    }

                    int numClusters = KmeansClustering(data, localindices, item.first, item.last, args, m_iSamples, m_fBalanceFactor, item.debug);
                    if (numClusters <= 1) {
                        unsigned end = min(item.last + 1, (unsigned)localindices.size());
                        std::sort(localindices.begin() + item.first, localindices.begin() + end);
                        m_pTreeRoots[item.index].centerid = (reverseIndices == nullptr) ? localindices[item.first] : reverseIndices->at(localindices[item.first]);
                        m_pTreeRoots[item.index].childStart = -m_pTreeRoots[item.index].childStart;
                        for (unsigned j = item.first + 1; j < end; j++) {
                            unsigned cid = (reverseIndices == nullptr) ? localindices[j] : reverseIndices->at(localindices[j]);
                            m_pTreeRoots.emplace_back(cid);
                            m_pSampleCenterMap[cid] = m_pTreeRoots[item.index].centerid;
                        }
                        m_pSampleCenterMap[-1 - m_pTreeRoots[item.index].centerid] = item.index;
                    }
                    else {
                        unsigned maxCount = 0;
                        for (int k = 0; k < m_iBKTKmeansK; k++) if (args.counts[k] > maxCount) maxCount = args.counts[k];
                        for (int k = 0; k < m_iBKTKmeansK; k++) {
                            if (args.counts[k] == 0) continue;
                            unsigned cid = (reverseIndices == nullptr) ? localindices[item.first + args.counts[k] - 1] : reverseIndices->at(localindices[item.first + args.counts[k] - 1]);
                            m_pTreeRoots.emplace_back(cid);
                            if (args.counts[k] > 1) ss.push(BKTStackItem((unsigned)(m_pTreeRoots.size() - 1), item.first, item.first + args.counts[k] - 1, item.debug && (args.counts[k] == maxCount)));
                            item.first += args.counts[k];
                        }
                    }
                }
                m_pTreeRoots[item.index].childEnd = (unsigned)m_pTreeRoots.size();
            }
            m_pTreeRoots.emplace_back(-1);
        }
    }

    inline std::uint64_t BufferSize() const
    {
        return sizeof(int) + sizeof(unsigned) * m_iTreeNumber +
            sizeof(unsigned) + sizeof(BKTNode) * m_pTreeRoots.size();
    }

    // void SaveTrees(std::shared_ptr<Helper::DiskIO> p_out) const
    // {
    //     std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
    //     IOBINARY(p_out, WriteBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
    //     IOBINARY(p_out, WriteBinary, sizeof(unsigned) * m_iTreeNumber, (char*)m_pTreeStart.data());
    //     unsigned treeNodeSize = (unsigned)m_pTreeRoots.size();
    //     IOBINARY(p_out, WriteBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
    //     IOBINARY(p_out, WriteBinary, sizeof(BKTNode) * treeNodeSize, (char*)m_pTreeRoots.data());
    // }

    // ErrorCode SaveTrees(std::string sTreeFileName) const
    // {
    //     auto ptr = f_createIO();
    //     if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
    //     return SaveTrees(ptr);
    // }

    // ErrorCode LoadTrees(char* pBKTMemFile)
    // {
    //     m_iTreeNumber = *((int*)pBKTMemFile);
    //     pBKTMemFile += sizeof(int);
    //     m_pTreeStart.resize(m_iTreeNumber);
    //     memcpy(m_pTreeStart.data(), pBKTMemFile, sizeof(unsigned) * m_iTreeNumber);
    //     pBKTMemFile += sizeof(unsigned)*m_iTreeNumber;

    //     unsigned treeNodeSize = *((unsigned*)pBKTMemFile);
    //     pBKTMemFile += sizeof(unsigned);
    //     m_pTreeRoots.resize(treeNodeSize);
    //     memcpy(m_pTreeRoots.data(), pBKTMemFile, sizeof(BKTNode) * treeNodeSize);
    //     if (m_pTreeRoots.size() > 0 && m_pTreeRoots.back().centerid != -1) m_pTreeRoots.emplace_back(-1);
    //     return ErrorCode::Success;
    // }

    // ErrorCode LoadTrees(std::shared_ptr<Helper::DiskIO> p_input)
    // {
    //     IOBINARY(p_input, ReadBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
    //     m_pTreeStart.resize(m_iTreeNumber);
    //     IOBINARY(p_input, ReadBinary, sizeof(unsigned) * m_iTreeNumber, (char*)m_pTreeStart.data());

    //     unsigned treeNodeSize;
    //     IOBINARY(p_input, ReadBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
    //     m_pTreeRoots.resize(treeNodeSize);
    //     IOBINARY(p_input, ReadBinary, sizeof(BKTNode) * treeNodeSize, (char*)m_pTreeRoots.data());

    //     if (m_pTreeRoots.size() > 0 && m_pTreeRoots.back().centerid != -1) m_pTreeRoots.emplace_back(-1);
    //     return ErrorCode::Success;
    // }

    // ErrorCode LoadTrees(std::string sTreeFileName)
    // {
    //     auto ptr = f_createIO();
    //     if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
    //     return LoadTrees(ptr);
    // }

    template <typename T>
    void SearchTrees(const float* data, DISTFUNC fComputeDistance, const float* p_query, const int nprobe) const
    {
        using queue = std::priority_queue<std::pair<float, unsigned>>;
        for (char i = 0; i < m_iTreeNumber; i++) {
            const BKTNode& node = m_pTreeRoots[m_pTreeStart[i]];
            if (node.childStart < 0) {
                p_space.m_SPTQueue.insert(NodeDistPair(m_pTreeStart[i], fComputeDistance(p_query, data[node.centerid], data.C())));
            } else {
                for (unsigned begin = node.childStart; begin < node.childEnd; begin++) {
                    unsigned index = m_pTreeRoots[begin].centerid;
                    p_space.m_SPTQueue.insert(NodeDistPair(begin, fComputeDistance(p_query, data[index], data.C())));
                }
            }
        }
        while (!p_space.m_SPTQueue.empty())
        {
            NodeDistPair bcell = p_space.m_SPTQueue.pop();
            const BKTNode& tnode = m_pTreeRoots[bcell.node];
            if (tnode.childStart < 0) {
                if (!p_space.CheckAndSet(tnode.centerid)) {
                    p_space.m_iNumberOfCheckedLeaves++;
                    p_space.m_NGQueue.insert(NodeDistPair(tnode.centerid, bcell.distance));
                }
                if (p_space.m_iNumberOfCheckedLeaves >= p_limits) break;
            }
            else {
                if (!p_space.CheckAndSet(tnode.centerid)) {
                    p_space.m_NGQueue.insert(NodeDistPair(tnode.centerid, bcell.distance));
                }
                for (unsigned begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                    unsigned index = m_pTreeRoots[begin].centerid;
                    p_space.m_SPTQueue.insert(NodeDistPair(begin, fComputeDistance(p_query, data[index], data.C())));
                } 
            }
        }
    }

private:
    std::vector<unsigned> m_pTreeStart;
    std::vector<BKTNode> m_pTreeRoots;
    std::unordered_map<unsigned, unsigned> m_pSampleCenterMap;

public:
    std::unique_ptr<std::shared_timed_mutex> m_lock;
    int m_iTreeNumber, m_iBKTKmeansK, m_iBKTLeafSize, m_iSamples, m_bfs;
    float m_fBalanceFactor;
};
