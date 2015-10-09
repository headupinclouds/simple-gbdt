#include <iostream>
#include <deque>
#include <algorithm>
#include <set>
#include <sstream>

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <iomanip>
#include <cmath>
#include <limits>

#include "gbdt.h"

#if HAS_TBB
#  include "tbb/parallel_sort.h"
#endif

//using namespace tbb;

using std::cout;
using std::endl;
using std::deque;
using std::set;
using std::string;

static bool compareNodeReduced ( nodeReduced n0, nodeReduced n1 )
{
    return n0.m_size < n1.m_size;
}

static int64_t Milliseconds()
{
    struct timeval t;    
    ::gettimeofday(&t, NULL);    
    int64_t curTime;    
    curTime = t.tv_sec;    
    curTime *= 1000;              // sec -> msec    
    curTime += t.tv_usec / 1000;  // usec -> msec    
    return curTime;
}

GBDT::GBDT()
{
    m_max_epochs = 400;
    m_global_mean = 0.0;
    m_max_tree_leafes = 20;
    m_feature_subspace_size = 40;
    m_use_opt_splitpoint = true;
    m_lrate = 0.01;
    m_train_epoch = 0;
    m_data_sample_ratio = 0.4;
}


GBDT::~GBDT()
{
    for(auto & n : m_trees)
        n.destroy();
}

bool GBDT::Init()
{
    //m_trees = new node[m_max_epochs];
    m_trees.resize(m_max_epochs);
    
    for (unsigned int i = 0; i < m_max_epochs ; ++i )
    {
        m_trees[i].m_featureNr = -1;
        m_trees[i].m_value = 1e6;
        m_trees[i].m_toSmallerEqual = 0;
        m_trees[i].m_toLarger = 0;
        m_trees[i].m_trainSamples.clear();
    }

    srand(time(0));
#if 0
    cout << "configure--------" << endl;
    cout <<  "  max_epochs: " << m_max_epochs << endl;
    cout <<  "  max_tree_leafes: " << m_max_tree_leafes << endl;
    cout <<  "  feature_subspace_size: " << m_feature_subspace_size << endl;
    cout <<  "  use_opt_splitpoint: " << m_use_opt_splitpoint << endl;
    cout <<  "  learn_rate: " << m_lrate << endl;
    cout <<  "  data_sample_ratio: " << m_data_sample_ratio << endl;
    cout << endl;
#endif
    
    return true;
}

// https://www.ross.click/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/
static inline void loadbar(unsigned int x, unsigned int n, unsigned int w = 50)
{
    if ( (x != n) && (x % (n/100+1) != 0) ) return;
    
    float ratio  =  x/(float)n;
    int   c      =  ratio * w;
    
    std::cout << std::setw(3) << (int)(ratio*100) << "% [";
    for (int x=0; x<c; x++) std::cout << "=";
    for (int x=c; x<w; x++) std::cout << " ";
    std::cout << "]\r" << std::flush;
}

bool GBDT::Train(const Data& data)
{
    m_tree_target.resize( data.m_target.size() );
    m_feature_subspace_size = 
        m_feature_subspace_size > data.m_dimension ? data.m_dimension : m_feature_subspace_size;

    double pre_rmse = -1;
    unsigned int train_epoch =0;
    //TODO or rmse rise up
    for ( ; train_epoch < m_max_epochs; train_epoch++ )
    {
        double rmse = 0.0;
        
        loadbar(train_epoch, m_max_epochs, 128);

        ModelUpdate(data, train_epoch, rmse);
        
        //if (pre_rmse < rmse && pre_rmse != -1)
        //if (pre_rmse - rmse < ( m_lrate * 0.001 ) && pre_rmse != -1)
        if (pre_rmse < rmse && pre_rmse != -1)
        {
            //cout << "debug: rmse:" << rmse << " " << pre_rmse << " " << pre_rmse - rmse << std::endl;
            break;
        }
        pre_rmse = rmse;
    }
    m_train_epoch = train_epoch - 1;
    
    // No need to store empty trees beyond max training epoch:
    m_trees.erase(m_trees.begin() + train_epoch, m_trees.end());
    
    return true;
}

bool GBDT::ModelUpdate(const Data& data, unsigned int train_epoch, double& rmse)
{
    //int64_t t0 = Milliseconds();

    int nSamples = data.m_num;
    unsigned int nFeatures = data.m_dimension;
    
    std::deque<bool> usedFeatures(data.m_dimension);
    std::vector<T_DTYPE> inputTmp((nSamples+1)*m_feature_subspace_size);
    std::vector<T_DTYPE> inputTargetSort((nSamples+1)*m_feature_subspace_size);
    std::vector<int> sortIndex(nSamples);
    
    //----first epoch----
    if (train_epoch == 0)
    {
        double mean = 0.0;
        if ( true )
        {
            for ( unsigned int  j=0; j< data.m_num ;j++ )
                mean += data.m_target[j];
            mean /= ( double ) data.m_num;
        }
        m_global_mean = mean;
        //std::cout << "globalMean:"<< mean<<" "<<std::flush;

        //align by targets mean
        for ( unsigned int j=0 ; j<data.m_num ; j++ )
            m_tree_target[j] = data.m_target[j] - mean;
    }
    
    deque< nodeReduced > largestNodes;
    //----init feature mask----
    for ( unsigned int j=0; j< data.m_dimension; j++ )
        usedFeatures[j] = false;

    //----data should be sampled !!!!----
    int data_sample_num = int(nSamples * m_data_sample_ratio);
    if( data_sample_num < 10 ) data_sample_num = nSamples;
    
    m_trees[train_epoch].m_trainSamples.resize(data_sample_num);
    auto &trainSamples = m_trees[train_epoch].m_trainSamples;

    set<int> used_data_ids;
    int sampled_count = 0;
    while (sampled_count < data_sample_num)
    {
        int id = rand()%nSamples;
        if ( used_data_ids.find(id) == used_data_ids.end() ) //can't find the id
        {
            trainSamples[sampled_count] = id;
            sampled_count++;
            used_data_ids.insert(id);
        }
    }
    ///////////////////
    
    //----init first node for split----
    nodeReduced firstNode;
    firstNode.m_node = & ( m_trees[train_epoch] );
    firstNode.m_size = data_sample_num;
    largestNodes.push_back ( firstNode );
    push_heap( largestNodes.begin(), largestNodes.end(), compareNodeReduced );  //heap for select largest num node

    //----sample feature----
    int randFeatureIDs[m_feature_subspace_size];
    // this tmp array is used to fast drawing a random subset
    if ( m_feature_subspace_size < data.m_dimension ) // select a random feature subset
    {
        for ( unsigned int i=0;i<m_feature_subspace_size;i++ )
        {
            unsigned int idx = rand() % nFeatures; //
            while ( usedFeatures[idx] || (data.m_valid_id.find(idx) == data.m_valid_id.end() ) ) //TODO check valid num;
                idx = rand() % nFeatures;
            randFeatureIDs[i] = idx;
            usedFeatures[idx] = true;
        }
    }
    else  // take all features
        for ( unsigned int i=0;i<m_feature_subspace_size;i++ )
            randFeatureIDs[i] = i;

    //---- train the tree loop wise----
    // call trainSingleTree recursive for the largest node
    for ( unsigned int j=0; j<m_max_tree_leafes; j++ )
    {
        node* largestNode = largestNodes[0].m_node;
        
        //std::cout << "Train tree: " << j << "/" << m_max_tree_leafes << " = " << float(j)/m_max_tree_leafes << std::endl;

        TrainSingleTree(
            largestNode, 
            largestNodes, 
            data, 
            &usedFeatures[0],
            &inputTmp[0],
            &inputTargetSort[0],
            &sortIndex[0],
            randFeatureIDs
            );
        
    }
    // unmark the selected inputs
    for ( unsigned int i=0;i<nFeatures;i++ )
        usedFeatures[i] = false;
    
    //----delete the train lists per node, they are not necessary for prediction----
    cleanTree ( & ( m_trees[train_epoch] ) );

    // update the targets/residuals and calc train error
    double trainRMSE = 0.0;
    //fstream f("tmp/a0.txt",ios::out);
    for ( int j=0;j<nSamples;j++ )
    {
        T_DTYPE p = predictSingleTree ( & ( m_trees[train_epoch] ), data, j);

        //f<<p<<endl;
        m_tree_target[j] -= m_lrate * p;
        double err = m_tree_target[j];
        trainRMSE += err * err;
    }
    rmse = sqrt ( trainRMSE/ ( double ) nSamples );
    
    //cout<<"RMSE:"<< rmse <<" " << trainRMSE << " "<<std::flush;
    //cout<<"cost: " << Milliseconds() -t0<<"[ms]"<<endl;
    
    return true;
}

void GBDT::cleanTree ( node* n )
{
    n->m_trainSamples.clear();

    if ( n->m_toSmallerEqual )
        cleanTree ( n->m_toSmallerEqual );
    if ( n->m_toLarger )
        cleanTree ( n->m_toLarger );
}

template <typename T>
static inline constexpr T pow2(const T& x) { return x*x; }

template <typename T>
T rmse(const T &sumLow, const T &sum2Low, const T &cntLow, const T &sumHi, const T &sum2Hi, const T&cntHi)
{
    double rmse = ((sum2Low/cntLow) - pow2(sumLow/cntLow)) * cntLow;
    rmse += ((sum2Hi/cntHi) - pow2(sumHi/cntHi)) * cntHi;
    rmse = std::sqrt(rmse/(cntLow+cntHi));
    return rmse;
}

void GBDT::TrainSingleTree(
    node * n,
    std::deque<nodeReduced> &largestNodes,
    const Data& data,
    bool* usedFeatures,
    T_DTYPE* inputTmp, 
    T_DTYPE* inputTargetsSort,
    int* sortIndex,
    const int* randFeatureIDs
    )
{
    unsigned int nFeatures = data.m_dimension;
    
    // break criteria: tree size limit or too less training samples
    unsigned int nS = largestNodes.size();
    if ( nS >= m_max_tree_leafes || n->m_trainSamples.size() <= 1 )
        return;

    // delete the current node (is currently the largest element in the heap)
    if ( largestNodes.size() > 0 )
    {
        //largestNodes.pop_front();
        pop_heap ( largestNodes.begin(),largestNodes.end(),compareNodeReduced );
        largestNodes.pop_back();
    }

    // the number of training samples in this node
    int nNodeSamples = n->m_trainSamples.size();

    // precalc sums and squared sums of targets
    double sumTarget = 0.0, sum2Target = 0.0;
    for ( int j=0;j<nNodeSamples;j++ )
    {
        T_DTYPE v = m_tree_target[n->m_trainSamples[j]];
        sumTarget += v;
        sum2Target += v*v;
    }
    
    const T_DTYPE infinity = std::numeric_limits<T_DTYPE>::max();

    int bestFeature = -1;
    T_DTYPE bestFeatureRMSE = infinity;
    T_DTYPE optFeatureSplitValue = infinity;

    //TODO check m_feature_subspace_size not larger than nFeatures!!
    // search optimal split point in all tmp input features
    for ( unsigned int i=0;i<m_feature_subspace_size;i++ )
    {
        // search the optimal split value, which reduce the RMSE the most
        T_DTYPE optimalSplitValue = 0.0;
        double rmseBest = infinity;
        int bestPos = -1;
        double sumLow = 0.0, sum2Low = 0.0, sumHi = sumTarget, sum2Hi = sum2Target, cntLow = 0.0, cntHi = nNodeSamples;
        T_DTYPE* ptrInput = inputTmp + i * nNodeSamples;
        T_DTYPE* ptrTarget = inputTargetsSort + i * nNodeSamples;

        //  copy current feature into preInput
        int nr = randFeatureIDs[i];
        for ( int j=0;j<nNodeSamples;j++ )
            ptrInput[j] = data.m_data[ n->m_trainSamples[j] ][nr];  //line :n->m_trainSamples[j] , row:nr

     
        if(m_use_opt_splitpoint)  // search for the optimal threshold value, goal: best RMSE reduction split
        {
            // fast sort of the input dimension
            std::vector<std::pair<T_DTYPE, int> > list(nNodeSamples);
            for(int j=0;j<nNodeSamples;j++)
            {
                list[j].first = ptrInput[j];
                list[j].second = j;
            }

#if HAS_TBB
            tbb::parallel_sort(list.begin(),list.end());
#else
            std::sort(list.begin(), list.end());
#endif
            for(int j=0;j<nNodeSamples;j++)
            {
                ptrInput[j] = list[j].first;
                sortIndex[j] = list[j].second;
                ptrTarget[j] = m_tree_target[n->m_trainSamples[sortIndex[j]]];
            }
            
            for(int j = 0; j < (nNodeSamples -1); j++)
            {
                T_DTYPE t = ptrTarget[j];
                sumLow += t;
                sum2Low += t*t;
                sumHi -= t;
                sum2Hi -= t*t;
                cntLow += T_DTYPE(1);
                cntHi -= T_DTYPE(1);

                T_DTYPE v0 = ptrInput[j], v1 = ptrInput[j+1];
                if ( v0 == v1 ) // skip equal successors
                    continue;

                //rmse:
                double rmse = ::rmse(sumLow, sum2Low, cntLow, sumHi, sum2Hi, cntHi);
                if ( rmse < rmseBest )
                {
                    optimalSplitValue = v0;
                    rmseBest = rmse;
                }
            }
        }
        else // use a random threshold value
        {
            for ( int j=0;j<nNodeSamples;j++ )
                ptrTarget[j] = m_tree_target[n->m_trainSamples[j]];
            
            T_DTYPE* ptrInput = inputTmp + i * nNodeSamples;//TODO: del ????
            assert(nNodeSamples > 0);
            bestPos = rand() % nNodeSamples;
            optimalSplitValue = ptrInput[bestPos];
            sumLow = sum2Low = cntLow = sumHi = sum2Hi = cntHi = 0.0;
            for ( int j=0;j<nNodeSamples;j++ )
            {
                //T_DTYPE v = ptrInput[j];
                T_DTYPE t = ptrTarget[j];
                if ( ptrInput[j] <= optimalSplitValue )
                {
                    sumLow += t;
                    sum2Low += t*t;
                    cntLow += 1.0;
                }
                else
                {
                    sumHi += t;
                    sum2Hi += t*t;
                    cntHi += 1.0;
                }
            }
            rmseBest = ::rmse(sumLow, sum2Low, cntLow, sumHi, sum2Hi, cntHi);
        }

        if ( rmseBest < bestFeatureRMSE )
        {
            bestFeature = i;
            bestFeatureRMSE = rmseBest;
            optFeatureSplitValue = optimalSplitValue;
        }
    }

    n->m_featureNr = randFeatureIDs[bestFeature];
    n->m_value = optFeatureSplitValue;
    assert(optFeatureSplitValue != infinity);

    if ( n->m_featureNr < 0 || n->m_featureNr >= (int)nFeatures )
    {
        //cout<<"f="<<n->m_featureNr<<endl;
        assert ( false );
    }

    // count the samples of the low node
    int cnt = 0;
    for ( int i=0;i<nNodeSamples;i++ )
    {
        int nr = n->m_featureNr;
        if ( data.m_data[ n->m_trainSamples[i] ][nr] <= optFeatureSplitValue )
            cnt++;
    }

    std::vector<int> lowList(cnt);
    std::vector<int> hiList(nNodeSamples-cnt);
    
    //int* lowList = new int[cnt];
    //int* hiList = new int[nNodeSamples-cnt];
    //if ( cnt == 0 )
    //    lowList = 0;
    //if ( nNodeSamples-cnt == 0 )
    //    hiList = 0;

    int lowCnt = 0, hiCnt = 0;
    double lowMean = 0.0, hiMean = 0.0;
    for ( int i=0;i<nNodeSamples;i++ )
    {
        int nr = n->m_featureNr;
        if ( data.m_data[ n->m_trainSamples[i] ][nr] <= optFeatureSplitValue )
        {
            lowList[lowCnt] = n->m_trainSamples[i];
            lowMean += m_tree_target[n->m_trainSamples[i]];
            lowCnt++;
        }
        else
        {
            hiList[hiCnt] = n->m_trainSamples[i];
            hiMean += m_tree_target[n->m_trainSamples[i]];
            hiCnt++;
        }
    }
    lowMean /= lowCnt;
    hiMean /= hiCnt;
    
    assert( !( hiCnt+lowCnt != nNodeSamples || lowCnt != cnt ) );
    ///////////////////////////
    
    // break, if too less samples
    if ( lowCnt < 1 || hiCnt < 1 )
    {
        n->m_featureNr = -1;
        n->m_value = lowCnt < 1 ? hiMean : lowMean;
        
        assert(!std::isnan(n->m_value));
        assert(!std::isinf(n->m_value));
        
        n->m_toSmallerEqual = 0;
        n->m_toLarger = 0;
        n->m_trainSamples.clear();

        nodeReduced currentNode;
        currentNode.m_node = n;
        currentNode.m_size = 0;
        largestNodes.push_back ( currentNode );
        push_heap ( largestNodes.begin(), largestNodes.end(), compareNodeReduced );

        return;
    }

    // prepare first new node
    n->m_toSmallerEqual = new node;
    n->m_toSmallerEqual->m_featureNr = -1;
    n->m_toSmallerEqual->m_value = lowMean;
    
    n->m_toSmallerEqual->m_toSmallerEqual = 0;
    n->m_toSmallerEqual->m_toLarger = 0;
    n->m_toSmallerEqual->m_trainSamples = lowList;
    assert(lowList.size() == lowCnt);

    // prepare second new node
    n->m_toLarger = new node;
    n->m_toLarger->m_featureNr = -1;
    n->m_toLarger->m_value = hiMean;
    
    n->m_toLarger->m_toSmallerEqual = 0;
    n->m_toLarger->m_toLarger = 0;
    n->m_toLarger->m_trainSamples = hiList;
    assert(hiList.size() == hiCnt);

    // add the new two nodes to the heap
    nodeReduced lowNode, hiNode;
    lowNode.m_node = n->m_toSmallerEqual;
    lowNode.m_size = lowCnt;
    hiNode.m_node = n->m_toLarger;
    hiNode.m_size = hiCnt;

    largestNodes.push_back ( lowNode );
    push_heap ( largestNodes.begin(), largestNodes.end(), compareNodeReduced );

    largestNodes.push_back ( hiNode );
    push_heap ( largestNodes.begin(), largestNodes.end(), compareNodeReduced );
    
}

// Original code (deprecated: sequential=0.28 vs recursive=0.35)
#define DO_RECURSIVE 0

T_DTYPE GBDT::predictSingleTree ( node* n, const Data& data, int data_index )
{
    assert(n != 0);
    
#if DO_RECURSIVE
    const auto &nFeatures = data.m_dimension;
    const auto &nr = n->m_featureNr;
    
    assert(!(nr < -1 || nr >= nFeatures));
    
    // here, on a leaf: predict the constant value
    if ( n->m_toSmallerEqual == 0 && n->m_toLarger == 0 )
        return n->m_value;

    assert(!(nr < 0 || nr >= nFeatures));
    
    const auto & thresh = n->m_value;
    const auto & feature = data.m_data[data_index][nr];
    return predictSingleTree ( (feature <= thresh) ? n->m_toSmallerEqual : n->m_toLarger, data, data_index );
#else
    const auto &features = data.m_data[data_index];
    while( ! (n->m_toSmallerEqual == 0 && n->m_toLarger == 0) )
        n = (features[n->m_featureNr] <= n->m_value) ? n->m_toSmallerEqual : n->m_toLarger;

    return n->m_value;
#endif
}

void GBDT::PredictAllOutputs ( const Data& data, T_VECTOR& predictions)
{
    unsigned int nSamples = data.m_num;
    predictions.resize(nSamples);
    
    // predict all values 
    for ( unsigned int i=0; i< nSamples; i++ )
    {
        double sum = m_global_mean;
        // for every boosting epoch : CORRECT, but slower
        for ( unsigned int k=0; k<m_trees.size(); k++ )
        {
            T_DTYPE v = predictSingleTree ( & ( m_trees[k] ), data, i );
            
            assert(!std::isnan(v) && !std::isinf(v));
            
            //std::cout << "* " << v << std::endl;
            sum += m_lrate * v;  // this is gradient boosting
        }
        predictions[i] = sum;
    }
}

