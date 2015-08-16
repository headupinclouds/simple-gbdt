/// @Brief: tree structures define from ELF
/// @Date: 2012��5��28�� 12:27:04
/// @Author: wangben

#include "types.h"

#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>

struct node
{
    node() : m_featureNr(-1), m_value(T_DTYPE(0)), m_toSmallerEqual(0),  m_toLarger(0) {}
    ~node() {}

    void destroy()
    {
        if(m_toLarger) remove(m_toLarger);
        if(m_toSmallerEqual) remove(m_toSmallerEqual);
    }
    
    int m_featureNr;                  // decision on this feature
    T_DTYPE m_value;                  // the prediction value
    node* m_toSmallerEqual;   // pointer to node, if:  feature[m_featureNr] <=  m_value
    node* m_toLarger;         // pointer to node, if:  feature[m_featureNr] > m_value
    std::vector<int> m_trainSamples;  // a list of indices of the training samples in this node
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        // NOTE: There is no need to save the m_trainSamples
#if DO_SQUEEZE
        uint16_t featureNr;
        if(Archive::is_loading::value)
        {
            ar & featureNr;
            m_featureNr = featureNr;
        }
        else
        {
            featureNr = m_featureNr;
            ar & featureNr;
        }
#else
        ar & m_featureNr;
#endif

#if DO_HALF_FLOAT
        half_float::detail::uint16 half;
        if(Archive::is_loading::value)
        {
            ar & half;
            m_value = half_float::detail::half2float(half);

        }
        else
        {
            half = half_float::detail::float2half<std::round_indeterminate>(m_value);
            ar & half;
        }
#else
        ar & m_value;
#endif
        ar & m_toSmallerEqual;
        if(m_toSmallerEqual != NULL)
            ar & m_toLarger;
    }
    
protected:
    
    static void remove(node *n)
    {
        if(!n) return;
        if(n->m_toLarger) delete n->m_toLarger;
        if(n->m_toSmallerEqual) delete n->m_toSmallerEqual;
        delete n;
        n = 0;
    }
};

#if DO_HALF_FLOAT
struct MiniNode
{
    MiniNode() {}
    MiniNode(float value, int feature)
    : m_value16u(half_float::detail::float2half<std::round_indeterminate>(value))
    , m_toSmallerEqual(0)
    , m_toLarger(0)
    , m_feature(uint16_t(feature))
    , m_value(value)
    {}
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_value16u;
        ar & m_toSmallerEqual;
        if(m_toSmallerEqual)
        {
            ar & m_toLarger;
            ar & m_feature;
        }
        
        if(Archive::is_loading::value)
        {
            // We don't serialize this, but make sure it is available for fast comparisons:
            m_value = half_float::detail::half2float( m_value16u );
        }
    }
    
    // Serialize these values:
    half_float::detail::uint16 m_value16u;
    uint16_t m_toSmallerEqual;
    uint16_t m_toLarger;
    uint16_t m_feature;
    
    // Use this value for faster in memory comparison:
    float m_value;
};

struct MiniTree
{
    MiniTree() {}
    
    MiniTree(node *tree) { init(tree); }
    
    float operator()(const std::vector<float> &features) const
    {
        const auto *n = &nodes.front();
        while(n->m_toSmallerEqual != 0)
        {
            int index = (features[n->m_feature] <= n->m_value) ? n->m_toSmallerEqual : n->m_toLarger;
            n = &nodes[index];
        }
        //std::cout << n->m_value << std::endl;
        return n->m_value;
    }
    
    std::vector<MiniNode> nodes;
    
    // Boost serialization:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & nodes;
    }
    
    int init(node *tree)
    {
        int index = nodes.size();
        nodes.emplace_back(tree->m_value, tree->m_featureNr);
        if(tree->m_toSmallerEqual)
        {
            nodes[index].m_toSmallerEqual = init(tree->m_toSmallerEqual);
            nodes[index].m_toLarger = init(tree->m_toLarger);
        }
        return index;
    }
    
protected:
    
};

struct MiniForest
{
    MiniForest() {}
    
    MiniForest(float globalMean, float learningRate)
    : globalMean(globalMean)
    , learningRate(learningRate)
    {}
    
    float globalMean;
    float learningRate;
    std::vector<MiniTree> trees;
    
    float operator()(const std::vector<float> &features)
    {
        float score = globalMean;
        for(const auto & t : trees)
            score += learningRate * t(features);

        return score;
    }
    
    // Boost serialization:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & globalMean;
        ar & learningRate;
        ar & trees;
    }
};

#endif

struct nodeReduced
{
    node* m_node;
    uint m_size;
};

