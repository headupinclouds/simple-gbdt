/// @Brief: tree structures define from ELF
/// @Date: 2012��5��28�� 12:27:04
/// @Author: wangben

#include "types.h"

#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>

typedef struct node_
{
    int m_featureNr;                  // decision on this feature
    T_DTYPE m_value;                  // the prediction value
    struct node_* m_toSmallerEqual;   // pointer to node, if:  feature[m_featureNr] <=  m_value
    struct node_* m_toLarger;         // pointer to node, if:  feature[m_featureNr] > m_value
    std::vector<int> m_trainSamples;  // a list of indices of the training samples in this node
    int m_nSamples;                   // the length of m_trainSamples
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_featureNr;
        ar & m_value;
        ar & m_trainSamples;
        ar & m_nSamples;
        ar & m_toSmallerEqual;
        ar & m_toLarger;
    }
    
} node;

typedef struct nodeReduced_
{
    node* m_node;
    uint m_size;
} nodeReduced;

