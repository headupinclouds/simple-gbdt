/// @Brief: tree structures define from ELF
/// @Date: 2012Äê5ÔÂ28ÈÕ 12:27:04
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
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        // NOTE: There is no need to save the m_trainSamples
        ar & m_featureNr;
        ar & m_value;
        ar & m_toSmallerEqual;
        ar & m_toLarger;
    }
    
    ~node_()
    {
        if(m_toLarger) delete m_toLarger;
        if(m_toSmallerEqual) delete m_toSmallerEqual;
    }
    
} node;

typedef struct nodeReduced_
{
    node* m_node;
    uint m_size;
} nodeReduced;

