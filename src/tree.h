/// @Brief: tree structures define from ELF
/// @Date: 2012Äê5ÔÂ28ÈÕ 12:27:04
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
        ar & m_featureNr;
        ar & m_value;
        ar & m_toSmallerEqual;
        ar & m_toLarger;
        ar & m_trainSamples;
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

struct nodeReduced
{
    node* m_node;
    uint m_size;
};

