/// @Brief: read dataset from file
/// @Date: 2012Äê5ÔÂ28ÈÕ 11:18:27
/// @Author: wangben

#ifndef __ML_DATA_H__
#define __ML_DATA_H__

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>

#include <vector>
#include <string>
#include <set>

#include "types.h"

class Data
{
public:
    Data(){}
    ~Data(){}
public:
    T_MATRIX m_data;
    T_VECTOR m_target;
    std::set<int> m_valid_id;
    unsigned int m_dimension;
    unsigned int m_num;
private:
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        // NOTE: There is no need to save the m_trainSamples
        ar & m_data;
        ar & m_target;
        ar & m_valid_id;
        ar & m_dimension;
        ar & m_num;
    }
    
}; //end of class Data


class DataReader
{
public:
    DataReader(){}
    ~DataReader(){}
    bool ReadDataFromL2R(const std::string& input_file, Data& data, unsigned int dimentions);
    bool ReadDataFromCVS(const std::string& input_file, Data& data);
    
private:
}; //end of class DataReader


#endif /* __ML_DATA_H__ */
