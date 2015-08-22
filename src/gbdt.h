/// @Brief: gbdt class
/// @Date: 2012Äê5ÔÂ28ÈÕ 12:27:04
/// @Author: wangben

#ifndef __GBDT_H__
#define __GBDT_H__

#include <deque>
#include <string>
#include <fstream>

#include "tree.h"
#include "ml_data.h"

// TODO: Create Impl and hide this
#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>

class GBDT
{
public:
    GBDT();

    ~GBDT();

    bool Init();

    bool Train(const Data& data);

    void PredictAllOutputs ( const Data& data, T_VECTOR& predictions);

    bool LoadConfig(const std::string& conf_file);
    
    void setMaxEpochs(unsigned int n) { m_max_epochs = n; }
    void setMaxTreeLeafs(unsigned int n) { m_max_tree_leafes = n; }
    void setFeatureSubspaceSize(unsigned int n) { m_feature_subspace_size = n; }
    void setLearningRate(double r) { m_lrate = r; }
    void setDataSampleRatio(double r) { m_data_sample_ratio = r; }
    
    unsigned int getMaxEpochs() const { return m_max_epochs; }
    unsigned int getMaxTreeLeafs() const { return m_max_tree_leafes; }
    unsigned int getFeatureSubspaceSize()  const { return m_feature_subspace_size; }
    double getLearningRate() const { return m_lrate; }
    double getDataSampleRatio() const { return m_data_sample_ratio; }
    double getGlobalMean() const { return m_global_mean; }
    
    // Original serialization:
    void SaveWeights(const std::string& model_file);
    void LoadWeights(const std::string& model_file);
    
    std::vector<node> & getTrees() { return m_trees; }
    
private:
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_lrate;
        ar & m_train_epoch;
        ar & m_global_mean;
        ar & m_trees;
    }
        
    bool ModelUpdate(const Data& data, unsigned int train_epoch, double& rmse);
    
    void TrainSingleTree(
        node * n,
        std::deque<nodeReduced> &largestNodes,
        const Data& data,
        bool* usedFeatures,
        T_DTYPE* inputTmp, 
        T_DTYPE* inputTargetsSort,
        int* sortIndex,
        const int* randFeatureIDs
        );
    
    T_DTYPE predictSingleTree(node* n, const Data& data, int data_index);
    
    void cleanTree ( node* n );
    
    void SaveTreeRecursive ( node* n, std::fstream &f );
    void LoadTreeRecursive ( node* n, std::fstream &f , std::string prefix);

private:
    
    //node * m_trees;
    std::vector<node> m_trees;
    unsigned int m_max_epochs;
    unsigned int m_max_tree_leafes;
    unsigned int m_feature_subspace_size;
    bool m_use_opt_splitpoint;
    double m_lrate;
    unsigned int m_train_epoch;
    float m_data_sample_ratio;
    
    T_VECTOR m_tree_target;

    T_DTYPE m_global_mean;
    
}; //end of class GBDT


#endif /* __GBDT_H__ */
