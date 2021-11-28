## DeepID
A deep learning model for accurate diagnosis of infection using antibody repertoires

### Guides
Running DeepID requires the python (3.7 version or later) runtime environment.
1) Make sure that [paddlepaddle](https://github.com/paddlepaddle/paddle) 1.8.4 has installed for current python environment; 
2) The RLM.pdparams, SLM.pdparams, RLM.py and SLM.py are necessary files for DeepID model. RLM.pdparams and SLM.pdparams are the trianed models of RLM and SLM. RLM.py and SLM.py are the scripts are used to predict the test set;
3) Here we provide a test set for example: test_repertoire_level_features.npy and test_sequence_level_features.npy are 547 repertoire-level features and  160 sequence-level features, respectively. The names and order of these features are listed in the Feature names.xlsx;
4) test_repertoire_level_features.npy is a 120\*547 matrix with 120 samples and 547 features; test_sequence_level_features.npy is a 120\*160\*160 matrix, in which the three dimensions are samples, clones and sequence_level_features, respectively;
5) Run the RLM.py and SLM.py to get two output files named RLM_test_rlt.csv and SLM_rlt_4feat.csv, respectively. The five columns of the CSV files are Probability of 0 (infection), Probability of 1 (healthy), samples are predicted to be class 0, samples are predicted to be class 1 and the Observed category.
6) The user also can apply it to their own data by replacing the input data (test_repertoire_level_features.npy and test_sequence_level_features.npy);
7) Here, only four sequence-level features including complexity of reads, clone fraction, KF8, and F5 were used for testing as the saved_feat in SLM.py was set to [158,159,114,58] 
    
Source: 
Yuan Chen, Zhiming Ye, Yanfang Zhang, et al. A deep learning model for accurate diagnosis of infection using antibody repertoires.
Contact me：chenyuan0510@126.com
