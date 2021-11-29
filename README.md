## DeepID
A deep learning model for accurate diagnosis of infection using antibody repertoires

### Guides
#### Quick start:
1) Running DeepID requires the python (3.7 version or later) runtime environment; 
2) Make sure that [paddlepaddle](https://github.com/paddlepaddle/paddle) 1.8.4 has installed for current python environment; 
3) Download the RLM.pdparams, SLM.pdparams, RLM.py and SLM.py, DeepID.py, test_repertoire_level_features.npy, test_sequence_level_features.npy and y_test.npy to the running directory;
4) The command for evaluating RLM on the test_repertoire_level_features is: 
   ```
   python RLM.py test_repertoire_level_features y_test RLM_test_rlt.csv
   ```
5) The command for evaluating SLM on the test_sequence_level_features is: 
   ```
   python SLM.py test_sequence_level_features y_test SLM_rlt_4feat.csv
   ```
6) The command for evaluating DeepID on the test set is (Must run RLM and SLM first):
   ```
   DeepID.py test_rlt.csv rlt_4feat.csv DeepID_rlt.csv
   ```
#### File details:
1) 

7) The RLM.pdparams, SLM.pdparams, RLM.py and SLM.py are necessary files for DeepID model. RLM.pdparams and SLM.pdparams are the trianed models of RLM and SLM. RLM.py and SLM.py are the scripts are used to predict the test set;
8) Here we provide a test set for example: test_repertoire_level_features.npy and test_sequence_level_features.npy are 547 repertoire-level features and  160 sequence-level features, respectively. The names and order of these features are listed in the Feature names.xlsx;
9) test_repertoire_level_features.npy is a 120\*547 matrix with 120 samples and 547 features; test_sequence_level_features.npy is a 120\*160\*160 matrix, in which the three dimensions are samples, clones and sequence_level_features, respectively;
10) Run the RLM.py and SLM.py to get two output files named RLM_test_rlt.csv and SLM_rlt_4feat.csv, respectively. The five columns of the CSV files are Probability of 0 (infection), Probability of 1 (healthy), samples are predicted to be class 0, samples are predicted to be class 1 and the Observed category.
11) The user also can apply it to their own data by replacing the input data (test_repertoire_level_features.npy and test_sequence_level_features.npy);
12) Here, only four sequence-level features including complexity of reads, clone fraction, KF8, and F5 were used for testing as the saved_feat in SLM.py was set to [158,159,114,58].
    
### Source: 
Yuan Chen, Zhiming Ye, Yanfang Zhang, et al. A deep learning model for accurate diagnosis of infection using antibody repertoires.

If you have any questions or problems, please e-mail chenyuan0510@126.com
