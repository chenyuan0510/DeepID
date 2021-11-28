## DeepID
A deep learning model for accurate diagnosis of infection using antibody repertoires

### Guides
Running DeepID requires the python (3.7 version or later) runtime environment.
1) Make sure that [paddlepaddle](https://github.com/paddlepaddle/paddle) 1.8.4 has installed for current python environment; 
2) The RLM.pdparams, SLM.pdparams, RLM.py and SLM.py are necessary files for DeepID model. RLM.pdparams and SLM.pdparams are the trianed models of RLM and SLM. RLM.py and SLM.py are the scripts are used to predict the test set;
3) Here we provide a test set for example: test_repertoire_level_features.npy and test_sequence_level_features.npy are 547 repertoire-level features and  160 sequence-level features, respectively. The names and order of these features are listed in the Feature names.xlsx;
4) test_repertoire_level_features.npy is a 120*547 matrix
5) Run the RLM.py and SLM.py to get two output files named RLM_test_rlt.csv and SLM_rlt_4feat.csv, respectively. The five columns of the CSV files are Probability of 0 (infection), Probability of 1 (healthy), samples are predicted to be class 0, samples are predicted to be class 1 and the Observed category.
6) The user also can apply it to their own data by replacing the input data (test_repertoire_level_features.npy and test_sequence_level_features.npy).

 1) MATLAB is the tool of Chi-MIC;    
 2)  Make sure that c++ has installed in your computer for compilation;  In the matlab runtime environment, use the ```mex -setup``` command to set: <use of'Microsoft Visual C++' for C++ language compilation>
 3)  Running program “make.m” to compile the equipartitionYaxis2.c, getsuper2var.c and getmutualI2var_fix4.c to mex files;
> The first column of data is Y (dependent variable), the rest of the columns (X) independent variable;
    
    make;  
    num=randperm(size(data,1));   
    data=data(num',:);% scramble the samples  

 while Y is numerical data
    sample_num=size(data,1); 
    [MIC,bestc,mycertain,certain_seg,mutual_I_1,mutual_I_2]=MIC_OIC_chi_1_1(data,sample_num^0.55,5,sample_num);
    
 while Y is discrete data
    [MIC,~]=MIC_OIC_chi_1_1_class(data,0.55,5)
    
Source: 
Chen Yuan, Zeng Ying, Luo Feng*, Yuan Zheming*. [A New Algorithm to Optimize Maximal Information Coefficient.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0157567) PLoS ONE, 2016 11(6): e0157567
 Contact me：chenyuan0510@126.com
