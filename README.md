## DeepID
A deep learning model for accurate diagnosis of infection using antibody repertoires

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
