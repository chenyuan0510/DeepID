import csv, sys, os, math
import pandas as pd
import numpy as np

if __name__ == '__main__':
    rlmrlt = sys.argv[1]
    slmrlt = sys.argv[2]
    outputfile = sys.argv[3]
    rlm_rlt=pd.read_csv(rlmrlt)
    slm_rlt=pd.read_csv(slmrlt)
    rlm_rlt.columns=['prob0','prob1','pred_to_0','pred_to_1','true_lable']
    slm_rlt.columns=['prob0','prob1','pred_to_0','pred_to_1','true_lable']
    rlm_rlt['prob0']=rlm_rlt['prob0']+slm_rlt['prob0']
    rlm_rlt['prob1']=rlm_rlt['prob1']+slm_rlt['prob1']
    rlm_rlt['pred_to_0']=0
    rlm_rlt['pred_to_1']=0
    rlm_rlt['pred_to_0'] = np.where(rlm_rlt['prob0']>1, 1, rlm_rlt['pred_to_0'])
    rlm_rlt['pred_to_1'] = np.where(rlm_rlt['prob1']>1, 1, rlm_rlt['pred_to_1'])
    rlm_rlt.to_csv(outputfile, index=False,header=True)