#!/usr/bin/env python
# coding: utf-8

# In[10]:

import DGGL as fg
import numpy as np
import pandas as pd
import glob
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy
import sys

input1 = sys.argv[1]
input2 = sys.argv[2]

# In[2]:

affinity_refine_df = pd.read_csv('v2007_refine_list.csv')
print('The size of refine list is {}'.format(affinity_refine_df.shape[0]))
affinity_refine_dict = {}
for i in range(affinity_refine_df.shape[0]):
    affinity_refine_dict[affinity_refine_df.iloc[i]['id']] = affinity_refine_df.iloc[i]['num']
# In[6]:

affinity_core_df = pd.read_csv('v2007_core_list.csv')
print('The size of core list is {}'.format(affinity_core_df.shape[0]))
affinity_core_dict = {}
for i in range(affinity_core_df.shape[0]):
    affinity_core_dict[affinity_core_df.iloc[i]['id']] = affinity_core_df.iloc[i]['num']

# In[ ]:
print('Generating train and test data')
root = 'v2007'
Xtrain = []
ytrain = []
Xtest = []
ytest = []
num_cplx_processed = 0
for filepath in glob.glob(root+'/*/*_protein.pdb'):
    cplx = (re.findall(r'\/(.*?)\/', filepath))[0]
    plfg = fg.protein_ligand_feature_generator(beta=float(input1),tau=float(input2))
    features = plfg.generator(filepath, root+'/'+cplx+'/'+cplx+'_ligand.mol2')
    if cplx in affinity_refine_dict.keys():
        Xtrain.append(features)
        ytrain.append(affinity_refine_dict[cplx])
    elif cplx in affinity_core_dict.keys():
        Xtest.append(features)
        ytest.append(affinity_core_dict[cplx])
        
    num_cplx_processed += 1
    if num_cplx_processed % 10 == 0: 
        print('{} protein-ligand complices are processed'.format(num_cplx_processed))        

print('Saving files.')
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
Xtest = np.array(Xtest)
ytest = np.array(ytest)
np.savetxt('Xtrain_'+input1+'_'+input2+'.csv', Xtrain)
np.savetxt('ytrain_'+input1+'_'+input2+'.csv', ytrain)
np.savetxt('Xtest_'+input1+'_'+input2+'.csv', Xtest)
np.savetxt('ytest_'+input1+'_'+input2+'.csv', ytest)


# In[ ]:
print('Training')

res = []
for i in range(20):
    reg = GradientBoostingRegressor(n_estimators=10000, max_depth = 7, min_samples_split = 3, learning_rate = 0.01, loss = 'ls', subsample = 0.3, max_features = 'sqrt')
    reg.fit(Xtrain, np.ravel(ytrain))
    pred = reg.predict(Xtest)
    R_P = scipy.stats.pearsonr(pred, np.ravel(ytest))
    RMSE = np.sqrt(mean_squared_error(pred, np.ravel(ytest)))*1.3633
    res.append([RMSE, R_P[0]])
    print("The RMSE is {:.3f} and the PCC is {:.3f}".format(RMSE, R_P[0]))

print(np.mean(np.array(res), axis = 0))
res.append(np.mean(np.array(res), axis = 0))
res = np.array(res)
np.savetxt('res_'+input1+ '_' +input2+'.csv', res)
