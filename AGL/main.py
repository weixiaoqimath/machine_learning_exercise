#!/usr/bin/env python
# coding: utf-8

# In[10]:


import AGL_1229 as AGL
import numpy as np
import pandas as pd
import glob
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy
import sys
import os


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


input1 = sys.argv[1]
input2 = sys.argv[2]
input3 = sys.argv[3]
input4 = sys.argv[4]
input5 = sys.argv[5]
input6 = sys.argv[6]

input7 = sys.argv[7]
input8 = sys.argv[8]
input9 = sys.argv[9]
input10 = sys.argv[10]
input11 = sys.argv[11]
input12 = sys.argv[12]

affinity_refine_df = pd.read_csv('v2007_refine_list.csv')
print('The size of refine list is {}'.format(affinity_refine_df.shape[0]))
affinity_refine_dict = {}
for i in range(affinity_refine_df.shape[0]):
    affinity_refine_dict[affinity_refine_df.iloc[i]['id']] = affinity_refine_df.iloc[i]['num']

affinity_core_df = pd.read_csv('v2007_core_list.csv')
print('The size of core list is {}'.format(affinity_core_df.shape[0]))
affinity_core_dict = {}
for i in range(affinity_core_df.shape[0]):
    affinity_core_dict[affinity_core_df.iloc[i]['id']] = affinity_core_df.iloc[i]['num']


print('Generating train and test data')
root = 'v2007'
Xtrain = []
Xtrain_ = []
ytrain = []
Xtest = []
Xtest_ = []
ytest = []
num_cplx_processed = 0
for filepath in glob.glob(root+'/*/*_protein.pdb'):
    cplx = (re.findall(r'\/(.*?)\/', filepath))[0]
    plfg_1 = AGL.feature_generator(kernel = input1, matrix = input2, beta=float(input3),tau=float(input4))
    plfg_2 = AGL.feature_generator(kernel = input1, matrix = input2, beta=float(input5),tau=float(input6))
    plfg_3 = AGL.feature_generator(kernel = input7, matrix = input8, beta=float(input9),tau=float(input10))
    plfg_4 = AGL.feature_generator(kernel = input7, matrix = input8, beta=float(input11),tau=float(input12))
    
    features_1 = plfg_1.generator(filepath, root+'/'+cplx+'/'+cplx+'_ligand.mol2')
    features_2 = plfg_2.generator(filepath, root+'/'+cplx+'/'+cplx+'_ligand.mol2') 
    features_3 = plfg_3.generator(filepath, root+'/'+cplx+'/'+cplx+'_ligand.mol2')
    features_4 = plfg_4.generator(filepath, root+'/'+cplx+'/'+cplx+'_ligand.mol2')
    features = np.concatenate((features_1, features_2), axis=None)
    features_ = np.concatenate((features_3, features_4), axis = None)
    if cplx in affinity_refine_dict.keys():
        Xtrain.append(features)
        Xtrain_.append((features_))
        ytrain.append(affinity_refine_dict[cplx])
    elif cplx in affinity_core_dict.keys():
        Xtest.append(features)
        Xtest_.append(features_)
        ytest.append(affinity_core_dict[cplx])
        
    num_cplx_processed += 1
    if num_cplx_processed % 10 == 0: 
        print('{} protein-ligand complices are processed'.format(num_cplx_processed))        



print('Saving files.')

features_directory = "{}/features".format(os.getcwd())
mkdir_p(features_directory)

Xtrain = np.array(Xtrain)
Xtrain_ = np.array(Xtrain_)
ytrain = np.array(ytrain)
Xtest = np.array(Xtest)
Xtest_ = np.array(Xtest_)
ytest = np.array(ytest)

np.savetxt('features/Xtrain_{}_{}_{}_{}_{}_{}.csv'.format(input1, input2, input3, input4, input5, input6), Xtrain)
np.savetxt('features/Xtrain_{}_{}_{}_{}_{}_{}.csv'.format(input7, input8, input9, input10, input11, input12), Xtrain_)
np.savetxt('features/Xtest_{}_{}_{}_{}_{}_{}.csv'.format(input1, input2, input3, input4, input5, input6), Xtest)
np.savetxt('features/Xtest_{}_{}_{}_{}_{}_{}.csv'.format(input7, input8, input9, input10, input11, input12), Xtest_)

# In[ ]:
print('Training')

pred = np.zeros(195)
for i in range(50):
    reg = GradientBoostingRegressor(n_estimators=10000, max_depth = 7, min_samples_split = 3, learning_rate = 0.01, loss = 'ls', subsample = 0.3, max_features = 'sqrt')
    reg.fit(Xtrain, np.ravel(ytrain))
    pred += reg.predict(Xtest)

for i in range(50):
    reg = GradientBoostingRegressor(n_estimators=10000, max_depth = 7, min_samples_split = 3, learning_rate = 0.01, loss = 'ls', subsample = 0.3, max_features = 'sqrt')
    reg.fit(Xtrain_, np.ravel(ytrain))
    pred += reg.predict(Xtest_)



pred = pred/100
R_P = scipy.stats.pearsonr(pred, np.ravel(ytest))
RMSE = np.sqrt(mean_squared_error(pred, np.ravel(ytest)))*1.3633
print("The RMSE is {:.3f} and the PCC is {:.3f}".format(RMSE, R_P[0]))
res = R_P
res = np.append(R_P, RMSE)

    
res_directory = "{}/res".format(os.getcwd())
mkdir_p(res_directory)

np.savetxt('res/res_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12), res)


# In[ ]:




