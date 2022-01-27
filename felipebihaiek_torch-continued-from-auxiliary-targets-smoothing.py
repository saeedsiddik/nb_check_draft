#!/usr/bin/env python
# coding: utf-8

# 
# References :
# 1. @abhishek and @artgor 's Parallel Programming video https://www.youtube.com/watch?v=VRVit0-0AXE
# 2. @yasufuminakama 's Amazying Notebook https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter 
# 3. @namanj27 mostly  from here https://www.kaggle.com/namanj27/new-baseline-pytorch-moa
# 
# torch BCE smoothing as implemented here https://gist.github.com/MrRobot2211
# 
# 

# ## Update:
# 1. Model updated
# 2. Changed to reduce LR in plateau
# 3. Increased Seeds

# # If you like it, Do Upvote :)

# In[ ]:


import sys 
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
#sys.path.append('..')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[ ]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sn
#import mlflow

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import mlflow

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#from tools.loaders import train_short_form_loader, test_short_form_loader


# In[ ]:


exp_name="original_torch_moa_5_folds_continued"
#mlflow.set_experiment(exp_name)


# In[ ]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            #print("******************************")
            #print("Column: ",col)
            #print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
           # print("dtype after: ",props[col].dtype)
           # print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


# In[ ]:




def train_short_form_loader(feature_file,target_file,extra_target_file=None):
    '''takes the original target and features and creates a train dataset 
    in col long format'''


    train_features = pd.read_csv(feature_file)

    train_targets = pd.read_csv(target_file)
    train_features,_= reduce_mem_usage(train_features)
    train_targets,_ = reduce_mem_usage(train_targets)


    if extra_target_file is not None:
        extra_targets = pd.read_csv(extra_target_file)
        extra_targets,_ = reduce_mem_usage(extra_targets)
        train_targets = pd.concat([train_targets,extra_targets])
        del extra_targets

    targets = train_targets.columns[1:]

    train_melt=train_targets.merge(train_features,how="left",on="sig_id")


    del train_features,train_targets


    train_melt.set_index("sig_id",inplace=True)

    #train_melt["variable"]= train_melt["variable"].astype('category')
    train_melt["cp_type"]= train_melt["cp_type"].astype('category')
    train_melt["cp_dose"]= train_melt["cp_dose"].astype('category')

    return train_melt , targets



def test_short_form_loader(feature_file):
    '''takes the original target and features and creates a train dataset 
    in col long format'''


    train_features = pd.read_csv(feature_file)

    #train_targets = pd.read_csv(target_file)
    train_features,_= reduce_mem_usage(train_features)
    #train_targets,_ = reduce_mem_usage(train_targets)

    train_melt =  train_features.copy()
    del train_features


    train_melt.set_index("sig_id",inplace=True)

    #train_melt["variable"]= train_melt["variable"].astype('category')
    train_melt["cp_type"]= train_melt["cp_type"].astype('category')
    train_melt["cp_dose"]= train_melt["cp_dose"].astype('category')

    return train_melt 


# In[ ]:


#os.listdir('../input/lish-moa') 


# In[ ]:


train,target=train_short_form_loader('../input/lish-moa/train_features.csv','../input/lish-moa/train_targets_scored.csv')


# In[ ]:


train.head()


# In[ ]:


target


# In[ ]:


#train_features = pd.read_csv('../input/train_features.csv')
#train_targets_scored = pd.read_csv('../input/train_targets_scored.csv')
#train_targets_nonscored = pd.read_csv('../input/train_targets_nonscored.csv')

#test_features = pd.read_csv('../input/test_features.csv')
#sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


#len(sample_submission)


# In[ ]:


#GENES = [col for col in train_features.columns if col.startswith('g-')]
#CELLS = [col for col in train_features.columns if col.startswith('c-')]


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# # feature Selection using Variance Encoding

# In[ ]:


def supress_controls(df):
    
    df = df[train['cp_type']!='ctl_vehicle']
    df = df.drop('cp_type', axis=1)

    return df


# In[ ]:


def map_controls(df):
    
    df['cp_type']=df['cp_type'].map({'ctl_vehicle': 0, 'trt_cp': 1})
    df['cp_type']=df['cp_type'].astype(int)
    return df

def map_dose(df):
    
    df['cp_dose']=df['cp_dose'].map({'D1': 1, 'D2': 0})
    df['cp_dose']=df['cp_dose'].astype(int)
    return df

def map_time(df):
    
    df['cp_time']=df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_time']=df['cp_time'].astype(int)
    return df


# In[ ]:


def build_preprocess(preprocesses=[map_time,map_dose,map_controls]):
    
    def preprocesser(df):
        for proc in preprocesses:
            df = proc(df)
        return df
    
    return preprocesser
    
    
    


# In[ ]:


preprocess_data=build_preprocess()


# In[ ]:


#data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})


# In[ ]:





# # CV folds

# In[ ]:


def multifold_indexer(train,target_columns,n_splits=10,random_state=12347,**kwargs):
    folds = train.copy()

    mskf = MultilabelStratifiedKFold(n_splits=n_splits,random_state=random_state,**kwargs)
    folds[ 'kfold']=0
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=train[target_columns])):
        folds.iloc[v_idx,-1] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    return folds


# # Dataset Classes

# In[ ]:


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
    


# In[ ]:


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
#         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if not  scheduler.__class__ ==  torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, scheduler, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    if scheduler.__class__ ==  torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(final_loss)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds
   
    


# In[ ]:


import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


# # Model

# In[ ]:


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,drop_rate1=0.2,drop_rate2=0.5,drop_rate3=0.8):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(drop_rate1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(drop_rate2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(drop_rate3)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x


# # Preprocessing steps

# In[ ]:


def process_data(data):
    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

# --------------------- Normalize ---------------------
#     for col in GENES:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#     for col in CELLS:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#--------------------- Removing Skewness ---------------------
#     for col in GENES + CELLS:
#         if(abs(data[col].skew()) > 0.75):
            
#             if(data[col].skew() < 0): # neg-skewness
#                 data[col] = data[col].max() - data[col] + 1
#                 data[col] = np.sqrt(data[col])
            
#             else:
#                 data[col] = np.sqrt(data[col])
    
    return data


# In[ ]:





# In[ ]:


# HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-7
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

#num_features=len(feature_cols)
#num_targets=len(target_cols)
hidden_size=512


# # Single fold training

# In[ ]:


def initialize_from_past_model(model,past_model_file):

   # pretrained_dict = torch.load('FOLD0_.pth')
    pretrained_dict = torch.load(past_model_file)
    model_dict = model.state_dict()

    pretrained_dict['dense3.bias']=pretrained_dict['dense3.bias'][:206]

    pretrained_dict['dense3.weight_g']=pretrained_dict['dense3.weight_g'][:206]

    pretrained_dict['dense3.weight_v']=pretrained_dict['dense3.weight_v'][:206]

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    


# In[ ]:


#exp_name =  "test_flow"


# In[ ]:


def run_training(X_train,y_train,X_valid,y_valid,X_test,fold, seed,verbose=False,**kwargs):
    
    seed_everything(seed)
    
   
    
    train_dataset = MoADataset(X_train, y_train)
    valid_dataset = MoADataset(X_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features= X_train.shape[1] ,
        num_targets=  y_train.shape[1],
        hidden_size=hidden_size,**kwargs
    )
    
    model.to(DEVICE)
    
    initialize_from_past_model(model,f"../input/pytorchauxtargets5f1/FOLD{fold}_original_torch_moa_5_folds_AUX.pth")
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e2, 
                                          #max_lr=5e-4, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3)
    
    loss_val = nn.BCEWithLogitsLoss()
    
    loss_tr = SmoothBCEwLogits(smoothing =0.001)
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
    #todo el guardado de los resultados se puede mover a kfold que si tiene info de los indices
    #oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    
    
    
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
        if verbose:
            print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_fn(model,scheduler, loss_val, validloader, DEVICE)
        if verbose:
            print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if verbose:
            print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof = valid_preds
        
        
        
            torch.save(model.state_dict(), f"FOLD{fold}_{exp_name}.pth")
        
        elif(EARLY_STOP == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    
    #--------------------- PREDICTION---------------------
   
    testdataset = TestDataset(X_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     model = Model(
#          num_features= X_train.shape[1] ,
#         num_targets=  y_train.shape[1],
#         hidden_size=hidden_size,**kwargs
#     )
    
#     model.load_state_dict(torch.load(f"../results/FOLD{fold}_{exp_name}.pth"))
    model.to(DEVICE)
    
    #predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# In[ ]:


def run_k_fold(folds,target_cols,test,NFOLDS, seed,verbose=False,**kwargs):
    
    
    train = folds
    test_ = test
    
    
    #oof = np.zeros((len(folds), len(target_cols)))
    oof = train[target_cols].copy()
    predictions = np.zeros((len(test), len(target_cols)))
    
    #print(test_.head())
    for fold in range(NFOLDS):
        
        #trn_idx = train[train['kfold'] != fold].reset_index().index
        #val_idx = train[train['kfold'] == fold].reset_index().index
    
        train_df = train[train['kfold'] != fold]#.reset_index(drop=True)
        valid_df = train[train['kfold'] == fold]#.reset_index(drop=True)
        
       # print(len(train_df))
        #print(len(valid_df))
        
        feature_cols = [col  for col in train_df.columns if not (col in target_cols.to_list()+['kfold'])]
        
        #print(feature_cols)
        
        X_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
        X_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
        X_test = test_[feature_cols].values
            
        oof_, pred_ = run_training(X_train,y_train,X_valid,y_valid,X_test,fold, seed,verbose,**kwargs)
        
        oof[train['kfold'] == fold] = oof_
        
        
        
        predictions += pred_ / NFOLDS
        
        
    return oof, predictions


# In[ ]:


params ={'drop_rate1':0.5,'drop_rate2':0.2,'drop_rate3':0.2}


# In[ ]:


# Averaging on multiple SEEDS

SEED = [0,12347,565657,123123,78591]
#SEED = [0]
train,target_cols = train_short_form_loader('../input/lish-moa/train_features.csv','../input/lish-moa/train_targets_scored.csv')
test = test_short_form_loader("../input/lish-moa/test_features.csv")



train = preprocess_data(train)
test = preprocess_data(test)
    

oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:
   
    folds = multifold_indexer(train,target_cols,n_splits=NFOLDS)
    
    
    oof_, predictions_ = run_k_fold(folds,target_cols,test,NFOLDS, seed,verbose=True,**params)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)

#train[target_cols] = oof
test[target_cols] = predictions


# In[ ]:


folds['kfold'].unique()


# In[ ]:


#valid_results = train.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
#valid_results

y_true = train[target_cols].values
y_pred = oof

score = 0
for i in range(len(target_cols)):
   # print(log_loss(y_true[:, i], y_pred[:, i])/ len(target_cols))
    score_ = log_loss(y_true[:, i], y_pred.iloc[:, i],labels=[0,1])
    #if score_ > 0.02:
     #   print(score_)
    score +=( score_ / len(target_cols))
    
print("CV log_loss: ", score)
    


# In[ ]:


sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission.set_index('sig_id',inplace=True)
test_features.set_index('sig_id',inplace=True)
test_features = test_features.loc[sample_submission.index]

sub = sample_submission.drop(columns=target_cols).merge(test[target_cols], on='sig_id', how='left').fillna(0)
#sub.set_index('sig_id',inplace=True)
sub.loc[test_features['cp_type']=='ctl_vehicle', target_cols] =0
sub.to_csv('./submission.csv', index=True)

