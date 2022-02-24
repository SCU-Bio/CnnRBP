# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:54:22 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import os
import xgboost as xgb
import h5py
from lightgbm import  LGBMClassifier as LGBM
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from HH import to_categorical,categorical_probas_to_classes,calculate_performace
from sklearn.preprocessing import scale,StandardScaler,MaxAbsScaler,maxabs_scale
from keras.layers import Dense, merge,Input,Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras import optimizers
from keras import regularizers
from keras import backend as K
from sklearn.metrics import precision_recall_curve
from keras.layers import Conv1D,MaxPooling1D,AveragePooling1D
from imblearn.over_sampling import SMOTE,SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler,NearMiss
import joblib
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
from scipy import interp

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles
'''
shu1 = pd.read_csv("E:/RBP_9606.csv")
shu2 = pd.read_csv("E:/NRBP_9606.csv")
shu3 = pd.read_csv("E:/pro_data_400/train/RBP_9606.csv")
shu4 = pd.read_csv("E:/pro_data_400/train/NRBP_9606.csv")
'''
shu1 = pd.read_csv("E:/raojun/train/Salmonella/RBP_590.csv")
shu2 = pd.read_csv("E:/raojun/train/Salmonella/NRBP_590.csv")
shu3 = pd.read_csv("E:/pro_data_400/train/RBP_590.csv")
shu4 = pd.read_csv("E:/pro_data_400/train/NRBP_590.csv")

'''
shu1 = pd.read_csv("E:/raojun/train/E_coli/RBP_561.csv")
shu2 = pd.read_csv("E:/raojun/train/E_coli/NRBP_561.csv")
shu3 = pd.read_csv("E:/pro_data_400/train/RBP_561.csv")
shu4 = pd.read_csv("E:/pro_data_400/train/NRBP_561.csv")
'''

shu1 = np.array(shu1)
shu2 = np.array(shu2)
shu3 = np.array(shu3)
shu4 = np.array(shu4)
shu_zheng = np.concatenate((shu1,shu3),axis=1)
shu_fu = np.concatenate((shu2,shu4),axis=1)
#shu = np.concatenate((shu1,shu2),axis=0)
shu = np.concatenate((shu_zheng,shu_fu),axis=0)
[row1,column1]=np.shape(shu_zheng)
[row2,column2]=np.shape(shu_fu)
#[row1,column1]=np.shape(shu1)
#[row2,column2]=np.shape(shu2)
label_P = np.ones(int(row1))
label_N = np.zeros(int(row2))
label = np.hstack((label_P,label_N))
shu=np.array(shu)
label=np.array(label)
shu = maxabs_scale(shu)
def minmax(data):
    sum_data = np.sum(data,axis=0)
    #max_data = np.max(data,axis=0)
    return (data)/(sum_data)
#shu = minmax(shu)
'''
shu5 = pd.read_csv("E:/raojun/test/E_coli/RBP_561.csv")
shu6 = pd.read_csv("E:/raojun/test/E_coli/NRBP_561.csv")
shu7 = pd.read_csv("E:/pro_data_400/test/RBP_561.csv")
shu8 = pd.read_csv("E:/pro_data_400/test/NRBP_561.csv")
'''
shu5 = pd.read_csv("E:/raojun/test/Salmonella/RBP_590.csv")
shu6 = pd.read_csv("E:/raojun/test/Salmonella/NRBP_590.csv")
shu7 = pd.read_csv("E:/pro_data_400/test/RBP_590.csv")
shu8 = pd.read_csv("E:/pro_data_400/test/NRBP_590.csv")
'''
shu5 = pd.read_csv("E:/raojun/test/human/test_RBP_9606.csv")
shu6 = pd.read_csv("E:/raojun/test/human/test_NRBP_9606.csv")
shu7 = pd.read_csv("E:/pro_data_400/test/RBP_9606.csv")
shu8 = pd.read_csv("E:/pro_data_400/test/NRBP_9606.csv")
'''
shu5 = np.array(shu5)
shu6 = np.array(shu6)
shu7 = np.array(shu7)
shu8 = np.array(shu8)
shu_zheng_test = np.concatenate((shu5,shu7),axis=1)
shu_fu_test = np.concatenate((shu6,shu8),axis=1)
#shu = np.concatenate((shu1,shu2),axis=0)
shu_test = np.concatenate((shu_zheng_test,shu_fu_test),axis=0)
[row3,column3]=np.shape(shu_zheng_test)
[row4,column4]=np.shape(shu_fu_test)
#[row1,column1]=np.shape(shu1)
#[row2,column2]=np.shape(shu2)
label_P_test = np.ones(int(row3))
label_N_test = np.zeros(int(row4))
label_test = np.hstack((label_P_test,label_N_test))

shu_test=np.array(shu_test)
label_test=np.array(label_test)


shu_test=pd.DataFrame(shu_test)
shu_test = maxabs_scale(shu_test)

#shu_test = minmax(shu_test)
#shu = SelectKBest(f_classif, k=1500).fit_transform(shu,label.ravel())
'''
xgb_model=LGBM()
#xgb_model=xgb.XGBClassifier()
xgbresult1=xgb_model.fit(shu,label.ravel())
feature_importance=xgbresult1.feature_importances_
feature_number=-feature_importance
H1=np.argsort(feature_number)
mask=H1[:1530]
shu = shu[:,mask]
joblib.dump(filename='./LR1.model',value=mask,)

print(H1)
train_data=shu[:,mask]
xgbclassifier_model=xgb.XGBClassifier()
xgbcla=xgbclassifier_model.fit(train_data,label.ravel())
feature_importance=xgbcla.feature_importances_
feature_number=-feature_importance
H2 = np.argsort(feature_number)
mask=H2[:1530]
print(H2)
X=train_data[:,mask]
'''

X = shu
y=label

X_test = shu_test
y_test = label_test
#joblib.dump(filename='./LR2.model',value=mask,)


#X_resampled_smote, y_resampled_smote = SMOTE(sampling_strategy=1 ,random_state=42).fit_sample(X,y)
#X_resampled_smote, y_resampled_smote = RandomUnderSampler(ratio={0:17538}).fit_sample(X_resampled_smote, y_resampled_smote)
#sorted(Counter(y_resampled_smote).items())

#data= X_resampled_smote
#label=y_resampled_smote
#shu=scale(data_AAindex)
#shu=scale(data)
#X=data
#y=label

[m,n]=np.shape(X)
[sample_num,input_dim]=np.shape(X)

[m_test,n_test]=np.shape(X_test)
[sample_num_test,input_dim_test]=np.shape(X_test)

out_dim=2
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5


def get_CNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'SAME', activation= 'relu',kernel_initializer = 'lecun_normal',bias_initializer='zeros'))
    model.add(AveragePooling1D(pool_size=4,strides=4,padding="SAME"))
    model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'SAME', activation= 'relu',kernel_initializer = 'lecun_normal',bias_initializer='zeros'))
    model.add(AveragePooling1D(pool_size=4,strides=4,padding="SAME"))
    #model.add(MaxPooling1D(pool_size=2,padding="SAME")) 
    model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'SAME', activation= 'relu',kernel_initializer = 'lecun_normal',bias_initializer='zeros'))
    model.add(AveragePooling1D(pool_size=4,strides=2,padding="SAME"))
    model.add(Conv1D(filters = 128, kernel_size = 2, padding = 'SAME', activation= 'relu',kernel_initializer = 'lecun_normal',bias_initializer='zeros'))
    model.add(AveragePooling1D(pool_size=4,strides=2,padding="SAME"))
    model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'SAME', activation= 'relu',kernel_initializer = 'lecun_normal',bias_initializer='zeros'))
    #model.add(AveragePooling1D(pool_size=4,strides=2,padding="SAME"))
    model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'SAME', activation= 'relu',kernel_initializer = 'lecun_normal',bias_initializer='zeros'))
    model.add(AveragePooling1D(pool_size=2,strides=2,padding="SAME"))
    #model.add(MaxPooling1D(pool_size=2,padding="SAME")) 
    model.add(Flatten())
    #model.add(LSTM(int(input_dim/4), return_sequences=False))
    #model.add(Dropout(0.5))
    #model.add(Dense(int(input_dim), activation = 'relu'))
    #model.add(Dropout(0.75))
    model.add(Dense(int(input_dim/4), activation = 'relu'))
    model.add(Dropout(0.75))
    model.add(Dense(out_dim, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'categorical_crossentropy', optimizer =optimizers.Adam(learning_rate = 0.001), metrics =['accuracy'])
    
    return model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5,factor=0.1, mode='min',threshold=0.0001)

X_resampled_smote, y_resampled_smote = SMOTE(sampling_strategy=1,random_state=42).fit_sample(X,y)
X1 = X_resampled_smote
y1 = y_resampled_smote
y_train=to_categorical(y1)#generate the resonable results
#y_train=to_categorical(y[train])
cv_clf =get_CNN_model(input_dim,out_dim)
X_train=np.reshape(X1,(-1,1,input_dim))
#X_train=np.reshape(X,(-1,1,input_dim))
#X_test_op=np.reshape(X_test,(-1,1,input_dim_test))
cv_clf.fit(X_train, 
           y_train,
           batch_size=128,
           epochs=130,
           validation_split=0.1,
           callbacks=[reduce_lr])

y_test_op=to_categorical(y_test)#generate the test
X_test_op=np.reshape(X_test,(-1,1,input_dim_test)) 
ytest=np.vstack((ytest,y_test_op))
y_test_tmp=y_test      
y_score=cv_clf.predict(X_test_op)#the output of  probability
print(y_score)  
#print(y_score)
#print(y_test_tmp)
print(y_score.shape)
yscore=np.vstack((yscore,y_score))
fpr_CNN, tpr_CNN, _ = roc_curve(y_test_op[:,0], y_score[:,0])
roc_auc = auc(fpr_CNN, tpr_CNN)
y_class_score = y_score[:,1]
p, r, thresh  = precision_recall_curve(y_test_tmp,y_class_score,pos_label = 1,sample_weight= None)

distances = np.sqrt(np.sum((np.array([1, 1]) - np.array([r, p]).T)**2, axis=1))
idx = np.argmin(distances)
best_threshold = thresh[idx]
print ("DLRBP: Optimal classification threshold: {}".format(best_threshold))
#print(thresh)
'''    
plt.plot(r, p, color='blue',
lw=2, label='CNN PR')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('recall')
plt.ylabel('precsion')
plt.title('PR')
plt.legend(loc="lower right")
plt.show()
'''
y_class = (y_class_score > 0.9).astype(bool).astype(int)
#print(y_class)
#y_class= categorical_probas_to_classes(y_score)
print(y_class)
acc, precision,npv, sensitivity, specificity, mcc,f1,bacc = calculate_performace(len(y_class), y_class, y_test_tmp)
sepscores.append([acc,precision,npv, sensitivity, specificity, mcc,f1,bacc,roc_auc])
print('GTB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,bacc=%f,roc_auc=%f'
      % (acc,precision,npv, sensitivity, specificity, mcc,f1,bacc, roc_auc))
hist=[]
cv_clf=[]
scores=np.array(sepscores)

print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("bacc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[8]*100,np.std(scores, axis=0)[8]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yscore_sum_CNN.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_CNN.csv')
fpr_CNN, tpr_CNN, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[8]
lw=2

'''
plt.plot(fpr_CNN, tpr_CNN, color='red',
lw=lw, label='CNN ROC (area = %.2f%%)' % auc_score)
'''
'''
# PARAMETERS
taxon = 590
base_train_dir = 'E:/TriPepSVM/paper/data/balanced_training/pred_{}'.format(taxon)

pred_all = []
sets_all = []
no_cv = 0
for cv_dir in os.listdir(base_train_dir):
    cv_dirname = os.path.join(base_train_dir, cv_dir)
    if os.path.isdir(cv_dirname):
        res_pos = pd.read_csv(os.path.join(cv_dirname, 'results_balanced_{}.RBP.pred.txt'.format(taxon)),
                              sep='\t', header=None,
                              names=['Uniprot_ID', 'Score', 'Class'])
        res_pos['Label'] = 1
        res_neg = pd.read_csv(os.path.join(cv_dirname, 'results_balanced_{}.NRBP.pred.txt'.format(taxon)),
                              sep='\t', header=None,
                              names=['Uniprot_ID', 'Score', 'Class'])
        res_neg['Label'] = -1
        test_res = pd.concat((res_pos, res_neg))
        pred_all.append(test_res)
        no_cv += 1
print ("Read predictions from {} CV runs".format(no_cv))
print ("Each CV run contains {} positive and {} negative proteins".format(res_pos.shape[0], res_neg.shape[0]))
# RBPPred
rbppred_pos = pd.read_csv('E:/TriPepSVM/paper/data/test/RBPPred/RBP_{}.RBPPred.pred.temp'.format(taxon), sep='\t', header=0)
rbppred_pos['Label'] = 1
rbppred_neg = pd.read_csv('E:/TriPepSVM/paper/data/test/RBPPred/NRBP_{}.RBPPred.pred.temp'.format(taxon), sep='\t', header=0)
rbppred_neg['Label'] = -1
rbppred_test_all = pd.concat((rbppred_pos, rbppred_neg))
print ("Loaded predictions for {} RBPs and {} NRBPs".format(rbppred_pos.shape[0], rbppred_neg.shape[0]))
# RNApred
rnapred_pos = pd.read_csv('E:/TriPepSVM/paper/data/test/RNAPred/RBP_{}.RNAPred.pred.txt'.format(taxon), sep='\t', header=0, names=['Name', 'Score', 'Class'])
rnapred_pos['Label'] = 1
rnapred_neg = pd.read_csv('E:/TriPepSVM/paper/data/test/RNAPred/NRBP_{}.RNAPred.pred.txt'.format(taxon), sep='\t', header=0, names=['Name', 'Score', 'Class'])
rnapred_neg['Label'] = -1
rnapred_test_all = pd.concat((rnapred_pos, rnapred_neg))
print ("Loaded predictions for {} RBPs and {} NRBPs".format(rnapred_pos.shape[0], rnapred_neg.shape[0]))
# SPOT-seq-RNA
spot_pos = pd.read_csv('E:/TriPepSVM/paper/data/test/SpotSeqRna/RBP_{}.SpotSeqRna.pred.txt'.format(taxon), sep='\t\t', header=None, names=['Name', 'Class'])
spot_pos['Score'] = spot_pos.Class == 'RNA-binding protein'
spot_pos['Label'] = 1
spot_neg = pd.read_csv('E:/TriPepSVM/paper/data/test/SpotSeqRna/NRBP_{}.SpotSeqRna.pred.txt'.format(taxon), sep='\t\t', header=None, names=['Name', 'Class'])
spot_neg['Score'] = spot_neg.Class == 'RNA-binding protein'
spot_neg['Label'] = 0
spot_test_all = pd.concat((spot_pos, spot_neg))
print ("Loaded predictions for {} RBPs and {} NRBPs".format(spot_pos.shape[0], spot_neg.shape[0]))
fig = plt.figure(figsize=(14, 8))

linewidth = 6
labelfontsize = 20
ticksize = 17

# TriPepSVM single runs
#plt.plot(fpr_CNN, tpr_CNN,label=r'CNN (AUC = %0.2f)' % (auc_score),lw=6, alpha=.8)
k = 1
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
for pred in pred_all:
    fpr, tpr, _ = roc_curve(y_score=pred.Score, y_true=pred.Label)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    auroc = roc_auc_score(y_score=pred.Score, y_true=pred.Label)
    aucs.append(auroc)
    plt.plot(fpr, tpr, lw=linewidth//2, alpha=0.1)
    k += 1

# RBPPred
rbppred_fpr, rbppred_tpr, t_rbppr = roc_curve(y_score=rbppred_test_all.Score, y_true=rbppred_test_all.Label)
rbppred_roc_auc = roc_auc_score(y_score=rbppred_test_all.Score, y_true=rbppred_test_all.Label)

# RNAPred
rnapred_fpr, rnapred_tpr, t_rnapr = roc_curve(y_score=rnapred_test_all.Score, y_true=rnapred_test_all.Label)
rnapred_roc_auc = roc_auc_score(y_score=rnapred_test_all.Score, y_true=rnapred_test_all.Label)

# SPOT-seq
spot_fpr, spot_tpr, t_spot = roc_curve(y_score=spot_test_all.Score, y_true=spot_test_all.Label)
#spot_roc_auc = roc_auc_score(y_score=spot_test_all.Score, y_true=spot_test_all.Label)


# plot mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr,
         label=r'Mean Balanced TriPepSVM (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=6, alpha=.8)

# plot std dev
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                 label=r'$\pm$ 1 std. dev. (TriPepSVM)')

# plot competitors
plt.plot(fpr_CNN, tpr_CNN,lw=linewidth,label='CNN (AUC = %0.2f)' % (auc_score))
plt.plot(rbppred_fpr, rbppred_tpr, lw=linewidth, label='RBPPred (AUC = {0:.2f})'.format(rbppred_roc_auc))
plt.plot(rnapred_fpr, rnapred_tpr, lw=linewidth, label='RNApred (AUC = {0:.2f})'.format(rnapred_roc_auc))
plt.plot(spot_fpr[-2], spot_tpr[-2], markersize=15, marker='o', lw=linewidth, label='SPOT-Seq RNA', color='purple') # second last point is the only non-inferred one
plt.plot([0, 1], [0, 1], color='gray', lw=linewidth, linestyle='--', label='Random')

plt.xlabel('False Positive Rate', fontsize=labelfontsize)
plt.ylabel('True Positive Rate', fontsize=labelfontsize)
plt.tick_params(axis='both', labelsize=ticksize)
#plt.title('Performance: ROC Curve ({})'.format(taxon), fontsize=labelfontsize)

species_name = 'Human'
if taxon == 561:
    species_name = 'E.Coli'
elif taxon == 590:
    species_name = 'Salmonella'

plt.title('{} ({}): ROC Curve on Testset'.format(species_name, taxon),
          fontsize=labelfontsize)
plt.legend(loc='lower right', prop={'size': 20})
fig.savefig('roc_curve_testset_{}.pdf'.format(taxon))
fig = plt.figure(figsize=(14, 8))



linewidth = 6
labelfontsize = 20
ticksize = 17

# TriPepSVM single runs
k = 1
y_true = []
y_pred = []
pr_values = []
rec_values = []
sample_thresholds = np.linspace(0, 1, 100)
no_pos = []
no_total = []
for pred in pred_all:
    pr, rec, thr = precision_recall_curve(probas_pred=pred.Score, y_true=pred.Label)
    no_pos.append(pred.Label.sum())
    no_total.append(pred.shape[0])
    pr_values.append(interp(sample_thresholds, thr, pr[:-1]))
    #pr_values[-1][-1] = 1.0
    rec_values.append(interp(sample_thresholds, thr, rec[:-1]))
    aupr = average_precision_score(y_score=pred.Score, y_true=pred.Label)
    plt.plot(rec, pr, lw=linewidth//2, alpha=0.1)
    y_true.append(pred.Label)
    y_pred.append(pred.Score)
    k += 1
    

# RBPPred
rbppred_pr, rbppred_rec, t_pr_rbppr = precision_recall_curve(probas_pred=rbppred_test_all.Score, y_true=rbppred_test_all.Label)
rbppred_pr_auc = average_precision_score(y_score=rbppred_test_all.Score, y_true=rbppred_test_all.Label)
zero_idx_rbp = (np.abs(t_pr_rbppr - 0.5)).argmin() # probabilities and cutoff is 0.5

# RNApred
rnapred_pr, rnapred_rec, t_pr_rnapr = precision_recall_curve(probas_pred=rnapred_test_all.Score, y_true=rnapred_test_all.Label)
rnapred_pr_auc = average_precision_score(y_score=rnapred_test_all.Score, y_true=rnapred_test_all.Label)
zero_idx_rna = (np.abs(t_pr_rnapr + 0.2)).argmin() # cutoff is -0.2

# SPOT-seq RNA
spot_pr, spot_rec, t_pr_spot = precision_recall_curve(probas_pred=spot_test_all.Score, y_true=spot_test_all.Label)
#spot_pr_auc = average_precision_score(y_score=spot_test_all.Score, y_true=spot_test_all.Label)

# plot mean PR curve for TriPepSVM
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
mean_precision, mean_recall, mean_thresholds = precision_recall_curve(y_true, y_pred)

plt.plot(r_CNN, p_CNN, color='red',
lw=linewidth, label='CNN (AUPR=%.2f)' %(auc(r_CNN,p_CNN)))

label = 'Mean Bal. TriPepSVM (AUPR=%.2f)' % (auc(mean_recall, mean_precision))
plt.plot(mean_recall, mean_precision, label=label, lw=linewidth)

# plot std dev
std_pr = np.std(pr_values, axis=0)
mean_pr = np.mean(pr_values, axis=0)
mean_rec = np.mean(rec_values, axis=0)
pr_upper = np.minimum(mean_pr + std_pr, 1)
pr_lower = np.maximum(mean_pr - std_pr, 0)
pr_upper = np.append(pr_upper, 1.)
pr_lower = np.append(pr_lower, 1.)
mean_rec = np.append(mean_rec, 0.)
plt.fill_between(mean_rec, pr_lower, pr_upper, color='grey', alpha=.3,
                 label=r'$\pm$ 1 std. dev.')

# plot competitors
plt.plot(rbppred_rec, rbppred_pr, lw=linewidth, label='RBPPred (AUC = {0:.2f})'.format(rbppred_pr_auc))#, color='darkgreen')
plt.plot(rnapred_rec, rnapred_pr, lw=linewidth, label='RNAPred (AUC = {0:.2f})'.format(rnapred_pr_auc))#, color='darkred')
plt.plot(spot_rec[1], spot_pr[1], markersize=15, lw=linewidth, marker='o', label='SPOT-seq RNA', color='purple')
random_y = test_res[test_res.Label == 1].shape[0] / test_res.shape[0]
plt.plot([0, 1], [random_y, random_y], color='gray', lw=3, linestyle='--', label='Random')


plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.tick_params(axis='both', labelsize=ticksize)
plt.title('{} ({}): Precision-Recall Curve on Testset'.format(species_name, taxon),
          fontsize=labelfontsize)
plt.legend(loc='upper right', prop={'size': 20})
plt.ylim([-0.05, 1.05])
fig.savefig('pr_curve_testset_{}.pdf'.format(taxon))
'''
'''
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
'''
'''
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('CNN.csv')
'''