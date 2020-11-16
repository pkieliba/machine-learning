#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import fbeta_score, log_loss
from sklearn.metrics.ranking import _binary_clf_curve

def eda(df):
    """Prints a dataframe including basic statistics of all the columns of the input dataframe"""
    
    allVal = pd.DataFrame({'type': df.dtypes, 'count': df.count(), 
                           'missing': df.isnull().sum(), 'unique': df.nunique()})
    numeric = pd.DataFrame({'mean': df.mean(), 'std': df.std(), 'min': df.min(), 
                            'max': df.max(), 'kurtosis': df.kurtosis(), 'skew': df.skew()})
    return allVal.join(numeric, how="left") 
    
def plotCM(cm, normalize=False, cmap=plt.cm.YlOrRd, ax=None, xlabel=True, ylabel=True):
    """Plots a confusion matrix"""
    
    if ax is None:
        ax = plt.gca()
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        
    plt.style.use('seaborn-whitegrid')
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()*1.4/2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        ax.text(j, i, format(cm[i,j], fmt),
                horizontalalignment = "center",
                color = "white" if cm[i,j] > thresh else "black",
                size = 20
               )
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_ylabel('True class', fontsize=20)
    ax.set_xlabel('Predicted class', fontsize=20)
    if not ylabel:
        ax.set_ylabel('')
    if not xlabel:
        ax.set_xlabel('')
    ax.grid(None)
    
def plotROC(Ytest, Yprobs, threshold_selected=None, ax=None, xlabel=True, ylabel=True):
    """Plots ROC"""
    
    if ax is None:
        ax = plt.gca()
        
    nsProbs = np.zeros(len(Ytest), dtype='int8')
    fpr, tpr, thresh = roc_curve(Ytest, Yprobs)
    ns_fpr, ns_tpr, _ = roc_curve(Ytest, nsProbs)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    bestThr = thresh[ix]

    plt.style.use('seaborn-whitegrid')
    ax.plot(ns_fpr, ns_tpr, linestyle='--', label="Random", color='grey')
    step_kwargs = ({'step': 'post'})
    ax.step(fpr, tpr, alpha=0.4, color='skyblue', where='post')
    ax.fill_between(fpr, tpr, alpha=0.4, color='skyblue', **step_kwargs)
    ax.scatter(fpr[ix], tpr[ix], marker='o', color='red', label='Best')
    ax.set_xlabel('FP rate', fontsize=18)
    ax.set_ylabel('TP rate', fontsize=18)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, 1])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.legend(fontsize=14)
    ax.set_aspect('equal')
    if not ylabel:
        ax.set_ylabel('')
    if not xlabel:
        ax.set_xlabel('') 
    roc = pd.DataFrame({'fpr': fpr[:-1], 'tpr': tpr[:-1],
                         'threshold': thresh[:-1]}).transpose()
                         
    return roc, bestThr

def plotPR(Ytest, Yprobs, ax=None, xlabel=True, ylabel=True):
    """Plots precision-recall curve"""
    
    if ax is None:
        ax = plt.gca()
        
    precision, recall, thresh = precision_recall_curve(Ytest, Yprobs)
    ns = len(Ytest[Ytest==1])/len(Ytest)
    plt.style.use('seaborn-whitegrid')
    ax.plot([0, 1], [ns, ns], linestyle='--', label='Random', color="gray")
    step_kwargs = ({'step': 'post'})
    ax.fill_between(recall, precision, alpha=0.4, color='skyblue', **step_kwargs)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.nanargmax(fscore)
    bestThr = thresh[ix]
    ax.scatter(recall[ix], precision[ix], marker='o', color='red', label='Best')
    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.legend(fontsize=14)
    ax.set_aspect('equal')
    if not ylabel:
        ax.set_ylabel('')
    if not xlabel:
        ax.set_xlabel('')
    pr = pd.DataFrame({'precision': precision[:-1], 'recall': recall[:-1],
                        'threshold': thresh}).transpose()
    
    return pr, bestThr
    
def plotIncrementalProfit(Ytest, Yprobs, ax=None, xlabel=True, ylabel=True):
    """Plots incremental profit obtained by the model as a function of probability threshold"""
    
    if ax is None:
        ax = plt.gca()
        
    fps , tps , thresh = _binary_clf_curve(Ytest,Yprobs)
    thresh = thresh
    profit = (9.85*tps - 0.15*fps) / (sum(Ytest)*9.85)
    plt.style.use('seaborn-whitegrid')
    step_kwargs = ({'step': 'post'})
    ax.fill_between(thresh, profit, alpha=0.4, color='skyblue', **step_kwargs)
    ix = np.nanargmax(profit)
    bestThr = thresh[ix]
    ax.scatter(thresh[ix], profit[ix], marker='o', color='red', label='Best')
    ax.set_xlabel('Threshold', fontsize=18)
    ax.set_ylabel('Incremental profit (percentage of max)', fontsize=18)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.legend(fontsize=14)
    if not ylabel:
        ax.set_ylabel('')
    if not xlabel:
        ax.set_xlabel('')
    ax.set_aspect('equal')
    profit = pd.DataFrame({'profit': profit[:-1],
                         'threshold': thresh[:-1]}).transpose()
                         
    return  profit, bestThr
    
def metricsTraditional(Ytest, Yprobs, Ypred=None, thresholds=[0.5], columns=['orig']):
    """Returns a dataframe containing evaluation metrics of a model"""
    
    if len(columns) < len(thresholds):
        columns = thresholds
        
    auc = []
    prec = []
    recall = []
    f1 = []
    f2 = []
    NIR = []
    averageProfitTarget = []
    
    if Ypred is not None:
        auc.append(roc_auc_score(Ytest, Ypred))
        prec.append(precision_score(Ytest, Ypred))
        recall.append(recall_score(Ytest, Ypred))
        f1.append(f1_score(Ytest, Ypred))
        f2.append(fbeta_score(Ytest, Ypred, beta=2))
        NIR.append(incremental_profit_score(Ytest, Ypred))
        averageProfitTarget.append(incremental_profit_score(Ytest, Ypred)/np.sum(Ypred))
        
    for thresh in thresholds:
        auc.append(roc_auc_score(Ytest, (Yprobs>thresh).astype(int)))
        prec.append(precision_score(Ytest, (Yprobs>thresh).astype(int)))
        recall.append(recall_score(Ytest, (Yprobs>thresh).astype(int)))
        f1.append(f1_score(Ytest, (Yprobs>thresh).astype(int)))
        f2.append(fbeta_score(Ytest, (Yprobs>thresh).astype(int), beta=2))
        NIR.append(incremental_profit_score(Ytest, (Yprobs>thresh).astype(int)))
        averageProfitTarget.append(
            incremental_profit_score(
                Ytest, (Yprobs>thresh).astype(int)
            ) / np.sum((Yprobs>thresh).astype(int))
        )
        
    metrics = pd.DataFrame({'AUC': auc, 'Precission': prec, 'Recall': recall, 'F1': f1, 'F2': f2, 
                            'NIR': NIR, 
                            'Average NIR per Targeted Customer': averageProfitTarget}).transpose().round(decimals=2)
    metrics.columns = columns
    
    return metrics
    
    
def metricsUplift(Ytest, Yprobs, Ypred=None, thresholds=[0.5], columns=['orig']):
    """Returns a dataframe containing evaluation metrics of an uplift model"""

    if len(columns) < len(thresholds):
        columns = thresholds
        
    auc = []
    prec = []
    recall = []
    f1 = []
    f2 = []
    logloss = []

    if Ypred is not None:
        auc.append(roc_auc_score(Ytest, Ypred))
        prec.append(precision_score(Ytest, Ypred))
        recall.append(recall_score(Ytest, Ypred))
        f1.append(f1_score(Ytest, Ypred))
        f2.append(fbeta_score(Ytest, Ypred, beta=2))
        logloss.append(log_loss(Ytest, Ypred))
        
    for thresh in thresholds:
        auc.append(roc_auc_score(Ytest, (Yprobs>thresh).astype(int)))
        prec.append(precision_score(Ytest, (Yprobs>thresh).astype(int)))
        recall.append(recall_score(Ytest, (Yprobs>thresh).astype(int)))
        f1.append(f1_score(Ytest, (Yprobs>thresh).astype(int)))
        f2.append(fbeta_score(Ytest, (Yprobs>thresh).astype(int), beta=2))
        logloss.append(log_loss(Ytest, Yprobs))
        
    metrics = pd.DataFrame({'AUC': auc, 'Precission': prec, 'Recall': recall, 'F1': f1, 'F2': f2,
                            'Log Loss': logloss}).transpose().round(decimals=2)
    metrics.columns = columns

    return metrics
    
def incremental_profit_score(Ytest, Ypred):
    """Calculates incremental profit generated by the model"""
    
    cm = confusion_matrix(Ytest, Ypred)
    tps = cm[1,1]
    fps = cm[0,1]
    profit = 9.85*tps - 0.15*fps
    
    return profit.astype(int)
    
def IIRscore(Ytest,Ypred):
    """Calculates IIR generated by the model"""
    
    cm = confusion_matrix(Ytest, Ypred)
    tps = cm[1,1]
    fps = cm[0,1]
    tng = cm[0,0]
    fng = cm[1,0]
    IIR = tps/(tps+fps) - fng/(tng+fng)
    
    return IIR


