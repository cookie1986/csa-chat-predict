import os

import pandas as pd
import matplotlib.pyplot as plt

#### load data
metricsDF = pd.read_csv('C:/Users/Darren Cook/Documents/PhD Research/csa_chats/predicting_predators/models/model1_metrics.csv')
metricsDF = metricsDF.iloc[: , 1:]



#### Comparing performance between algorithms

## Visualization 

# compare performance of algorithms
metricsDF_mean = metricsDF.groupby('model').mean()

# boxplot
metricsDF.boxplot( 
                  by='model',
                  layout=(3,5),
                  figsize=(8,8),
                  fontsize=10)
plt.suptitle('')
# plt.savefig("plots/algorithm_comparisonbox.png", bbox_inches='tight')
plt.show()


## ANOVA
import scipy.stats as stats

# recall
recall = metricsDF.drop(['Precision','F1'], axis=1)
recall_unstacked = recall.pivot(columns='model')['Recall']
recall_unstacked = recall_unstacked.apply(lambda x: pd.Series(x.dropna().values))

# run test
fvalue, pvalue = stats.f_oneway(recall_unstacked['lr'],
                                recall_unstacked['nb'], 
                                recall_unstacked['rf'], 
                                recall_unstacked['svm'])
print(fvalue, pvalue)


# run post-hocs (Tukey HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

recall_posthoc = pairwise_tukeyhsd(endog=recall['Recall'],
                          groups=recall['model'],
                          alpha=0.01)

print(recall_posthoc)



# F1
f1 = metricsDF.drop(['Precision','Recall'], axis=1)
f1_unstacked = f1.pivot(columns='model')['F1']
f1_unstacked = f1_unstacked.apply(lambda x: pd.Series(x.dropna().values))

# run test
fvalue, pvalue = stats.f_oneway(f1_unstacked['lr'],
                                f1_unstacked['nb'], 
                                f1_unstacked['rf'], 
                                f1_unstacked['svm'])
print(fvalue, pvalue)


# run post-hocs (Tukey HSD)
f1_posthoc = pairwise_tukeyhsd(endog=f1['F1'],
                          groups=f1['model'],
                          alpha=0.01)

print(f1_posthoc)