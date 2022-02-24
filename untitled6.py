# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:15:41 2020

@author: Administrator
"""
'''
import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
waters = ('Sensitivity', 'Specificity', 'BACC', 'MCC')

RNAPred_value = [0.88, 0.44, 0.66, 0.22]
SPOT_value = [0.23, 0.95, 0.59, 0.22]
RBPPred_value = [0.76, 0.64, 0.70, 0.27]
TriPepSVM_value = [0.52,0.94,0.73,0.49]
iDRBP_MMC_value = [0.78,0.91,0.85,0.61]
CnnRBP_value = [0.72, 0.95, 0.83, 0.66]


RNAPred_value = [0.80, 0.64, 0.72, 0.34]
SPOT_value = [0.29, 1.00, 0.65, 0.50]
RBPPred_value = [0.68, 0.84, 0.76, 0.45]
TriPepSVM_value = [0.55,0.98,0.76,0.63]
iDRBP_MMC_value = [0.57,0.97,0.77,0.62]
CnnRBP_value = [0.75, 0.96, 0.85, 0.68]



RNAPred_value = [0.74, 0.53, 0.64, 0.18]
SPOT_value = [0.35, 1.00, 0.67, 0.56]
RBPPred_value = [0.63, 0.84, 0.74, 0.37]
TriPepSVM_value = [0.65,0.96,0.81,0.63]
iDRBP_MMC_value = [0.61,0.94,0.77,0.57]
CnnRBP_value = [0.68, 0.98, 0.83, 0.73]

bar_width = 0.1  # 条形宽度
index_RNAPred = np.arange(len(waters))  
index_SPOT = index_RNAPred + bar_width 
index_RBPPred = index_SPOT + bar_width
index_TriPepSVM = index_RBPPred + bar_width
index_iDRBP_MMC = index_TriPepSVM + bar_width
index_CnnRBP = index_iDRBP_MMC + bar_width



# 使用两次 bar 函数画出两组条形图
plt.bar(index_RNAPred, height=RNAPred_value, width=bar_width, color='deepskyblue', label='RNAPred')
plt.bar(index_SPOT, height=SPOT_value, width=bar_width, color='steelblue', label='SPOT-Seq-RNA')
plt.bar(index_RBPPred, height=RBPPred_value, width=bar_width, color='darkseagreen', label='RBPPred')
plt.bar(index_TriPepSVM, height=TriPepSVM_value, width=bar_width, color='darksalmon', label='TriPepSVM')
plt.bar(index_iDRBP_MMC, height=iDRBP_MMC_value, width=bar_width, color='palevioletred', label='iDRBP_MMC')
plt.bar(index_CnnRBP, height=CnnRBP_value, width=bar_width, color='indianred', label='CnnRBP')

plt.legend(loc='upper right', prop={'size': 7.7})  # 显示图例
plt.ylim([0.00, 1.19])
plt.xticks(index_RNAPred + bar_width*3, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('Value')  # 纵坐标轴标题
plt.title('Human Testdata')  # 图形标题
plt.savefig("Human.eps",format='eps',dpi=1000)
#plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#指定默认字体
'''
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
'''

RNAPred_value = [0.50,0.44,0.88,0.19,0.65,0.31,0.22]
RBPPred_value = [0.66,0.64,0.76,0.24,0.70,0.37,0.27]
SPOT_value = [0.86,0.95,0.23,0.39,0.59,0.29,0.22]
TriPepSVM_value = [0.89,0.94,0.52,0.59,0.73,0.55,0.49]
iDRBP_MMC_value = [0.89,0.91,0.78,0.57,0.85,0.66,0.61]
CnnRBP_value = [0.92,0.95,0.72,0.69,0.85,0.71,0.66]

'''
RNAPred_value = [0.67,0.64,0.80,0.32,0.72,0.46,0.34]
RBPPred_value = [0.81,0.84,0.68,0.48,0.76,0.56,0.45]
SPOT_value = [0.87,1.00,0.29,1.00,0.65,0.45,0.50]
TriPepSVM_value = [0.90,0.98,0.55,0.85,0.76,0.67,0.63]
iDRBP_MMC_value = [0.93,0.97,0.57,0.75,0.77,0.65,0.62]
CnnRBP_value = [0.93,0.96,0.75,0.70,0.85,0.72,0.68]
'''

'''
RNAPred_value = [0.49,0.44,0.86,0.17,0.65,0.29,0.20]
RBPPred_value = [0.81,0.84,0.63,0.35,0.74,0.45,0.37]
SPOT_value = [0.92,1.00,0.34,1.00,0.67,0.51,0.56]
TriPepSVM_value = [0.92,0.96,0.65,0.69,0.81,0.67,0.63]
iDRBP_MMC_value = [0.88,0.94,0.61,0.68,0.77,0.64,0.58]
CnnRBP_value = [0.93,0.98,0.68,0.88,0.83,0.76,0.73]
'''
 
ind_x = np.arange(1,42,6) # the x locations for the groups
width = 0.8  # the width of the bars
 
fig,ax = plt.subplots()
rects1 = ax.bar(ind_x + width, RNAPred_value, width, color='SkyBlue',align='edge', label='RNAPred')
rects2 = ax.bar(ind_x + 2*width, RBPPred_value, width,color='IndianRed',align='edge', label='RBPPred')
rects3 = ax.bar(ind_x + 3*width, SPOT_value, width, color='Cyan',align='edge', label='SPOT-Seq-RNA')
rects4 = ax.bar(ind_x + 4*width, TriPepSVM_value, width, color='Magenta',align='edge', label='TriPepSVM')
rects5 = ax.bar(ind_x + 5*width, iDRBP_MMC_value, width, color='darkseagreen',align='edge', label='iDRBP_MMC')
rects6 = ax.bar(ind_x + 6*width, CnnRBP_value, width, color='red',align='edge', label='CnnRBP')

 
# Add some text for labels, title and custom x-axis tick labels, etc.


#ax.set_title('Scores by group and gender')
plt.ylim([0.00, 1.19])
#plt.xticks((width*4),ind)
plt.xticks(ind_x + width*6,('Acc','Sp','Se', 'Pre', 'BACC','F1','MCC' ))
plt.ylabel('Value')  # 纵坐标轴标题
plt.title('Human Testdata')  # 图形标题
ax.legend(loc='upper right',prop={'size': 7})
plt.savefig("Human_data.pdf")
plt.show()

'''
x = [1000,1100,1200,1300,1400,1500,1600,1700] 

y3 = [90.63,92.39,95.99,94.86,96.17,96.28,91.39,90.53] 
y2 = [94.75,95.62,97.96,98.27,98.97,99.68,98.36,98.04]
y1 = [97.91,98.16,98.19,98.80,98.62,99.58,98.96,98.62]  
plt.figure(figsize=(8,6)) #创建绘图对象
plt.grid(True) 
plt.ylim(75,105)
plt.plot(x,y1,marker='*',color='r',linewidth=2,label="Human datasets")
plt.plot(x,y2,marker='*',color='g',linewidth=2,label="E.coli datasets")
plt.plot(x,y3,marker='*',color='b',linewidth=2,label="Salmonella datasets") 
plt.xlabel("Feature dimension") #X轴标签 
plt.ylabel("ROC_AUC(%)") #Y轴标签 
#plt.title("Line plot") #图标题 
#plt.title('Salmonella datasets')
plt.legend(loc='upper right',prop={'size': 15})
plt.legend(loc="lower right")
plt.savefig("feature dim.pdf") #保存图
plt.show()
'''
