---
layout:     post
title:      "Evaluating Machine Learning Models"
subtitle:   ""
date:       2021-01-29 00:00:00
author:     "louis"
header-img: "img/year-book-plan.jpg"
header-mask: 0.3
catalog:    true
tags:
    - machine learning
---

### 常见机器学习评估指标

> 分类评估指标详解

* 准确率 – Accuracy

    预测正确的结果占总样本的百分比
    
    准确率 =(TP+TN)/(TP+TN+FP+FN)

    > 虽然准确率可以判断总的正确率，但是在样本不平衡 的情况下，并不能作为很好的指标来衡量结果

* 精确率（差准率）- Precision

    所有被预测为正的样本中实际为正的样本的概率
    
    精准率 =TP/(TP+FP)
    
    > 精准率和准确率看上去有些类似，但是完全不同的两个概念。精准率代表对正样本结果中的预测准确程度，而准确率则代表整体的预测准确程度，既包括正样本，也包括负样本。
 
* 召回率（查全率）- Recall

    实际为正的样本中被预测为正样本的概率
    
    召回率=TP/(TP+FN)

    > 召回率越高，代表实际坏用户被预测出来的概率越高

* F1分数
    ![](https://github.com/Jaxon-xy/Jaxon-xy.github.io/raw/master/portfolio/images/post-precision-Recall.png)
    
    F1=(2×Precision×Recall)/（Precision+Recall）

* ROC 曲线
    
    ROC 曲线中的主要两个指标就是真正率和假正率，上面也解释了这么选择的好处所在。其中横坐标为假正率（FPR），纵坐标为真正率（TPR），下面就是一个标准的ROC曲线图
    
    ![](https://github.com/Jaxon-xy/Jaxon-xy.github.io/raw/master/portfolio/images/post-roc.png)
    

    
    