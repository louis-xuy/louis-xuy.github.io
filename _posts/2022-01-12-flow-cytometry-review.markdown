---
layout:     post
title:      "flowPeaks 算法过程"
subtitle:   ""
date:       2021-12-31 00:00:00
author:     "louis"
header-img: "img/post-bg-rwd.jpg"
header-mask: 0.3
catalog:    true
mathjax:    true
tags:
    - cluster
---

### 在数据的每个维度中应用Freedman Diaconis公式，以获得K-均值的聚类数K

$$ K_{j}=\left(x_{(n)}^{j}-x_{(1)}^{j}\right) /\left\{2 \cdot \operatorname{IQR}\left(x^{j}\right) \cdot n^{-1 / 3}\right\} \text { for } j=1, \ldots, d $$

其中$x_{(n)}^{j}$, $x_{(1)}^{j}$ 表示第j个维度的最大值和最小值。$x^j=\left(x^i_{1},x^j_{2} \text , \ldots, x^j_{n}\right)$.   
IQR（·）是数据的四分位范围，定义为第75百分位和第25百分位之间的差异 

$$ K=\left\lceil\operatorname{median}\left(K_{1}, \ldots, K_{d}\right)\right\rceil $$

### 使用k-means++算法初始化最初的k个聚类种子, 

### 在Lloyd’s的k-means算法中应用k-d树数据表示，直到算法收敛  

Lloyd’s的k-means算法，即经典kmeans算法  
kd-tree（k-dimensional树的简称），是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构

### 进一步应用Hartingan-Wong k-Means算法来提高聚类的紧致性
    
Hartingan-Wong k-Means算法与经典kmeans略有不同。 
在第3步的kmeans算法收敛之后，运用Hartingan-Wong k-Means算法重新计算样本的聚类中心点和每个样本所属簇。以减少目标函数$\sum^n_{i=1}{||x_i-c_{L_i}||^2}$,其中$L_i$=1,...K, 是$x_i$的聚类类别。$c_k$是聚类中心  
**我们也可以直接应用该算法去获取聚类初始中种子, 但是该算法运算太慢了** 

### 基于k-means的结果计算$w_k, \mu_k, \sum_k \text { for } k=1, \ldots, K$  

其中  
- $w_k$ 样本比例，满足 $\sum^{K}_{k=1}w_k=1$
- $\mu_k$ 样本均值， 
- $\sum_k$ 样本方差，$\sum_k$ 噪音可能太多， 该文章采用了一种平滑的表示方法:  

$$
\tilde{\Sigma}_{k}=\lambda_{k} \cdot h \Sigma_{k}+\left(1-\lambda_{k}\right) \cdot h_{0} \Sigma_{0}
$$

其中h和h0是定制的参数，调整后可使密度函数更平滑或粗糙  
$ \lambda_{k}=nw_{k}/\left(k+nw_{k}\right) $,   
$ \Sigma_{0} $ 是一个方差矩阵，假设数据在整个数据范围内均匀分布，并且是一个对角线矩阵，对于元素(j,j)有: $ \Sigma^{jj}_{0}=\{(x^j_{(n)}-x^{j}_{(1)})/k^{1/d}\}^2  \text { for }j=1, \ldots, d $ 

### 基于高斯有限混合模型生成的密度函数，计算从中心$\mu_k，k=1,\ldots,k$开始的局部峰值。
$$
f(x)=\sum_{k=1}^{K} w_{k} \cdot \phi\left(x ; \mu_{k}, \Sigma_{k}\right)
$$

### 应用附录中的算法A2分层合并峰值。

### 最终K-means算法的K个簇根据合并的峰值重新分组