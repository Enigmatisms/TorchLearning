# Torch Learning
### Pytorch 上手训练
#### Deeds
- 2020.9.3
  - 搞清楚了基本的自动求导，梯度计算以及数据结构转换
  - 使用Torch的自动求导对退火算法最终结果进行优化 (优化结果对于不同函数效果的好坏不一)
  - 使用Torch写了一个多项式函数拟合方法，与np.polyfit比较，十分接近polyfit的结果（测试了1, 2次函数）
#### TODOs
- [ ] 更多优化算法使用Torch的自动求导实现（个人感觉这样用就相当于一个python版的Ceres）