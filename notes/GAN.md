## GAN 生成对抗网络

---

[toc]

---

#### 编码器原理

##### 自动编码器

​		自动编码器，简单地说就是以下结构：

``` mermaid
graph LR
A(Original)-->B(Encoder)
B-->C(Code-Compressed)
C-->D(Decoder)
D-->E(Generated)
F(Perturb & Intervene)-->C
```

​		从原始输入，对其进行编码（编码过程可以使用感知机（前馈，无BP操作）或者神经网络（现在的架构一般是BP式的优化）），生成 **<u>压缩后的编码数据</u>**。再想办法从编码中（必然存在信息损失）恢复原来的图片。但是一般而言，自动编码器（AE）都是确定输入输出的，每一种编码会由训练得到唯一的输出。

​		我们希望生成更多的数据或者达到人工智能创作 / 想象的目的，需要随机的输出。我们希望可以加入编码的扰动（Perturbance）或者人工的干预（Intervention），让编码更加多样化 / 随机化，而且Decoder可以对这种加入的噪声鲁棒。这就得到了VAE（Variational Auto Encoder）变分自动编码器。

##### 变分自动编码器

​		一个想法：首先一个随机的编码器在数学上的表现应该是给定一个向量（比如编码）Z，从Z中恢复X（训练集的分布）。那么就是从$P_Z(x)$到$P_X(x)$的映射（编码分布空间到样本空间的映射），得到新的分布$P_{\hat X}(x)$。那么根据贝叶斯的想法，可以表示分布$P_X(x)$为：
$$
P_X(x)=\sum_kP(X|Z_k)P(Z_k)
$$
​		意思是：不同编码的分布 与 给定编码下，输出数据为对应类型X 两者的条件概率结合。那么数据生成和分布之间的关系又是什么呢?直观地理解一下：

​		世界上有几乎无数种猫，猫产生的后代也不是和其父母一致的。那么我们如果想从数据层面【生成猫】，应该首先知道 **不同表现型的猫的【总体分布】**，如果知道这个分布，显然只需要从这个分布中随机采样就可以得到任意新性状的猫。但是这个分布基本是不可知的，即使存在海量数据我们也没办法得到这个分布的表达式。所以我们希望通过别的方式来近似表达这个分布，比如使用式(1)。

​		假设Z不是编码而是不同的形状，那么可以建立一个性状以及含有对应组合表现型的猫的概率映射，通过贝叶斯全概率公式获得。而把Z换为编码（**更加抽象的形状表示**），也是一样的，编码存在分布（正如不同性状如黑白存在一定分布），而给定编码（给定形状）时也存在其他的分布（Z=公猫，公猫中的白猫 / 花猫分布等等）。

​		VAE中，$P(Z)$被建模成了标准正态分布（其他分布也可），原因有以下两点：

- 标准正态分布常见，并且方便进行熵计算（**<u>或者说，KL散度计算时不会出现问题（比如均匀分布会存在概率密度为0导致奇异性的现象）</u>**）
- 天然的exp性质，并且表示容易，只需要对$\mu,\sigma$进行建模表示即可。

#### GAN原理

#### GAN实现过程中的一些问题