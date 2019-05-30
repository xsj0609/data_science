<font size=5 color='blue'>一、表达式</font>

从线性回归开始说起

线性回归的表达式：

![](https://latex.codecogs.com/gif.latex?f(x)=w_0+w_1x_1+w_2x_2+\cdots+w_nx_n)

![](https://latex.codecogs.com/gif.latex?f(x))的值域为![](https://latex.codecogs.com/gif.latex?(-\infty, +\infty))，![](https://latex.codecogs.com/gif.latex?x)为输入变量，![](https://latex.codecogs.com/gif.latex?w)为系数

线性回归的训练过程，就是将![](https://latex.codecogs.com/gif.latex?w)以<font color='red'>某种规律</font>变动，每变动一次![](https://latex.codecogs.com/gif.latex?w)，就与固定的输入![](https://latex.codecogs.com/gif.latex?x)计算得到训练中的![](https://latex.codecogs.com/gif.latex?f(x))输出（记为![](https://latex.codecogs.com/gif.latex?y^{(i)})）。

<font size=5 color='blue'>二、损失函数</font>

训练是要达到好的效果的，怎么评估训练效果呢，最简单直接的方法就是算原始标签![](https://latex.codecogs.com/gif.latex?y)与训练得到的![](https://latex.codecogs.com/gif.latex?y^{(i)})相减求<font color='red'>训练误差</font>。训练时所有样本的总训练误差的函数表示即误差函数，用公式表达可以写成：

![](https://latex.codecogs.com/gif.latex?L(w)=\sum_{i=1}^m (y^{(i)}-y]^2)

模型好就是训练误差小，即损失函数最小。因此训练目的就是取得L(w)最小时的系数w。（有了系数w，在新样本来时，就可以输入x变量，与系数计算得到预测y值了）

那L(w)什么时候最小呢，理论上是未知数导数=0的时候。

<font color='green' size='2'>（这里是训练，因此x是已知数，而w是未知数，是在对w求导并且导数=0时，L(w)最小。训练求的就是在x和y已知时的最好的w和b）</font>

> 这里![](https://latex.codecogs.com/gif.latex?L(w))是凸函数，![](https://latex.codecogs.com/gif.latex?y)和![](https://latex.codecogs.com/gif.latex?x)都是已知数，只有![](https://latex.codecogs.com/gif.latex?w)是未知数，中学就学过![](https://latex.codecogs.com/gif.latex?f(x)=(ax-b)^2)，其图形示意如图：

![image-20190530112440055](http://ww1.sinaimg.cn/large/006tNc79gy1g3jgyjutjvj303u04bweh.jpg)

> 理论上有全局最优解，![](https://latex.codecogs.com/gif.latex?f(x)=(ax-b)^2)求最小值咋求的，求导，令导数=0就能求得最小值对应的x。
>
> <font color='green' size='2'>（这里a和b是已知数，x是未知数，即是在对x求导并且导数=0时，f(x)最小）</font>

<font size=5 color='blue'>三、梯度下降</font>

既然未知数的导数=0时L(w)最小，那就求出L(w)对w的导数，并在每一轮迭代中用相同的x输入，让w迭代导数大小后参与下一轮训练，直到迭代到满意的w（让L(w)最小）就行了。

每次迭代导数大小，往导数=0的方向去，<font color='blue'>该方法即梯度下降</font>。梯度即导数，下降即往最小的方向去计算。

设第j个变量的系数为![](https://latex.codecogs.com/gif.latex?w_j)，![](https://latex.codecogs.com/gif.latex?\lambda)为按导数变化多少倍数，t为迭代第几轮，m为总样本数，![](https://latex.codecogs.com/gif.latex?\sigma)为导数。则w的梯度下降表达式为：

![](https://latex.codecogs.com/gif.latex?w_j^{(t+1)}=w_j^{(t)}-\lambda \sum_{i=1}^m \sigma(w))

> 虽然理论上令导数=0就可以直接求得最好的系数，实际应用的时候，由于变量数多、样本量大，自然不可能求解联立方程组一样去求解

