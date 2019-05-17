逻辑回归的代价函数为
$$
J(W)=-\frac1m \left[ \sum_{i=1}^{m} y^{(i)}log(h_w(x^{(i)}))+(1-y^{(i)})log(1-h_w(x^{(i)}))\right] + \frac{\lambda}{2m} \sum_{j=1}^{n}W^2
$$
神经网络的代价函数为
$$
J(W)=-\frac1m \left[ \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)}log(h_w(x^{(i)}))_k + (1-y^{(i)}_k) log(1-h_w(x^{(i)}))_k \right] + \frac{\lambda}{2m} \sum_{l=1}^{L} \sum_{i=1}^{\delta_l} \sum_{j=1}^{\delta_{l+1}} (W_{ji}^{(l)})^2
$$
> 为何神经网络的代价函数看上去有一丢丢不一样呢，因为逻辑回归是二分类，结果$y​$是一位标量，而神经网络可以是多分类，结果$y_k​$是k位的向量

> 该损失函数$J(W)$可近似看为$\frac12 || h_{W,b}(x)-y||^2$



按一般的做法，下一步就该最小化$J(W)​$（训练时的权重更新）了。真实运行时，逻辑回归和神经网络都是用梯度下降，但两者的“梯度”不完全是同一个东西。神经网络用一种原理相似（也根据导数迭代）的算法：反向传播

> 神经网络为啥要用反向传播来计算最小化的$J(W)​$，因为逻辑回归可以理解是一层神经网络，与样本输出$y​$相关的权重$W​$只有一层。而神经网络的$W​$是多层的、层层传递的，所以最小化$J(W)​$时就得各层都涉及到



对于总体样本的训练：

1. 计算单个样本$(x,y)$的代价函数$J(W)$的偏导为$\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b;x,y)$和$\frac{\partial}{\partial b_{ij}^{(l)}} J(W,b;x,y)$

   > 详情见下面反向传播算法的几个步骤

2. 对于一个epoch内的所有样本的权重变化值

   $\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) = \left [ \frac1m \sum_{i=1}^{m} \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b;x^{(i)},y^{(i)}) \right ] + \lambda W_{ij}^{(l)}$

   $\frac{\partial}{\partial b_{i}^{(l)}} J(W,b) = \frac1m \sum_{i=1}^{m} \frac{\partial}{\partial b_{i}^{(l)}} J(W,b;x^{(i)},y^{(i)})$

3. 一个epoch内的权重更新：

   $W_{ij}^{(l)} = W_{ij}^{(l)} - \alpha \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b)$

   $b_{i}^{(l)} = b_{i}^{(l)} - \alpha \frac{\partial}{\partial b_{i}^{(l)}} J(W,b)$



反向传播算法可表示为以下几个步骤：

1. 进行前馈传导计算，利用前向传导公式，得到$L_1, L_2, …$直到输出层$L_{nt}$的激活值$a^{(l)}$。

2. 对输出层（第$nt$层），计算：

   $\delta^{(nt)} = -(y-a^{(nt)}) \cdot f^{'}(z^{(nt)})$

3. 对于$l=n_l-1, n_l-2, n_l-3, …, 2$的各层，计算：

   $\delta^{(l)} = ((W^{(l)})^T \delta^{(l+1)}) \cdot f^{'}(z^{(l)})$

4. 计算最终需要的偏导值：

   $\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b;x,y) = a_j^{(l)} \delta_i^{(l+1)}​$

   $\frac{\partial}{\partial b_{i}^{(l)}} J(W,b;x,y) = \delta_i^{(l+1)}$

上面的1、2、3步骤都是为4步骤做提前计算的，推导4步骤中$\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b;x,y)​$ 为何等于 $a_j^{(l)} \delta_i^{(l+1)}​$

> 推导
>
> 以$W$的偏导为例，问题拆解：$\frac{\partial}{\partial W_{ij}^{(l)}} J(W) = \frac {\partial J(W)} {\partial Z_i^{(l+1)}} * \frac{\partial Z_i^{(l+1)}}{\partial W_{ij}^{(l)}}$
>
> 1. 神经元输出定义：$Z_i^{(l+1)} = \sum_i^n W_{ij}^{(l)} * a_j^{(l)}$），则$\frac{\partial Z_i^{(l+1)}}{\partial W_{ij}^{(l)}} = \frac{\partial \sum_j^n W_{ij}^{(l)} * a_j^{(l)}}{\partial W_{ij}^{(l)}} = a_j^{(l)}$
> 2. 令误差$\delta_i^{(l+1)} = \frac{\partial J(W)}{\partial Z_i^{(l+1)}}$
>
> 则$\frac{\partial }{\partial W_{ij}^{(l)}} J(W) = \delta_i^{l+1} * a_j^{(l)} \tag{for code}$



推导$\delta_i^{(l+1)}$

> 通过链式求导展开$\delta_j^{(l)}$的定义
>
> $\delta_j^{(l)} = \frac {\partial J(W)} {\partial z_j^{(l)}} = \sum_{k=1}^{ml+1} \frac {\partial J(W)} {\partial z_k^{(l+1)}} \frac {\partial z_k^{(l+1)}} {\partial z_j^{(l)}} = \sum_{k=1}^{ml+1} \frac {\partial J(W)} {\partial z_k^{(l+1)}} \frac {\partial z_k^{(l+1)}} {\partial a_j^{(l)}} \frac {\partial a_j^{(l)}} {\partial z_j^{(l)}} $
>
> 根据$\delta$的定义可知$\frac {\partial J(W)} {\partial z_k^{(l+1)}} = \delta_k^{(l+1)}$，根据$z_k^{(l+1)}$定义可知$\frac {\partial z_k^{(l+1)}} {\partial a_j^{(l)}} = w_{kj}^{(l+1)}$，根据$a_j^l$定义可知$\frac {\partial a_j^{(l)}} {\partial z_j^{(l)}} = h^{'}(z_j^{{(l)}})$，代入上式，则：
>
> $\delta_j^{(l)} = \sum_{k=1}^{ml+1} \delta_k^{(l+1)} w_{kj}^{(l+1)} h^{'}(z_j^{(l)}) = h^{'}(z_j^{(l)}) \sum_{k=1}^{ml+1} \delta_k^{(l+1)} w_{kj}^{(l+1)}$

> > 意思就是，实际上前层误差（靠近输入方向的一层）就等于后一层误差（靠近输出方向的一层）乘上激活函数导数和层间权重，某层的计算取决于后一层的结果，就是反向传播的重点



# 作业

> 至于梯度消失和梯度爆炸，则来源于$\delta_j^{(l)}$计算公式中的$h^{'}(z_j^{(l)})$，隐藏层激活函数的导数。
>
> 1. 梯度消失：如果隐藏层激活函数是$Sigmod$（举个例子），那它的导数为$h^{'}(x) = \left ( \frac{1}{1+e^{-x}} \right) ^ {'} =……= h(x)(1-h(x))$ 。而$Sigmod$函数$h(x)=\frac{1}{1+e^{-x}}$从图像看，取值范围就是0到1之间。那导数计算中的$h(x)$和$1-h(x)$都是0到1之间的小数，俩小数相乘自然更小。$\delta_j^{(l)}$=层间误差=需要改的部分≈$J(W)$的偏导，随着神经网络的层数的增加，每往后“反向传播”一层，当前层都会在前一层的误差结果上再乘一个0到1之间的小数（见$\delta_j^{(l)}$的计算公式），“当前层的误差”就会越来越小，乃至最后几乎不变了
>
> 2. 梯度爆炸：原理跟梯度消失类似，也是因为激活函数的导数。如果激活函数的导数都是大于1的，那经过N层的计算后，误差就会被乘的越来越大，所谓爆炸



> 梯度消失和梯度爆炸的解决
>
> 1. 选取合适的激活函数，比如ReLU
> 2. 权重正则化，for 梯度爆炸 
> 3. batchnorm，反向传播中，对输出的误差结果做规范，使每一层的输出规范为均值和方差一致
> 4. 残差结构，使之前几层几十层、容易梯度消失过快的网络变成轻松几百层上千层都不用太担心梯度问题
> 5. LSTM，长短期记忆网络

