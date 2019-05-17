sigmod表达式
$$
\begin{align}
g\left( z\right) & =\dfrac {1}{1+e^{-z}}
\end{align}
$$
求导：
$$
\begin{align}
g\left( z\right)^{'} & =\left( \dfrac {1}{1+e^{-z}}\right)^{'} \\
& = \dfrac {1^{'}\left( 1+e^{-z}\right) -1\left( 1+e^{-z}\right)^{'} }{\left( 1+e^{-z}\right) ^{2}} \\
& = \dfrac {-1\left( -e^{-z}\right) }{\left( 1+e^{-z}\right) ^{2}} \\
& = \dfrac {e^{-z} }{\left( 1+e^{-z}\right) ^{2}} \\
& = \dfrac {1}{1+e^{-z}}\dfrac {e^{-z}}{1+e^{-z}} \\
& = \dfrac {1}{1+e^{-z}}\dfrac {1+e^{-z}-1}{1+e^{-z}} \\
& = \dfrac {1}{1+e^{-z}}\left( 1-\dfrac {1}{1+e^{-z}}\right) \\
& = g(z)(1-g(z))
\end{align}
$$
作用：将输出$z​$激活映射到$(0,1)​$之间

缺点：

1. 当$z$非常大或非常小时，sigmod的导数$g(z)^{'}$将接近0，会导致向下层传播或反向传播更新权重$W$时，使$W$需要修改的值（$W$的梯度）非常小（接近0），使梯度更新非常缓慢，即梯度消失
2. 函数的输出不是以0为均值，均值是0.5，不便于下层的计算

使用：

1. 二分类算法（如逻辑回归）的最后（比如让逻辑回归输出值变成$(0, 1)$的值）
2. 神经网络的最后一层，不用在隐藏层中，作为输出层做二分类作用



tanh表达式
$$
g(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
求导
$$
\begin{align}
g(z)^{'} & = \left(\frac{e^z-e^{-z}}{e^z+e^{-z}} \right)^{'} \\
& = …… \\
& = \frac{4}{(e^z+e^{-z})^2} \\
& = …… \\
& = 1-g(z)^2
\end{align}
$$
作用：将输出$z$激活映射到$(-1,1)$之间

缺点：

1. 同sigmod，易梯度消失

优点：

1. 均值为0，比sigmod好

使用：

1. 神经网络最后一层，偶尔也能看到用在隐藏层



ReLU表达式
$$
\begin{align}
g(z) & = \begin{cases}
	z, & if\ z>0\\
	0, & if\ z<0
\end{cases}
\end{align}
$$
求导
$$
\begin{align}
g(z)^{'} & = \begin{cases}
	1, & if\ z>0\\
	0, & if\ z<0
\end{cases}
\end{align}
$$
缺点：

1. 当输入为负时，梯度为0，梯度消失

优点：

1. 在输入为正数时，不存在梯度消失问题
2. 计算速度快，计算公式是线性关系，sigmod和tanh要计算指数，计算慢