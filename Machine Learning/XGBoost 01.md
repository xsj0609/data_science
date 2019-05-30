## 得到LogisR的代价函数$J(\theta)$


LogisR与LinearR的$h_\theta​$函数不一样，若$J(\theta) ​$直接仿照LinearR【$ J(\theta)=\frac{1}{2m} \sum(h_\theta(x^{(i)})-y^{(i)})^2​$】，导数会得到非凸优化（不能用梯度下降求解）。

$J(\theta)​$的导数得是凸优化，因为$h_\theta​$不一样，所以$J(\theta)​$另行考虑为：
$$
J(\theta)=\frac1m \sum_{i=1}^{m}Cost(h_\theta(x^{(i)}), y^{(i)})
$$
其中
$$
\begin{align}
Cost(h_\theta(x), y) & = \begin{cases}
	-log(h_\theta(x)), & if\ y=1\\
	-log(1-h_\theta(x)), & if\ y=0
\end{cases}\\
& (p.s. 一条单调减曲线+一条单调增曲线合成凸形状) \\
& = -ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))
\end{align}
$$
因此
$$
J(\theta)=-\frac1m \sum_{i=1}^{m}\left[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))\right] \tag{p.s. for code}
$$

## 论证LogisR代价函数$J(\theta)$的梯度下降求解算法（导数）与LinearR的一样，也是凸优化问题

如果是凸优化问题，代价函数的导数应为$\frac1m \sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$形式

论证：

LogisR代价函数为：
$$
J(\theta)=-\frac1m \sum_{i=1}^{m}\left[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))\right]
$$
将$h_\theta(x^{(i)})=\frac{1}{1+e^{-\theta^Tx^{(i)}}}$带入，则：
$$
\begin{align}
    J(\theta) & = -\frac1m \sum_{i=1}^{m} \left[ y^{(i)} log(\frac{1}{ 1+e^{-\theta^Tx^{(i)}} })+ (1-y^{(i)}) log(1-\frac{1}{ 1+e^{-\theta^Tx^{(i)}} }) \right]\\
    & =-\frac1m \sum_{i=1}^{m} \left[ -y^{(i)} log( 1+e^{-\theta^Tx^{(i)}})- (1-y^{(i)}) log(1+e^{\theta^Tx^{(i)}}) \right]
\end{align}
$$
所以
$$
\begin{align}
	\frac{\partial}{\partial\theta_j}J(\theta) &= \frac{\partial}{\partial\theta_j}\left[-\frac1m \sum_{i=1}^{m} \left[ -y^{(i)} log( 1+e^{-\theta^Tx^{(i)}})- (1-y^{(i)}) log(1+e^{\theta^Tx^{(i)}}) \right] \right] \tag{1}\\
	& = -\frac1m \sum_{i=1}^{m} \left[ -y^{(i)} \frac{-x_j^{(i)}e^{-\theta^Tx^{(i)}}}{1+e^{-\theta^Tx^{(i)}}}- (1-y^{(i)}) \frac{x_j^{(i)}e^{\theta^Tx^{(i)}}}{1+e^{\theta^Tx^{(i)}}} \right] \tag{2} \\
	& = -\frac1m \sum_{i=1}^{m} \left[ y^{(i)} \frac{x_j^{(i)}}{1+e^{\theta^Tx^{(i)}}}- (1-y^{(i)}) \frac{x_j^{(i)}e^{\theta^Tx^{(i)}}}{1+e^{\theta^Tx^{(i)}}} \right] \tag{3} \\
	& = -\frac1m \sum_{i=1}^{m} \frac{y^{(i)}x_j^{(i)}-x_j^{(i)}e^{\theta^Tx^{(i)}}+y^{(i)}x_j^{(i)}e^{\theta^Tx^{(i)}}}{1+e^{\theta^Tx^{(i)}}} \tag{4} \\
	& = -\frac1m \sum_{i=1}^{m} \frac{y^{(i)}(1+e^{\theta^Tx^{(i)}})-e^{\theta^Tx^{(i)}}}{1+e^{\theta^Tx^{(i)}}}x_j^{(i)} \tag{5} \\
	& = -\frac1m \sum_{i=1}^{m} \left(y^{(i)}-\frac{e^{\theta^Tx^{(i)}}}{1+e^{\theta^Tx^{(i)}}} \right) x_j^{(i)} \tag{6} \\
	& = -\frac1m \sum_{i=1}^{m} \left(y^{(i)}-\frac{1}{1+e^{-\theta^Tx^{(i)}}} \right) x_j^{(i)} \tag{7} \\
	& = -\frac1m \sum_{i=1}^{m} \left(y^{(i)}-h_\theta(x^{(i)}) \right) x_j^{(i)} \tag{8} \\
	& = \frac1m \sum_{i=1}^{m} \left(h_\theta(x^{(i)})-y^{(i)} \right) x_j^{(i)} \tag{9}
\end{align}
$$


$$
Gain = -plog_2p-(1-p)log_2(1-p)
$$


举例:

1. 首先计算$H(D) = -\frac{9}{15}log_2\frac{9}{15}-\frac{6}{15}log_2\frac{6}{15}$

2. 然后计算各特征对数据集D的信息增益。分别以$A_1​$，$A_2​$，$A_3​$，$A_4​$表示年龄、有工作、有自己的房子和信贷情况4个特征，则

   $g(D,A_1)$

   $=H(D)-[\frac{5}{15}H(D_1)+\frac{5}{15}H(D_2)+\frac{5}{15}H(D_3)]​$

   $=0.971-[\frac{5}{15}(-\frac25log_2\frac25-\frac35log_2\frac35)+\frac{5}{15}(-\frac35log_2\frac35-\frac25log_2\frac25)+\frac{5}{15}(-\frac45log_2\frac45-\frac15log_2\frac15)]$

   $=0.971-0.888=0.083$


$$
Gain = \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} - \gamma
$$