## 得到LogisR的代价函数$L(w,b)$



LogisR与LinearR的$h_\theta$函数不一样，若$L(w,b) $直接仿照LinearR【$ L(w,b)=\frac{1}{2m} \sum(p(y=1|x,w)-y^{(i)})^2$】，导数会得到非凸优化（不能用梯度下降求解）。

$L(w,b)$的导数得是凸优化，因为$h_\theta$不一样，所以$L(w,b)$另行考虑为：
$$
L(w,b)= \sum_{i=1}^{m}Cost(p(y=1|x,w), y^{(i)})
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
L(w,b)=- \sum_{i=1}^{m}\left[y^{(i)}log(p(y=1|x,w))+(1-y^{(i)})log(1-p(y=1|x,w))\right] \tag{p.s. for code}
$$

## 论证LogisR代价函数$L(w,b)$的梯度下降求解算法（导数）与LinearR的一样，也是凸优化问题

如果是凸优化问题，代价函数的导数应为$ \sum_{i=1}^{m}(p(y=1|x,w)-y^{(i)})x_j^{(i)}$形式

论证：

LogisR代价函数为：
$$
L(w,b)=- \sum_{i=1}^{m}\left[y^{(i)}log(p(y=1|x,w))+(1-y^{(i)})log(1-p(y=1|x,w))\right]
$$
将$p(y=1|x,w)=\frac{1}{1+e^{-(w^Tx^{(i)}+b)}}$带入，则：
$$
\begin{align}
    L(w,b) & = - \sum_{i=1}^{m} \left[ y^{(i)} log(\frac{1}{ 1+e^{-(w^Tx^{(i)}+b)} })+ (1-y^{(i)}) log(1-\frac{1}{ 1+e^{-(w^Tx^{(i)}+b)} }) \right]\\
    & =- \sum_{i=1}^{m} \left[ -y^{(i)} log( 1+e^{-(w^Tx^{(i)}+b)})- (1-y^{(i)}) log(1+e^{w^Tx^{(i)}+b}) \right] 
\end{align}
$$
所以
$$
\begin{align}
	\frac{\partial}{\partial w}L(w,b) &= \frac{\partial}{\partial\theta_j}\left[- \sum_{i=1}^{m}\left[y^{(i)}log(p(y=1|x,w))+(1-y^{(i)})log(1-p(y=1|x,w))\right] \right] \tag{1}\\
	& = - \sum_{i=1}^{m} \left[ y^{(i)} log(\frac{1}{ 1+e^{-(w^Tx^{(i)}+b)} })+ (1-y^{(i)}) log(1-\frac{1}{ 1+e^{-(w^Tx^{(i)}+b)} }) \right]\\
  & =- \sum_{i=1}^{m} \left[ -y^{(i)} log( 1+e^{-(w^Tx^{(i)}+b)})- (1-y^{(i)}) log(1+e^{w^Tx^{(i)}+b}) \right]\\
	& = - \sum_{i=1}^{m} \left[ -y^{(i)} \frac{-x_j^{(i)}e^{-(w^Tx^{(i)}+b)}}{1+e^{-(w^Tx^{(i)}+b)}}- (1-y^{(i)}) \frac{x_j^{(i)}e^{w^Tx^{(i)}+b}}{1+e^{w^Tx^{(i)}+b}} \right] \tag{2} \\
	& = - \sum_{i=1}^{m} \left[ y^{(i)} \frac{x_j^{(i)}}{1+e^{w^Tx^{(i)}+b}}- (1-y^{(i)}) \frac{x_j^{(i)}e^{w^Tx^{(i)}+b}}{1+e^{w^Tx^{(i)}+b}} \right] \tag{3} \\
	& = - \sum_{i=1}^{m} \frac{y^{(i)}x_j^{(i)}-x_j^{(i)}e^{w^Tx^{(i)}+b}+y^{(i)}x_j^{(i)}e^{w^Tx^{(i)}+b}}{1+e^{w^Tx^{(i)}+b}} \tag{4} \\
	& = - \sum_{i=1}^{m} \frac{y^{(i)}(1+e^{w^Tx^{(i)}+b})-e^{w^Tx^{(i)}+b}}{1+e^{w^Tx^{(i)}+b}}x_j^{(i)} \tag{5} \\
	& = - \sum_{i=1}^{m} \left(y^{(i)}-\frac{e^{w^Tx^{(i)}+b}}{1+e^{w^Tx^{(i)}+b}} \right) x_j^{(i)} \tag{6} \\
	& = - \sum_{i=1}^{m} \left(y^{(i)}-\frac{1}{1+e^{-(w^Tx^{(i)}+b)}} \right) x_j^{(i)} \tag{7} \\
	& = - \sum_{i=1}^{m} \left(y^{(i)}-p(y=1|x,w) \right) x_j^{(i)} \tag{8} \\
	& =  \sum_{i=1}^{m} \left(p(y=1|x,w)-y^{(i)} \right) x_j^{(i)} \tag{9}
\end{align}
$$
