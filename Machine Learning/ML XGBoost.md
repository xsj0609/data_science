### 一、目标函数

首先，定义损失函数为：

<img src="https://latex.codecogs.com/svg.latex?L(\theta)=\sum_{i=1}^nl(y_i,\hat{y}_i^{(t)})+\sum_{k=1}^t\Omega(f_k)" align="left">

### 二、树模型累加集成

XGBoost的指导思想就是用后一棵树拟合前一棵树的残差，然后将多棵树结果相加进行预测。因此<img src="https://latex.codecogs.com/svg.latex?\hat{y}_i^{(t)}">是有先后顺序的多棵树累加的结果。

加法策略可以表示如下：

初始化（模型中没有树时，其预测结果为0）：<img src="https://latex.codecogs.com/svg.latex?\hat{y}_i^{(0)}=0">

往模型中加入第一棵树：<img src="https://latex.codecogs.com/svg.latex?\hat{y}_i^{(1)}=f_1(x_i)=\hat{y}_i^{(0)} + f_1(x_i)">

往模型中加入第二棵树：<img src="https://latex.codecogs.com/svg.latex?\hat{y}_i^{(2)} = f_1(x_i) + f_2(x_i) = \hat{y}_i^{(1)} + f_2(x_i)">

<img src="https://latex.codecogs.com/svg.latex?\cdots">

往模型中加入第t棵树：<img src="https://latex.codecogs.com/svg.latex?\hat{y}_i^{(t)}=\sum_{k=1}^t f_k(x_i) = \hat{y}_i^{(t-1)} + f_t(x_i)">

其中，<img src="https://latex.codecogs.com/svg.latex?f_k">表示第<img src="https://latex.codecogs.com/svg.latex?k">棵树，<img src="https://latex.codecogs.com/svg.latex?\hat{y}_i^{(t)}">表示组合<img src="https://latex.codecogs.com/svg.latex?t">棵树的模型对样本<img src="https://latex.codecogs.com/svg.latex?x_i">的预测结果



则损失函数可以变为：

<img src="https://latex.codecogs.com/svg.latex?L(\theta)=\sum_{i=1}^nl(y_i,\hat{y}_i^{(t)})+\sum_{k=1}^t\Omega(f_k)" align="left">

<img src="https://latex.codecogs.com/svg.latex?\dot\qquad\quad=\sum_{i=1}^nl(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\Omega(f_t)+C" align="left">



### 三、正则

另外对于目标损失函数中的正则项（复杂度）部分，我们从单一的树来考虑。对于其中每一棵回归树，其模型可以写为：

<img src="https://latex.codecogs.com/svg.latex?f_t(x)=w_{q(x)},\quad w\in R^T,\quad q:R^d\rightarrow {1,2,\cdots,T}">

其中w为叶子节点的得分值，q(x)表示样本x对应的叶子节点，T为该树的叶子节点个数。

因此，在这里，我们将该树的复杂度写成：

<img src="https://latex.codecogs.com/svg.latex?\Omega(f_t)=\gamma T+\frac12 \lambda \sum_{j=1}^T w_j^2">

其中T为叶子个数，<img src="https://latex.codecogs.com/svg.latex?w_j^2">为<img src="https://latex.codecogs.com/svg.latex?w">的L2模平方

> 复杂度计算示例：

![image-20190612161734846](http://ww3.sinaimg.cn/large/006tNc79gy1g3yg1ic1m6j305x03z3ze.jpg)

> <img src="https://latex.codecogs.com/svg.latex?\Omega=\gamma3+\frac12 \lambda (4+0.01+1)">



此时，对于XGBoost的目标函数我们可以写为：

<img src="https://latex.codecogs.com/svg.latex?L(\theta)=\sum_{i=1}^nl(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\Omega(f_t)+C" align="left">

<img src="https://latex.codecogs.com/svg.latex?\dot\qquad\quad=\sum_{i=1}^n l(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\gamma T+\frac12 \lambda \sum_{j=1}^T w_j^2 + C" align="left">

### 四、目标函数求解

#### 1. 泰勒展开

泰勒展开式：<img src="https://latex.codecogs.com/svg.latex?f(x+\Delta x) \approx f(x) + f\prime(x) \Delta(x) + \frac12 f\prime\prime(x) \Delta(x)^2 + \cdots">

这里用泰勒展开式来近似原来的目标函数，将<img src="http://latex.codecogs.com/svg.latex?f_t(x_i)">当作<img src="http://latex.codecogs.com/svg.latex?\Delta(x)">，则原目标函数可以写成：

<img src="https://latex.codecogs.com/svg.latex?L(\theta)=\sum_{i=1}^n l(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\gamma T+\frac12 \lambda \sum_{j=1}^T w_j^2 + C" align="left">

<img src="https://latex.codecogs.com/svg.latex?\approx \sum_{i=1}^n \left[l(y_i,\hat{y}_i^{(t-1)}) + \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)}) f_t(x_i) + \frac12 \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)}) f_t(x_i)^2 \right]+\gamma T+\frac12 \lambda \sum_{j=1}^T w_j^2 + C" align="left">

#### 2. 样本误差转换为叶节点误差

令<img src="https://latex.codecogs.com/svg.latex?g_i=\partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})">、<img src="https://latex.codecogs.com/svg.latex?h_i=\partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})">，同时对于第t棵树，<img src="https://latex.codecogs.com/svg.latex?l(y_i,\hat{y}_i^{(t-1)})">为常数。故目标损失函数可以写成：

<img src="https://latex.codecogs.com/svg.latex?L(\theta) \approx \sum_{i=1}^n \left[l(y_i,\hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac12 h_i f_t(x_i)^2 \right]+\gamma T+\frac12 \lambda \sum_{j=1}^T w_j^2 + C" align="left">

<img src="https://latex.codecogs.com/svg.latex?\dot\qquad\quad \approx \sum_{i=1}^n \left[g_i f_t(x_i) + \frac12 h_i f_t(x_i)^2 \right]+\gamma T+\frac12 \lambda \sum_{j=1}^T w_j^2 \qquad \{3\}" align="left">

<img src="https://latex.codecogs.com/svg.latex?\dot\qquad\quad \approx \sum_{j=1}^T \left[ (\sum_{i\in I_j} g_i) w_j + (\frac12 \sum_{i\in I_j} h_i) w_j^2 \right] + \gamma T + \frac12 \lambda \sum_{j=1}^T w_j^2 \qquad \{4\}" align="left">

<img src="https://latex.codecogs.com/svg.latex?\dot\qquad\quad \approx \sum_{j=1}^T \left[ (\sum_{i\in I_j} g_i) w_j + (\frac12 \sum_{i\in I_j} h_i + \lambda) w_j^2 \right] + \gamma T" align="left">

> 其中，<img src="https://latex.codecogs.com/svg.latex?T">为第<img src="https://latex.codecogs.com/svg.latex?t">棵树中总叶子节点的个数，<img src="https://latex.codecogs.com/svg.latex?I_j=\{i|q(x_i)=j\}">表示在第<img src="https://latex.codecogs.com/svg.latex?j">个叶子节点上的样本，<img src="https://latex.codecogs.com/svg.latex?w_j">为第<img src="https://latex.codecogs.com/svg.latex?j">个叶子节点的得分值。

> 一个叶子节点的得分值可以理解为满足同一条件的多个样本（决策树的某条分支）的预测值prediction，训练过程中某条分支上的多个样本的预测值都等于所处叶子节点的得分值。

> 在由公式3到公式4的推导中，代价函数的目标不变，仍然是最小化<img src="https://latex.codecogs.com/svg.latex?l(y_i,\hat{y}_i^{(t)})">，也就是原标签和树预测值的差距最小。只是计算逻辑从先算单个样本的误差再汇总所有样本，也就是<img src="https://latex.codecogs.com/svg.latex?\sum_{i=1}^n">，变成先算单个叶节点的误差，再汇总所有叶节点，也就是<img src="https://latex.codecogs.com/svg.latex?\sum_{j=1}^T">。而之所以能转换，是因为叶节点和样本x之间是对应的<img src="https://latex.codecogs.com/svg.latex?f_t(x)=w_{q(x)}">。而为什么要转换，自然是无法直接求解误差函数，而转换后就可以用树模型的训练、叶节点值的迭代来进行误差函数的计算了。

#### 3. 得到解表达式

令<img src="https://latex.codecogs.com/svg.latex?G_j=\sum_{i\in I_j} g_i">，<img src="https://latex.codecogs.com/svg.latex?H_j=\sum_{i\in I_j} h_i">，则：

<img src="https://latex.codecogs.com/svg.latex?L(\theta) = \sum_{j=1}^T \left [G_j w_j + \frac12 (H_j+\lambda) w_j^2 \right] + \gamma T">

对<img src="https://latex.codecogs.com/svg.latex?w_j">求偏导，并使其导数等于0，则：

<img src="https://latex.codecogs.com/svg.latex?G_j + (H_j + \lambda) w_j = 0">

求解得：

<img src="https://latex.codecogs.com/svg.latex?w_j^* = - \frac{G_j}{H_j+\lambda}">

即当<img src="https://latex.codecogs.com/svg.latex?w_j=-\frac{G_j}{H_j+\lambda}">时，目标函数<img src="https://latex.codecogs.com/svg.latex?L(\theta)">最小，此时<img src="https://latex.codecogs.com/svg.latex?L(\theta)= - \frac12 \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda}">。而该目标函数最小值表达式则可以用作决策树分支依据，即比较分支后的两部分<img src="https://latex.codecogs.com/svg.latex?L(\theta)">之和与分支前的<img src="https://latex.codecogs.com/svg.latex?L(\theta)">，即<img src="https://latex.codecogs.com/svg.latex?Gain">。比较不同变量不同类别值的<img src="https://latex.codecogs.com/svg.latex?Gain">，做出分支。即分支计算：

<img src="https://latex.codecogs.com/svg.latex?Gain=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma" align="left">