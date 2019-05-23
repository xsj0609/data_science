 先上一张评分卡

![image-20190523133945774](http://ww2.sinaimg.cn/large/006tNc79gy1g3b9j83uydj308l064mxk.jpg)

信贷业务评估的事客户的客户违约率（Percent of Default）即PD，是[0,1]的概率，比如2%即100个客户中有2个违约，简称为p。

评分卡中不直接用客户违约率p，而是用违约概率与正常概率的比值，称为Odds，即![](https://latex.codecogs.com/gif.latex?Odds=\frac{p}{1-p})、![](https://latex.codecogs.com/gif.latex?p=\frac{Odds}{1+Odds})

<b>评分卡的背后逻辑</b>是Odds的<b>变动</b>与评分<b>变动</b>的映射（把Odds映射为评分），分值是根据Odds的前提条件算出来的，不是人工取的。以单个客户在整张评分卡的得分的变动（比如评分从50分上升到70分）来反映Odds的变动（比如Odds从5%下降至1.25%），以及背后相对应的客户违约率PD的变动（比如从4.8%下降到1.2%）。违约率PD不直观、业务看起来不方便、不便计算，而评分就很直观、便于计算。如图所示。

![image-20190523140453482](http://ww2.sinaimg.cn/large/006tNc79gy1g3b9jkmsmtj308q055aac.jpg)

因此评分卡的生成过程，就是Odds变动映射成评分变动的过程。

Odds映射为评分的公式为：

![](https://latex.codecogs.com/gif.latex?Score=A-Blog(\frac{p}{1-p}))

要算出系数A、B的话，需要从业务角度先预设两个前提条件：

1. 在某个<font color='green'>特定的比率</font>![](https://latex.codecogs.com/gif.latex?\theta_0)设定<font color='blue'>特定的预期分值</font>![](https://latex.codecogs.com/gif.latex?P_0)
2. 指定比率翻番时<font color='red'>分数的变动值</font>（PDO）

> 解释：
>
> 1. 比如根据业务经验，消费金融信贷的客户违约率4.8%算正常（![](https://latex.codecogs.com/gif.latex?\theta_0=Odds=5\%)）。预设评分卡的分值为0-100分，那取预期分值![](https://latex.codecogs.com/gif.latex?P_0)为50分，并指定当Odds按双倍上下浮动时（比如2.5%或10%），分值则对应上下<font color='red'>变动10分</font>（比如60分或40分）。
> 2. 这里![](https://latex.codecogs.com/gif.latex?\theta_0)=5%是根据业务经验来的，没有数学依据；
> 3. 0-100分是根据做评分卡的需要来的，没有数学依据。要是想做成600-1000分的评分卡也可以，修改对应的![](https://latex.codecogs.com/gif.latex?P_0)和PDO就行；
> 4. ![](https://latex.codecogs.com/gif.latex?P_0)=50分是根据0-100分来的，也可以取45分或73分，不重要。重要的是随着Odds翻番变动时，分数也随之变动的联动变化体系（你翻番我就变PDO=10分）

设定好![](https://latex.codecogs.com/gif.latex?\theta_0)、![](https://latex.codecogs.com/gif.latex?P_0)、PDO后，联动变化为：Odds( ![](https://latex.codecogs.com/gif.latex?\theta_0))对应的分值为![](https://latex.codecogs.com/gif.latex?P_0)，且翻番的Odds(2![](https://latex.codecogs.com/gif.latex?\theta_0))对应的分值为![](https://latex.codecogs.com/gif.latex?P_0)+PDO。则有以下两式：

![](https://latex.codecogs.com/gif.latex?P_0=A-Blog(\theta_0))

![](https://latex.codecogs.com/gif.latex?P_0+PDO=A-Blog(2\theta_0))

解出A、B为：

![](https://latex.codecogs.com/gif.latex?B=\frac{PDO}{log(2)})

![](https://latex.codecogs.com/gif.latex?A=P_0+Blog(\theta_0))

> 按上面的解释举个例子：
>
> 设![](https://latex.codecogs.com/gif.latex?\theta_0)、![](https://latex.codecogs.com/gif.latex?P_0)、PDO为5%、50分、10分，则
>
> ![](https://latex.codecogs.com/gif.latex?B=\frac{10}{ln(2)}=14.43)
>
> ![](https://latex.codecogs.com/gif.latex?A=50+14.43*ln(0.05)=6.78)
>
> 则
>
> ![](https://latex.codecogs.com/gif.latex?Score = 6.78-14.43log(\frac{p}{1-p}))

按照公式，可以把所有Odds（![](https://latex.codecogs.com/gif.latex?\frac{p}{1-p})）和客户评分、客户违约概率（PD）的对应关系算出来

![image-20190523142625084](http://ww3.sinaimg.cn/large/006tNc79gy1g3b9jr7mlbj30jd0bf0yc.jpg)

那问题来了，现在能算Score了，但输入是Odds。但数据的输入是特征变量[![](https://latex.codecogs.com/gif.latex?x_1,x_2,x_3,\cdots,x_n)]，这里怎么对应呢？这就要说到逻辑回归本身了，先放结论：

![](https://latex.codecogs.com/gif.latex?log(\frac{p}{1-p})=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n)

怎么来的，以下详细讲。

逻辑回归来源于线性回归（二维空间中就是一条直线拟合所有样本点），虽然线性回归是回归算法，逻辑回归是分类算法，但从算法表达式上，逻辑回归就是在线性回归算法外面套了一层壳。

线性回归：

![](https://latex.codecogs.com/gif.latex?f(x)=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n)

![image-20190522163109474](http://ww4.sinaimg.cn/large/006tNc79gy1g3b9jvyypmj309k06pdg5.jpg)

逻辑回归：

![](https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{1+e^{-(\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n)}})

![image-20190522162447433](http://ww2.sinaimg.cn/large/006tNc79gy1g3b9jz9kg2j3080059q35.jpg)

可以看到，从表达式上看，逻辑回归只是在线性回归的表达式外面套了一层![](https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{1+e^{-x}})的壳。为什么要套这层壳，因为线性回归的值域为实数集R，但逻辑回归是二分类算法，需要输出的是类别1和类别2的概率，而概率是个[0, 1]之间的数。因此需要将线性回归的输出实数变成[0, 1]之间的概率，而能满足输入是实数而输出是[0, 1]的，就是Sigmoid函数，它的图形是个类S（见上面逻辑回归图）的限定在[0, 1]之间的函数。因此将Sigmoid函数套在线性回归外面，构成逻辑回归，拥有处理非线性的能力，可以做分类。

那么在信贷评分卡上，![](https://latex.codecogs.com/gif.latex?f(x))即为要预测的客户违约率（PD）![](https://latex.codecogs.com/gif.latex?p)，另将![](https://latex.codecogs.com/gif.latex?\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n)简写为向量形式![](https://latex.codecogs.com/gif.latex?\beta^Tx)，即：

![](https://latex.codecogs.com/gif.latex?p=\frac{1}{1+e^{-\beta^Tx}})

经过变换，可得![](https://latex.codecogs.com/gif.latex?ln(\frac{p}{1-p})=\beta^Tx)

好，回到主线，![](https://latex.codecogs.com/gif.latex?log(\frac{p}{1-p})=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n)，将score公式中的输入p变成输入特征变量X。到这里按理就可以结束了，有X就可以产出客户的Score，即：

![](https://latex.codecogs.com/gif.latex?Score=A-B(\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n))

但我们要做的是分组评分卡，X是要对应到每个分组，得到各变量分组的评分然后相加得到客户总评分的，那就还需要将X打散到各分类（用离散型数据入逻辑回归模型）。因此这里的输入X就不能是原始变量，而是原始变量分箱并算WOE后的woe值（类似离散变量中各类别的数值化），即：

![](https://latex.codecogs.com/gif.latex?Score=A-B[\beta_0+\beta_1(\delta_{11}w_{11}+\delta_{12}w_{12}+\delta_{13}w_{13})+\beta_2(\delta_{21}w_{21}+\delta_{22}w_{22}+\delta_{23}w_{23}+\delta_{24}w_{24})+\cdots+\beta_n(\delta_{n1}w_{n1}+\delta_{n2}w_{n2})])

> 1. 假设类别型变量![](https://latex.codecogs.com/gif.latex?x_1)、![](https://latex.codecogs.com/gif.latex?x_2)、![](https://latex.codecogs.com/gif.latex?x_n)分别有3、4、2个分类（数值型变量先分箱成类别型变量）
> 2. ![](https://latex.codecogs.com/gif.latex?\delta_{ij})代表第i个变量的第j个分类，客户数据参与评分时，某个变量x只会有1个数，只会对应一个分类。比如，变量![](https://latex.codecogs.com/gif.latex?x_1)的取值是第2个分类的话，那![](https://latex.codecogs.com/gif.latex?\delta_{12})为1，则第二个分类的woe值![](https://latex.codecogs.com/gif.latex?w_{12})生效，![](https://latex.codecogs.com/gif.latex?x_1)的其他两个![](https://latex.codecogs.com/gif.latex?\delta)则为0，对应的其他两个分类的woe值无效不参与计算

将上面的公式变下形式，变成最终可以组成评分卡的样式，即：

![](https://latex.codecogs.com/gif.latex?Score=A-B\{\beta_0+(\beta_1w_{11})\delta_{11}+(\beta_1w_{12})\delta_{12}+(\beta_1w_{13})\delta_{13}+(\beta_2w_{21})\delta_{21}+(\beta_2w_{22})\delta_{22}+(\beta_2w_{23})\delta_{23}+(\beta_2w_{24})\delta_{24}+\cdots+(\beta_nw_{n1})\delta_{n1}+(\beta_nw_{n2})\delta_{n2}\})

![](https://latex.codecogs.com/gif.latex?Score=(A-B\beta_0)-(B\beta_1w_{11})\delta_{11}-(B\beta_1w_{12})\delta_{12}-(B\beta_1w_{13})\delta_{13}-(B\beta_1w_{14})\delta_{14}-\cdots-(B\beta_nw_{n1})\delta_{n1}-(B\beta_n w_{n2})\delta_{n2})

![image-20190523150029384](http://ww3.sinaimg.cn/large/006tNc79gy1g3b9k40w7oj30b30c8wex.jpg)



> A、B已经算出，![](https://latex.codecogs.com/gif.latex?\beta)是逻辑回归模型的输出系数，![](https://latex.codecogs.com/gif.latex?\beta_0)是逻辑回归模型的输出截距项，w是分箱后的woe值

嗯，至此评分卡就可以生成了。