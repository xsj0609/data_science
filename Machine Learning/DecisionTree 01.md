

$$
Gain = -plog_2p-(1-p)log_2(1-p)
$$


举例:

1. 首先计算$H(D) = -\frac{9}{15}log_2\frac{9}{15}-\frac{6}{15}log_2\frac{6}{15}$

2. 然后计算各特征对数据集D的信息增益。分别以$A_1$，$A_2$，$A_3$，$A_4$表示年龄、有工作、有自己的房子和信贷情况4个特征，则

   $g(D,A_1)$

   $=H(D)-[\frac{5}{15}H(D_1)+\frac{5}{15}H(D_2)+\frac{5}{15}H(D_3)]$

   $=0.971-[\frac{5}{15}(-\frac25log_2\frac25-\frac35log_2\frac35)+\frac{5}{15}(-\frac35log_2\frac35-\frac25log_2\frac25)+\frac{5}{15}(-\frac45log_2\frac45-\frac15log_2\frac15)]$

   $=0.971-0.888=0.083$

