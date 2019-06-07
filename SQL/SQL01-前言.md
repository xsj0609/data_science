### 前言

首先，推荐一个SQL的工具书类网站，平时写SQL时记不清楚的语法直接到这查询就好

http://www.w3school.com.cn/sql/index.asp

### SQL特性

SQL虽然也是“编程”，但它与C、Java这些“技术部”使用的开发语言不太一样，完全不涉及技术底层。基本上只要会用Excel的（筛选、排序、vlookup匹配等），就都能学会SQL。在每天都能抽出一到两个小时持续刻意练习的前提下，快则一周慢则一月，就能学会。从而摆脱哪怕知道是一点简单的数据统计需求，自己都做不了，要找人帮忙的尴尬境地。

### 总框架

SQL总体上可以看作乐高积木的搭建，虽有不同形状，但更重要的是组合。原始形状并不多，无论多复杂的数据需求，只要SQL能做出来的，都是通过基础形状，按一定的规则，搭出来的。（这里只针对一般的SQL使用，不包括存储过程、触发器等略偏深度的数据库使用）

大体框架如下：

```sql
-- 基础形状1
-- 单表
SELECT (字段1，字段2)
FROM (表名)
```

```sql
-- 基础形状2
-- 多表+条件筛选
SELECT (字段1，字段2)
FROM (表1)
LEFT JOIN (表2)
ON ()
WHERE (条件)
GROUP BY (聚合)
```

是不是看上去也不难？其实SQL就这点东西，来个例子看下

### 举例

SQL的学习和练习推荐网上的《SQL经典50题》，是学生、课程、教师、成绩四张表的那套，这里暂不展开介绍，先通过这套数据了解下SQL的基础框架

1. 基础形状1：单表

![](http://ww1.sinaimg.cn/large/006tNc79gy1g3sdgr9xg9j306905ma9v.jpg)



2. 基础形状2：多表

   ![](http://ww4.sinaimg.cn/large/006tNc79gy1g3sdhdiu4vj30j40e4q35.jpg)

![](http://ww4.sinaimg.cn/large/006tNc79gy1g3sdhn44lsj30cw087glj.jpg)

如上图所示：

1. SQL语句中的蓝色字体为SQL语言关键字，无论主框架还是子积木部分中的关键字都是一样的（SQL关键字其实真不多，重点是组合，常用关键字下面会列出来供大家直接学习重点）
2. 可以发现无论主框架还是子积木部分，语句组合是很类似的。实际上SQL语言就是不断搭积木的过程（图中SQL只有两层，实际使用时，会有多层嵌套，都会在FROM或JOIN关键字后面一层层的嵌套）

### 关键字

如果访问w3school的SQL语法内容（仅以该网站内容举例，一个工具属性的网站也没啥好打广告的），从左侧导航目录部分会发现有好些内容，新学SQL的看官一上手就觉得麻爪，不知道从何入手。SQL君就来给看官大人们选中平时用的最多的几个语法，保证不多。

1. SQL基础教程

   select、where、AND & OR、Order By

   （定义输出字段、条件、多条件组合、排序）

2. SQL高级教程

   Like、通配符、In、Between、Left Join、Union、Not Null

   （模糊匹配、模糊匹配、条件、条件、多表联合、多表联合、非空判断）

3. SQL函数

   avg()、count()、max()、min()、sum()、Group By、Having、len()、round()

   （平均、计数、最大、最小、求和、聚合、聚合条件、长度、小数位数）

以上内容基本上涵盖日常八成的一般统计、BI报告需求了。

### 组合

关键字熟悉后，就是对关键字的组合来实现对数据的操作了。SQL对数据的操作我总结最重要的就两点：拼表、聚合。

拼表：用JOIN关键字，实现对多张表的合并。

聚合：通过GROUP BY关键字，实现对数据按某个维度的分组计算。

下面通过上面的SQL50题第14题的详细剖析来展示一段SQL语言的运行细节。

GROUP BY Cid，按Cid列分组

![](http://ww4.sinaimg.cn/large/006tNc79gy1g3sdhn1r1aj30jg0c4weq.jpg)

![](http://ww3.sinaimg.cn/large/006tNc79gy1g3sdhmzgg5j307l0l2glr.jpg)

各分组内部做聚合计算

![](http://ww2.sinaimg.cn/large/006tNc79gy1g3sdhmx4quj30ji0ce0sz.jpg)

![](http://ww4.sinaimg.cn/large/006tNc79gy1g3sdhmpwerj30h50keq39.jpg)

