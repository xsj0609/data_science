### TF-IDF：

$tfidf(w)=tf(d,w)*idf(w)$

其中，

$tf(d,w)$：文档$d$中$w$的词频

$idf(w)$：$\log\frac{N}{N(w)}$，$N$：语料库中文档的总数，$N(w)$：词语$w$出现在多少个文档



### 举例：

```python
文档1：今天 上 NLP 课程
文档2：今天 的 课程 有 意思
文档3：数据 课程 也 有 意思
```



##### step1，词库的构建

```python
v={今天，上，NLP，课程，的，有，意思，数据，也}，|v|=9
```

##### step2, 转换

$d1=(1*\log\frac32, 1*\log\frac31, 1*\log\frac31, 1*\log\frac33, 0*\log\frac31, 0*\log\frac32, 0*\log\frac32, 0*\log\frac31, 0*\log\frac31)$

$d2=……​$

$d3=……$


