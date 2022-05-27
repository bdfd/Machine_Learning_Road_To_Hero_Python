# 机器学习 100 天

英文原版请移步[Avik-Jain](https://github.com/Avik-Jain/100-Days-Of-ML-Code)。数据在[这里](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/tree/main/datasets)。

翻译前请先阅读[规范](Translation%20specification.MD)。常见问题解答见[FAQ](FAQ.MD)。

# 目录

- 有监督学习
  - [数据预处理](#数据预处理--第1天)
  - [简单线性回归](#简单线性回归--第2天)
  - [多元线性回归](#多元线性回归--第3天)
  - [逻辑回归](#逻辑回归--第4天)
  - [k 近邻法(k-NN)](#k近邻法k-nn--第7天)
  - [支持向量机(SVM)](#支持向量机svm--第12天)
  - [决策树](#决策树--第23天)
  - [随机森林](#随机森林--第33天)
- 无监督学习
  - [K-均值聚类](#k-均值聚类--第43天)
  - [层次聚类](#层次聚类--第54天)

## 数据预处理 | 第 1 天

[数据预处理实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%201_Data_Preprocessing.md)

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%201.jpg">
</p>

## 简单线性回归 | 第 2 天

[简单线性回归实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%202_Simple_Linear_Regression.md)

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%202.jpg">
</p>

## 多元线性回归 | 第 3 天

[多元线性回归实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%203_Multiple_Linear_Regression.md)

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%203.png">
</p>

## 逻辑回归 | 第 4 天

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%204.jpg">
</p>

## 逻辑回归 | 第 5 天

今天我深入研究了逻辑回归到底是什么，以及它背后的数学是什么。学习了如何计算代价函数，以及如何使用梯度下降法来将代价函数降低到最小。<br>
由于时间关系，我将隔天发布信息图。如果有人在机器学习领域有一定经验，并愿意帮我编写代码文档，也了解 github 的 Markdown 语法，请在领英联系我。

## 逻辑回归 | 第 6 天

[逻辑回归实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%206_Logistic_Regression.md)

## K 近邻法(k-NN) | 第 7 天

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%207.jpg">
</p>

## 逻辑回归背后的数学 | 第 8 天

为了使我对逻辑回归的见解更加清晰，我在网上搜索了一些资源或文章，然后我就发现了 Saishruthi Swaminathan 的<a href = "https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc">这篇文章</a><br>

它给出了逻辑回归的详细描述。请务必看一看。

## 支持向量机(SVM) | 第 9 天

直观了解 SVM 是什么以及如何使用它来解决分类问题。

## 支持向量机和 K 近邻法 | 第 10 天

了解更多关于 SVM 如何工作和实现 knn 算法的知识。

## K 近邻法(k-NN) | 第 11 天

[K 近邻法(k-NN)实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2011_K-NN.md)

## 支持向量机(SVM) | 第 12 天

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%2012.jpg">
</p>

## 支持向量机(SVM) | 第 13 天

[SVM 实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2013_SVM.md)

## 支持向量机(SVM)的实现 | 第 14 天

今天我在线性相关数据上实现了 SVM。使用 Scikit-Learn 库。在 scikit-learn 中我们有 SVC 分类器，我们用它来完成这个任务。将在下一次实现时使用 kernel-trick。Python 代码见[此处](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2013_SVM.py),Jupyter notebook 见[此处](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2013_SVM.ipynb)。

## 朴素贝叶斯分类器(Naive Bayes Classifier)和黑盒机器学习(Black Box Machine Learning) | 第 15 天

学习不同类型的朴素贝叶斯分类器同时开始<a href="https://bloomberg.github.io/foml/#home">Bloomberg</a>的课程。课程列表中的第一个是黑盒机器学习。它给出了预测函数，特征提取，学习算法，性能评估，交叉验证，样本偏差，非平稳性，过度拟合和超参数调整的整体观点。

## 通过内核技巧实现支持向量机 | 第 16 天

使用 Scikit-Learn 库实现了 SVM 算法以及内核函数，该函数将我们的数据点映射到更高维度以找到最佳超平面。

## 在 Coursera 开始深度学习的专业课程 | 第 17 天

在 1 天内完成第 1 周和第 2 周内容以及学习课程中的逻辑回归神经网络。

## 继续 Coursera 上的深度学习专业课程 | 第 18 天

完成课程 1。用 Python 自己实现一个神经网络。

## 学习问题和 Yaser Abu-Mostafa 教授 | 第 19 天

开始 Yaser Abu-Mostafa 教授的 Caltech 机器学习课程-CS156 中的课程 1。这基本上是对即将到来的课程的一种介绍。他也介绍了感知算法。

## 深度学习专业课程 2 | 第 20 天

完成改进深度神经网络第 1 周内容：参数调整，正则化和优化。

## 网页搜罗 | 第 21 天

观看了一些关于如何使用 Beautiful Soup 进行网络爬虫的教程，以便收集用于构建模型的数据。

## 学习还可行吗? | 第 22 天

完成 Yaser Abu-Mostafa 教授的 Caltech 机器学习课程-CS156 中的课程 2。学习 Hoeffding 不等式。

## 决策树 | 第 23 天

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%2023%20-%20Chinese.jpg">
</p>

## 统计学习理论的介绍 | 第 24 天

Bloomberg ML 课程的第 3 课介绍了一些核心概念，如输入空间，动作空间，结果空间，预测函数，损失函数和假设空间。

## 决策树 | 第 25 天

[决策树实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2025_Decision_Tree.md)

## 跳到复习线性代数 | 第 26 天

发现 YouTube 一个神奇的频道[3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)，它有一个播放列表《线性代数的本质》。看完了 4 个视频，包括了向量，线性组合，跨度，基向量，线性变换和矩阵乘法。

B 站播放列表在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=9450)。

## 跳到复习线性代数 | 第 27 天

继续观看了 4 个视频，内容包括三维变换、行列式、逆矩阵、列空间、零空间和非方矩阵。

B 站播放列表在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=9450)。

## 跳到复习线性代数 | 第 28 天

继续观看了 3 个视频，内容包括点积和叉积。

B 站播放列表在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=9450)。

## 跳到复习线性代数 | 第 29 天

观看了剩余的视频 12 到 14，内容包括特征向量和特征值，以及抽象向量空间。

B 站播放列表在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=9450)。

## 微积分的本质 | 第 30 天

完成上一播放列表后，YouTube 推荐了新内容《微积分的本质》，今天看完了其中的 3 个视频，包括导数、链式法则、乘积法则和指数导数。

B 站播放列表在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=13407)。

## 微积分的本质 | 第 31 天

观看了 2 个视频，内容包括隐分化与极限。

B 站播放列表在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=13407)。

## 微积分的本质 | 第 32 天

观看了剩余的 4 个视频，内容包括积分与高阶导数。

B 站播放列表在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=13407)。

## 随机森林 | 第 33 天

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%2033.png">
</p>

## 随机森林 | 第 34 天

[随机森林实现](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2034_Random_Forests.md)

## 什么是神经网络？ | 深度学习，第 1 章 | 第 35 天

Youtube 频道 3Blue1Brown 中有精彩的视频介绍神经网络。这个视频提供了很好的解释，并使用手写数字数据集演示基本概念。

B 站视频在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=26587)。

## 梯度下降法，神经网络如何学习 | 深度学习，第 2 章 | 第 36 天

Youtube 频道 3Blue1Brown 关于神经网络的第 2 部分，这个视频用有趣的方式解释了梯度下降法。推荐必须观看 169.

B 站视频在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=26587)。

## 反向传播法究竟做什么？ | 深度学习，第 3 章 | 第 37 天

Youtube 频道 3Blue1Brown 关于神经网络的第 3 部分，这个视频主要介绍了偏导数和反向传播法。

B 站视频在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=26587)。

## 反向传播法演算 | 深度学习，第 4 章 | 第 38 天

Youtube 频道 3Blue1Brown 关于神经网络的第 3 部分，这个视频主要介绍了偏导数和反向传播法。

B 站视频在[这里](https://space.bilibili.com/88461692/#/channel/detail?cid=26587)。

## 第 1 部分 | 深度学习基础 Python，TensorFlow 和 Keras | 第 39 天

视频地址在[这里](https://www.youtube.com/watch?v=wQ8BIBpya2k&t=19s&index=2&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)。
<br>中文文字版[notebook](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2039.ipynb)。

## 第 2 部分 | 深度学习基础 Python，TensorFlow 和 Keras | 第 40 天

视频地址在[这里](https://www.youtube.com/watch?v=wQ8BIBpya2k&t=19s&index=2&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)。
<br>中文文字版[notebook](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2040.ipynb)。

## 第 3 部分 | 深度学习基础 Python，TensorFlow 和 Keras | 第 41 天

视频地址在[这里](https://www.youtube.com/watch?v=wQ8BIBpya2k&t=19s&index=2&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)。
<br>中文文字版[notebook](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2041.ipynb)。

## 第 4 部分 | 深度学习基础 Python，TensorFlow 和 Keras | 第 42 天

视频地址在[这里](https://www.youtube.com/watch?v=wQ8BIBpya2k&t=19s&index=2&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)。
<br>中文文字版[notebook](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Code/Day%2042.ipynb)。

## K-均值聚类 | 第 43 天

转到无监督学习，并研究了聚类。可在[作者网站](http://www.avikjain.me/)查询。发现一个奇妙的[动画](http://shabal.in/visuals/kmeans/6.html)有助于理解 K-均值聚类。

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%2043.jpg">
</p>

## K-均值聚类 | 第 44 天

实现（待添加代码）

## 深入研究 | NUMPY | 第 45 天

得到 JK VanderPlas 写的书《Python 数据科学手册（Python Data Science HandBook）》，Jupyter notebooks 在[这里](https://github.com/jakevdp/PythonDataScienceHandbook)。
<br>**[高清中文版 pdf](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Other%20Docs/Python%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E6%89%8B%E5%86%8C.zip)**
<br>第 2 章：NumPy 介绍，包括数据类型、数组和数组计算。
<br>代码如下：
<br>[2 NumPy 入门](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.00-Introduction-to-NumPy.ipynb)
<br>[2.1 理解 Python 中的数据类型](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.01-Understanding-Data-Types.ipynb)
<br>[2.2 NumPy 数组基础](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb)
<br>[2.3 NumPy 数组的计算：通用函数](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.03-Computation-on-arrays-ufuncs.ipynb)

## 深入研究 | NUMPY | 第 46 天

第 2 章： 聚合, 比较运算符和广播。
<br>代码如下：
<br>[2.4 聚合：最小值、最大值和其他值](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.04-Computation-on-arrays-aggregates.ipynb)
<br>[2.5 数组的计算：广播](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.05-Computation-on-arrays-broadcasting.ipynb)
<br>[2.6 比较、掩码和布尔运算](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.06-Boolean-Arrays-and-Masks.ipynb)

## 深入研究 | NUMPY | 第 47 天

第 2 章： 花哨的索引，数组排序，结构化数据。
<br>代码如下：
<br>[2.7 花哨的索引](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.07-Fancy-Indexing.ipynb)
<br>[2.8 数组的排序](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.08-Sorting.ipynb)
<br>[2.9 结构化数据：NumPy 的结构化数组](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/02.09-<br>Structured-Data-NumPy.ipynb)

## 深入研究 | PANDAS | 第 48 天

第 3 章：Pandas 数据处理
<br>包含 Pandas 对象，数据取值与选择，数值运算方法，处理缺失值，层级索引，合并数据集。
<br>代码如下：
<br>[3 Pandas 数据处理](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.00-Introduction-to-Pandas.ipynb)
<br>[3.1 Pandas 对象简介](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.01-Introducing-Pandas-Objects.ipynb)
<br>[3.2 数据取值与选择](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.02-Data-Indexing-and-Selection.ipynb)
<br>[3.3 Pandas 数值运算方法](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.03-Operations-in-Pandas.ipynb)
<br>[3.4 处理缺失值](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.04-Missing-Values.ipynb)
<br>[3.5 层级索引](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.05-Hierarchical-Indexing.ipynb)
<br>[3.6 合并数据集：ConCat 和 Append 方法](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.06-Concat-And-Append.ipynb)

## 深入研究 | PANDAS | 第 49 天

第 3 章：完成剩余内容-合并与连接，累计与分组，数据透视表。
<br>代码如下：
<br>[3.7 合并数据集：合并与连接](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.07-Merge-and-Join.ipynb)
<br>[3.8 累计与分组](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.08-Aggregation-and-Grouping.ipynb)
<br>[3.9 数据透视表](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.09-Pivot-Tables.ipynb)

## 深入研究 | PANDAS | 第 50 天

第 3 章：向量化字符串操作，处理时间序列。
<br>代码如下：
<br>[3.10 向量化字符串操作](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.10-Working-With-Strings.ipynb)
<br>[3.11 处理时间序列](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.11-Working-with-Time-Series.ipynb)
<br>[3.12 高性能 Pandas：eval()与 query()](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/03.12-Performance-Eval-and-Query.ipynb)

## 深入研究 | MATPLOTLIB | 第 51 天

第 4 章：Matplotlib 数据可视化
<br>学习简易线形图, 简易散点图，密度图与等高线图.
<br>代码如下：
<br>[4 Matplotlib 数据可视化](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.00-Introduction-To-Matplotlib.ipynb)
<br>[4.1 简易线形图](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.01-Simple-Line-Plots.ipynb)
<br>[4.2 简易散点图](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.02-Simple-Scatter-Plots.ipynb)
<br>[4.3 可视化异常处理](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.03-Errorbars.ipynb)
<br>[4.4 密度图与等高线图](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.04-Density-and-Contour-Plots.ipynb)

## 深入研究 | MATPLOTLIB | 第 52 天

第 4 章：Matplotlib 数据可视化
<br>学习直方图，配置图例，配置颜色条，多子图。
<br>代码如下：
<br>[4.5 直方图](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.05-Histograms-and-Binnings.ipynb)
<br>[4.6 配置图例](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.06-Customizing-Legends.ipynb)
<br>[4.7 配置颜色条](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.07-Customizing-Colorbars.ipynb)
<br>[4.8 多子图](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.08-Multiple-Subplots.ipynb)
<br>[4.9 文字与注释](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.09-Text-and-Annotation.ipynb)

## 深入研究 | MATPLOTLIB | 第 53 天

第 4 章：Matplotlib 数据可视化
<br>学习三维绘图。
<br>[4.12 画三维图](https://github.com/jakevdp/PythonDataScienceHandbook/blob/main/notebooks/04.12-Three-Dimensional-Plotting.ipynb)

## 层次聚类 | 第 54 天

[动画演示](https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Other%20Docs/%E5%B1%82%E6%AC%A1%E8%81%9A%E7%B1%BB.gif)

<p align="center">
  <img src="https://github.com/bdfd/Machine_Learning_Road_To_Hero_Python/blob/main/Info-graphs/Day%2054.jpg">
</p>
