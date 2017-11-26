---
title: 解决matplotlib图例中文乱码问题
tags: 坑
categories: 技术
comments: true
date: 2017-11-24 23:27:37
---

### matplotlib图例中文乱码解决方案

**状况**：用`matplotlib`画图过程中，会用中文做图例(xlabel,title...)，会遇到中文乱码的问题（方格）。

<!--more-->

首先，下载中文字体`simhei.ttf`。

[下载地址](http://fontzone.net/download/simhei)

然后，找到`matplotlib`库画图所需的`font`文件的路径，在python编译器中键入：

```python
import matplotlib
matplotlib.matplotlib_fname()
```

获取`matplotlib`库存放`font`文件的路径，提供我的机器上面的路径做参考：

```
/home/enningxie/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf
```

最后，在程序中配置`matplotlib`画图参数：

```python
plt.rcParams['font.sas-serig']=['simhei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
```

以上，问题解决，图例中文显示正常。:)
