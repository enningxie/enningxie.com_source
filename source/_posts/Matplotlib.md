---
title: matplotlib
tags: matplotlib
categories: Python
comments: true
date: 2017-11-14 10:17:58
---


#### Matplotlib 包

##### Matplotlib 简介

Matplotlib 是一个非常强大的Python　画图工具；

<!--more-->

它能够帮你画出美丽的：

- 线图；
- 散点图；
- 等高线图；
- 条形图；
- 柱状图；
- 3D图形；

下面是一些例图：

![](http://oslivcbny.bkt.clouddn.com/matplotlib_01.png)

![](http://oslivcbny.bkt.clouddn.com/matplotlib_02.png)

![](http://oslivcbny.bkt.clouddn.com/matplotlib_03.png)


##### Matplotlib 安装

```
$ sudo apt-get install python3-matplotlib
```

```
pip install matplotlib
```

安装之前确保你已经安装了`numpy`。

##### 基本用法

一个简单的例子:

导入模块

```Python
import matplotlib.pyplot as plt
import numpy as np
```

制造数据：

```Python
x = np.linspace(-1, 1, 50)
y = 2*x+1
```

使用`plt.figure`定义一个图像窗口，使用`plt.plot`画图，使用`plt.show`显示图像。

```Python
plt.figure()
plt.plot(x, y)
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_1.png)

##### figure 图像

```Python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x+1
y2 = x**2

plt.figure()
plt.plot(x, y1)
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_1.png)

使用`plt.figure`定义一个图像窗口：编号为3(num)；大小为(8,5)(figsize)。使用`plt.plot`画(x, y2)曲线.使用`plt.plot`画(x, y1)曲线，曲线的颜色属性（color）为红色；曲线的宽度（linewidth）为1.0；曲线的类型（linestyle）为虚线。使用`plt.show`显示图像。

```Python
plt.figure(num=3, figsize=(8, 5))
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_3.png)

##### 设置坐标间隔和名称

```Python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x+1
y2 = x**2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
```

使用`plt.xlim`设置x坐标轴范围：(-1, 2);使用`plt.ylim`设置y坐标轴范围：(-2, 3)；使用`plt.xlabel`设置x坐标轴名称；使用`plt.ylabel`设置y坐标轴名称；

```Python
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel("i am x")
plt.ylabel("i am y")
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_2.png)

使用`np.linspace`定义范围以及个数：范围是(-1,2);个数是5. 使用print打印出新定义的范围. 使用`plt.xticks`设置x轴刻度：范围是(-1,2);个数是5.

```Python
new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)
```

使用`plt.yticks`设置y轴刻度以及名称：刻度为[-2, -1.8, -1, 1.22, 3]；对应刻度的名称为[‘really bad’,’bad’,’normal’,’good’, ‘really good’]. 使用`plt.show`显示图像.

```Python
plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_4.png)

##### 设置不同名字和位置

```Python
# coding=utf-8
# 设置不同名字和位置

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x+1
y2 = x**2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.xlim((-1, 2))  # 设置x轴范围
plt.ylim((-2, 3))  # 设置y轴范围

# 设置新的刻度
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3],['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])
```

使用`plt.gca`获取当前坐标轴信息。使用`.spines`设置边框：使用`.set_color`设置边框颜色：默认白色； 使用`.spines`设置边框：上边框；使用`.set_color`设置边框颜色：默认白色

![](http://oslivcbny.bkt.clouddn.com/Figure_5.png)

##### 调整坐标轴

使用`.xaxis.set_ticks_position`设置x坐标刻度数字或名称的位置：`bottom`.（所有位置：top，bottom，both，default，none）

```Python
ax.xaxis.set_ticks_position('bottom')
```

使用`.spines`设置边框：x轴；使用`.set_position`设置边框位置：y=0的位置；（位置所有属性：outward，axes，data）

```Python
ax.spines['bottom'].set_position(('data', 0))
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_6.png)

使用`.yaxis.set_ticks_position`设置y坐标刻度数字或名称的位置：`left`.（所有位置：left，right，both，default，none）

```Python
ax.yaxis.set_ticks_position('left')
```

使用`.spines`设置边框：y轴；使用`.set_position`设置边框位置：x=0的位置；（位置所有属性：outward，axes，data） 使用`plt.show`显示图像.

```Python
ax.spines['left'].set_position(('data', 0))
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_7.png)

##### Legend　图例

**添加图例**

matplotlib中的`legend`图例就是为了帮我们展示出每个数据对应的图像名称。更好的让读者认识到你的数据结构。

```Python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
#set x limits
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# set new sticks
new_sticks = np.linspace(-1, 2, 5)
plt.xticks(new_sticks)
# set tick labels
plt.yticks([-2, -1.8, -1, 1.22, 3],
           [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
```

本节中我们将对图中的两条线绘制图例，首先我们设置两条线的类型等信息（蓝色实线与红色虚线）。

```Python
# set line styles
l1, = plt.plot(x, y1, label="linear line")
l2, = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
```

`legend`将要显示的信息来自于上面代码中的`label`。所以我们要简单写一下代码，plt就能自动的为我们添加图例了。

```Python
plt.legend(loc='upper right')
```

参数`loc='upper right'`表示图例将添加在图中的右上角，ps：`loc='best'`表示，自动将图例放置在合适的地方。

![](http://oslivcbny.bkt.clouddn.com/Figure_8.png)

##### 调整位置和名称

如果我们想单独修改之前的 label 信息, 给不同类型的线条设置图例信息. 我们可以在 `plt.legend` 输入更多参数. 如果以下面这种形式添加 legend, 我们需要确保, 在上面的代码 `plt.plot(x, y2, label='linear line')` 和 `plt.plot(x, y1, label='square line')` 中有用变量 l1 和 l2 分别存储起来. 而且需要注意的是 l1, l2,要以逗号结尾, 因为`plt.plot()` 返回的是一个列表.

```Python
plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best')
```

这样我们就能分别重新设置线条对应的`label`了。

![](http://oslivcbny.bkt.clouddn.com/Figure_9.png)

其实`‘loc’`参数有很多种，'best'表示自动分配最佳的位置，其余如下：

```Python
'best' : 0,          
'upper right'  : 1,
'upper left'   : 2,
'lower left'   : 3,
'lower right'  : 4,
'right'        : 5,
'center left'  : 6,
'center right' : 7,
'lower center' : 8,
'upper center' : 9,
'center'       : 10,
```

##### Annotation标注

当图线中某些特殊地方需要标注时，我们可以使用 `annotation`. matplotlib 中的 `annotation` 有两种方法， 一种是用 plt 里面的 `annotate`，一种是直接用 plt 里面的 `text` 来写标注.

首先，我们先在坐标轴中绘制一条直线。

```Python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 2*x+1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y)
```

![](http://oslivcbny.bkt.clouddn.com/Figure_10.png)

然后我们移动坐标

```Python
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
```

![](http://oslivcbny.bkt.clouddn.com/Figure_11.png)

然后标注出点`(x0, y0)`的位置信息. 用`plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)` 画出一条垂直于x轴的虚线.

```Python
x0 = 1
y0 = 2*x0 + 1
plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
# set dot styles
plt.scatter([x0, ], [y0, ], s=50, color='b')
```

![](http://oslivcbny.bkt.clouddn.com/Figure_12.png)

##### 添加注释 annotate

接下来我们就对`(x0, y0)`这个点进行标注

```Python
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
```

其中参数`xycoords='data'` 是说基于数据的值来选位置, `xytext=(+30, -30)` 和 `textcoords='offset points'` 对于标注位置的描述 和 xy 偏差值, `arrowprops`是对图中箭头类型的一些设置.

![](http://oslivcbny.bkt.clouddn.com/Figure_13.png)

##### 添加注释 text

```Python
plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})
```

其中-3.7, 3,是选取text的位置, 空格需要用到转字符`\` ,`fontdict`设置文本字体.

![](http://oslivcbny.bkt.clouddn.com/Figure_14.png)

##### tick 能见度

当图片中的内容较多，相互遮盖时，我们可以通过设置相关内容的透明度来使图片更易于观察，也即是通过本节中的`bbox`参数设置来调节图像信息.

首先，绘制图像：

```Python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 0.1*x

plt.figure()
plt.plot(x, y, linewidth=10)
plt.ylim(-2, 2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
```

![](http://oslivcbny.bkt.clouddn.com/Figure_15.png)

调整坐标

然后对被遮挡的图像调节相关透明度，本例中设置 x轴 和 y轴 的刻度数字进行透明度设置

```Python
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
plt.show()
```

其中`label.set_fontsize(12)`重新调节字体大小，`bbox`设置目的内容的透明度相关参，`facecolor`调节 `box` 前景色，`edgecolor` 设置边框， 本处设置边框为无，`alpha`设置透明度. 最终结果如下:

![](http://oslivcbny.bkt.clouddn.com/Figure_16.png)

#### 画图种类

##### Scatter 散点图

本节我们将讲述各种不同的plot的方式。之前我们讲到了如何plot线，现在讲述我们如何plot散点图。该节用到的例子最终呈现的结果如下图：

![](http://oslivcbny.bkt.clouddn.com/Figure_17.png)

**散点图**

首先，引入`matplotlib.pyplot`简写为`plt`，再引入`numpy`用来产生一些随机数据。生成1024个呈标准正态分布的二维数据组（均值为0，方差为1）作为一个数据集合，并图像化这个数据集合。没一个点的颜色值用`T`来表示：

```python
import matplotlib.pyplot as plt
import numpy as np

n = 1024  # data size
X = np.random.normal(0, 1, n)  # 每一个点的X的值
Y = np.random.normal(0, 1, n)  # 每一个点的Y的值
T = np.arctan2(Y, X)  # for color value
```

数据集生成完毕，现在用`scatter`去plot这个点集，该函数输入参数一般：输入`X`和`Y`作为location,`size=75`,颜色为`T`, `color map`用默认值，透明度`alpha`为50%。X轴显示范围定位(-1.5, 1.5)，并用`xtick()`函数来隐藏x坐标轴，y轴同理：

```python
plt.figure()

plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()
```

##### Bar 柱状图

本节介绍用`matplotlib`来制作一个柱状图，今天的结果如下图：

![](http://oslivcbny.bkt.clouddn.com/Figure_18.png)

今天的柱状图分成上下两部分，每一个柱体上都有相应的数值标注，并且取消坐标轴的显示。

**生成基本图形**

向上向下分别生成12个数据，X为0到11的整数，Y是相应的均匀分布的随机数据。使用的函数是`plt.bar`，参数为X，Y：

```python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1-X/float(n))*np.random.uniform(0.5, 1.0, n)
Y2 = (1-X/float(n))*np.random.uniform(0.5, 1.0, n)

plt.figure()
plt.bar(X, +Y1)
plt.bar(X, -Y2)

plt.xlim(-5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()
```

这样我们就生成了下图所示的柱状图基本框架：

![](http://oslivcbny.bkt.clouddn.com/Figure_19.png)

**加颜色和数据**

下面我们就颜色和数值进行优化。用`facecolor`设置主体颜色，`edgecolor`设置边框颜色为白色

```python
plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
```

现在呈现的结果为：

![](http://oslivcbny.bkt.clouddn.com/Figure_20.png)

接下来我们用函数`plt.text`分别在柱体上方（下方）加上数值，用`%.2f`保留两位小数，横向居中对齐`ha='center'`，纵向底部（顶部）对齐`va='bottom'`：

```python
for x, y in zip(X, Y1):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')
```

最终结果就像开始一样：

![](http://oslivcbny.bkt.clouddn.com/Figure_18.png)

##### Contours等高线图

本节讲解生成等高线图，最终结果如下所示：

![](http://oslivcbny.bkt.clouddn.com/Figure_21.png)

**画等高线**

数据集即三维点(x, y)和对应的高度值，共有256个点。高度值使用一个height function`f(x,y)`生成。x, y分别是在区间[-3, 3]中均匀分布的256个值，并`meshgrid`在二维平面中将每一个x和每一个y分别对应起来，编织成栅格：

```python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    # the height function
    return (1 - x/2 + x**5 + y**3) * np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
```

接下来进行颜色填充。使用函数`plt.contourf`把颜色加进去，位置参数分别为：X, Y, f(X,Y)。透明度0.75，并将 f(X,Y) 的值对应到color map的暖色组中寻找对应颜色。

```python
# use plt.contourf to filling contours
# X, Y and value for (X,Y) point
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
```

接下来进行等高线绘制。使用`plt.contour`函数划线。位置参数为：X, Y, f(X,Y)。颜色选黑色，线条宽度选0.5。现在的结果如下图所示，只有颜色和线条，还没有数值Label：

```python
# use plt.contour to add contour lines
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
```

![](http://oslivcbny.bkt.clouddn.com/Figure_22.png)

**添加高度数字**

其中，8代表等高线的密集程度，这里被分为10个部分。如果是0，则图像被一分为二。

最后加入Label，inline控制是否将Label画在线里面，字体大小为10。并将坐标轴隐藏：

```Python
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
```

![](http://oslivcbny.bkt.clouddn.com/Figure_21.png)

##### Image 图片

**随机矩阵画图**

这一节讲解怎样在matplotlib中打印出图像，这里我们打印出的是纯粹的数字，而非自然图像。我们在这里用这样3*3的2D-array来表示点的颜色，每一个点就是一个pixel。

```Python
import matplotlib.pyplot as plt
import numpy as np

a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)
```

最终的成果图如下所示：

![](http://oslivcbny.bkt.clouddn.com/Figure_23.png)

三行三列的格子，a代表每一个值，图像右边有一个注释，白色代表值最大的地方，颜色越深值越小。

下面我们来看代码：

```python
plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
```

我们之前选cmap的参数时用的是：`cmap=plt.cmap.bone`，而现在，我们可以直接用单引号传入参数。 `origin='lower'`代表的就是选择的原点的位置。

**出图方式**

![](http://oslivcbny.bkt.clouddn.com/Figure_24.png)

这里我们使用的是内插法中的 `Nearest-neighbor` 的方法，其他的方式也都可以随意取选。

**colorbar**

下面我们添加一个colorbar ，其中我们添加一个shrink参数，使colorbar的长度变短为原来的92%：

```python
plt.colorbar(shrink=.92)

plt.xticks(())
plt.yticks(())
plt.show()
```

这样我们２D图像就创建完成了。

![](http://oslivcbny.bkt.clouddn.com/Figure_23.png)

##### 3D数据

**3D图**

首先在进行 `3D Plot` 时除了导入 `matplotlib` ，还要额外添加一个模块，即 `Axes 3D` 3D 坐标轴显示：

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

之后要先定义一个图像窗口，在窗口上添加3D坐标轴，显示成下图：

```python
fig = plt.figure()
ax = Axes3D(fig)
```

![](http://oslivcbny.bkt.clouddn.com/Figure_25.png)

接下来给进 X 和 Y 值，并将 X 和 Y 编织成栅格。每一个（X, Y）点对应的高度值我们用下面这个函数来计算。

```python
# X, Y value
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)
```

本节的结果：

![](http://oslivcbny.bkt.clouddn.com/Figure_26.png)

做出一个三维曲面，并将一个 `colormap rainbow` 填充颜色，之后将三维图像投影到 XY 平面上做一个等高线图。 plot 3D 图像：

```python
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
```

其中，`rstride` 和 `cstride` 分别代表 row 和 column 的跨度。

下面两个图分别是跨度为1 和 5 的效果：

![](http://oslivcbny.bkt.clouddn.com/Figure_27.png)

![](http://oslivcbny.bkt.clouddn.com/Figure_28.png)

**投影**

下面添加 XY 平面的等高线：

```python
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
```

如果 zdir 选择了x，那么效果将会是对于 XZ 平面的投影，效果如下：

![](http://oslivcbny.bkt.clouddn.com/Figure_29.png)

最终我们得到以下的图：

![](http://oslivcbny.bkt.clouddn.com/Figure_26.png)

#### 多图合并并显示

##### Subplot多合一显示

matplotlib是可以组合许多的小图，放在一张大图里面显示的。使用到的方法叫做Subplot

使用import导入`matplotlib.pyplot`模块, 并简写成plt. 使用`plt.figure`创建一个图像窗口.

```python
import matplotlib.pyplot as plt

plt.figure()
```

使用`plt.subplot`来创建小图. `plt.subplot(2,2,1)`表示将整个图像窗口分为2行2列, 当前位置为1. 使用`plt.plot([0,1],[0,1])`在第1个位置创建一个小图.

```python
plt.subplot(2,2,1)
plt.plot([0,1],[0,1])
```

`plt.subplot(2,2,2)`表示将整个图像窗口分为2行2列, 当前位置为2. 使用`plt.plot([0,1],[0,2])`在第2个位置创建一个小图.

```python
plt.subplot(2,2,2)
plt.plot([0,1],[0,2])
```

`plt.subplot(2,2,3)`表示将整个图像窗口分为2行2列,当前位置为3. `plt.subplot(2,2,3)`可以简写成`plt.subplot(223)`, matplotlib同样可以识别. 使用`plt.plot([0,1],[0,3])`在第3个位置创建一个小图.

```python
plt.subplot(223)
plt.plot([0,1],[0,3])
```

`plt.subplot(224)`表示将整个图像窗口分为2行2列, 当前位置为4. 使用`plt.plot([0,1],[0,4])`在第4个位置创建一个小图.

```python
plt.subplot(224)
plt.plot([0,1],[0,4])

plt.show()  # 展示
```

![](http://oslivcbny.bkt.clouddn.com/Figure_30.png)

**不均匀图中图**

如果希望展示的小图的大小不相同, 应该怎么做呢? 以上面的4个小图为例, 如果把第1个小图放到第一行, 而剩下的3个小图都放到第二行.

使用`plt.subplot(2,1,1)`将整个图像窗口分为2行1列, 当前位置为1. 使用`plt.plot([0,1],[0,1])`在第1个位置创建一个小图.

```python
plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
```

使用`plt.subplot(2,3,4)`将整个图像窗口分为2行3列, 当前位置为4. 使用`plt.plot([0,1],[0,2])`在第4个位置创建一个小图.

```python
plt.subplot(2,3,4)
plt.plot([0,1],[0,2])
```

这里需要解释一下为什么第4个位置放第2个小图. 上一步中使用`plt.subplot(2,1,1)`将整个图像窗口分为2行1列, 第1个小图占用了第1个位置, 也就是整个第1行. 这一步中使用`plt.subplot(2,3,4)`将整个图像窗口分为2行3列, 于是整个图像窗口的第1行就变成了3列, 也就是成了3个位置, 于是第2行的第1个位置是整个图像窗口的第4个位置.

使用`plt.subplot(235)`将整个图像窗口分为2行3列,当前位置为5. 使用`plt.plot([0,1],[0,3])`在第5个位置创建一个小图. 同上, 再创建`plt.subplot(236)`.

```python
plt.subplot(235)
plt.plot([0,1],[0,3])

plt.subplot(236)
plt.plot([0,1],[0,4])

plt.show()  # 展示
```

![](http://oslivcbny.bkt.clouddn.com/Figure_31.png)

##### Subplot分格显示

matplotlib 的subplot还可以是分格的，这里介绍三种方法

**subplot2grid**

使用`import` 导入`matplotlib.pyplot`模块，并简写成`plt`。使用`plt.figure()`创建一个图像窗口

```python
import matplotlib as plt
plt.figure()
```

使用`plt.subplot2grid`来创建第1个小图, (3,3)表示将整个图像窗口分成3行3列, (0,0)表示从第0行第0列开始作图，colspan=3表示列的跨度为3, rowspan=1表示行的跨度为1. colspan和rowspan缺省, 默认跨度为1

```python
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax1.plot([1, 2], [1, 2])  # 画小图
ax1.set_title("ax1_title")  # 设置小图的标题
```

使用`plt.subplot2grid`来创建第2个小图, (3,3)表示将整个图像窗口分成3行3列, (1,0)表示从第1行第0列开始作图，colspan=2表示列的跨度为2. 同上画出 ax3, (1,2)表示从第1行第2列开始作图，rowspan=2表示行的跨度为2. 再画一个 ax4 和 ax5, 使用默认 colspan, rowspan.

```python
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))
```

使用`ax4.scatter`创建一个散点图, 使用ax4.set_xlabel和ax4.set_ylabel来对x轴和y轴命名.

```python
ax4.scatter([1, 2], [2, 2])
ax4.set_xlabel('ax4_x')
ax4.set_ylabel('ax4_y')
```

![](http://oslivcbny.bkt.clouddn.com/Figure_32.png)

**gridspec**

使用import导入`matplotlib.pyplot`模块, 并简写成plt. 使用import导入`matplotlib.gridspec`, 并简写成gridspec.

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
```

使用`plt.figure()`创建一个图像窗口, 使用`gridspec.GridSpec`将整个图像窗口分成3行3列.

```python
plt.figure()
gs = gridspec.GridSpec(3, 3)
```

使用plt.subplot来作图, gs[0, :]表示这个图占第0行和所有列, gs[1, :2]表示这个图占第1行和第2列前的所有列, gs[1:, 2]表示这个图占第1行后的所有行和第2列, gs[-1, 0]表示这个图占倒数第1行和第0列, gs[-1, -2]表示这个图占倒数第1行和倒数第2列.

```python
ax6 = plt.subplot(gs[0, :])
ax7 = plt.subplot(gs[1, :2])
ax8 = plt.subplot(gs[1:, 2])
ax9 = plt.subplot(gs[-1, 0])
ax10 = plt.subplot(gs[-1, -2])
```

![](http://oslivcbny.bkt.clouddn.com/Figure_33.png)

**subplots**

使用`plt.subplots`建立一个2行2列的图像窗口，`sharex=True`表示共享x轴坐标, `sharey=True`表示共享y轴坐标. ((ax11, ax12), (ax13, ax14))表示第1行从左至右依次放ax11和ax12, 第2行从左至右依次放ax13和ax14.

```python
f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=True, sharey=True)
```

使用ax11.scatter创建一个散点图.

```python
ax11.scatter([1,2], [1,2])
```

`plt.tight_layout()`表示紧凑显示图像, `plt.show()`表示显示图像.

![](http://oslivcbny.bkt.clouddn.com/Figure_34.png)

##### 图中图

这次我们来讲matplotlib里一个很有意思的功能，叫做图中图(plot in plot)，最后的效果如下：

![](http://oslivcbny.bkt.clouddn.com/Figure_38.png)

可以看到，整个Figure 1包含了三个图，其中两个小图title inside 1和title inside 2又出现在大图title中。这是如何做到的呢？

**数据**

首先是一些准备工作：

```python
# 导入pyplot模块
import matplotlib.pyplot as plt

# 初始化figure
fig = plt.figure()

# 创建数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]
```

**大图**

接下来我们来绘制大图。首先是确定大图左下角的位置以及宽高：

```python
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
```

注意，4个值都是占整个figure坐标系的百分比。在这里，假设figure的大小是10x10，那么大图就被包含在由(1, 1)开始，宽8，高8的坐标系内。

将大图坐标系添加到figure中，颜色为r(red)，取名为title：

```python
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')
```

效果如下：

![](http://oslivcbny.bkt.clouddn.com/Figure_36.png)

**小图**

接着，我们来绘制左上角的小图，步骤和绘制大图一样，注意坐标系位置和大小的改变:

```python
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title inside 1')
```

效果如下：

![](http://oslivcbny.bkt.clouddn.com/Figure_37.png)

最后，我们来绘制右下角的小图。这里我们采用一种更简单方法，即直接往plt里添加新的坐标系：

```python
plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::-1], x, 'g') # 注意对y进行了逆序处理
plt.xlabel('x')
plt.ylabel('y')
plt.title('title inside 2')
```

最后显示图像：

```python
plt.show()
```

最后效果如下：

![](http://oslivcbny.bkt.clouddn.com/Figure_38.png)

##### 次坐标轴

**第一个y坐标**

有时候我们会用到次坐标轴，即在同个图上有第2个y轴存在。同样可以用matplotlib做到，而且很简单.

首先我们做一些准备工作;

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)

y1 = 0.05 * x**2

y2 = -1 * y1
```

可以看到，y2和y1是互相倒置的。接着，获取figure默认的坐标系 ax1：

```python
fig, ax1 = plt.subplots()
```

**第二个y坐标**

对ax1调用`twinx()`方法，生成如同镜面效果后的ax2：

```python
ax2 = ax1.twinx()
```

接着进行绘图, 将 y1, y2 分别画在 ax1, ax2 上：

```python
ax1.plot(x, y1, 'g-')   # green, solid line

ax1.set_xlabel('X data')

ax1.set_ylabel('Y1 data', color='g')

ax2.plot(x, y2, 'b-') # blue

ax2.set_ylabel('Y2 data', color='b')
```

显示图像：

```python
plt.show()
```

![](http://oslivcbny.bkt.clouddn.com/Figure_39.png)

最后，matplotlib还可以做动画，相关内容请自行google。

:)
