---
title: matplotlib
tags: matplotlib
categories: Python
comments: true
---

#### matplotlib 包

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
