---
title: Markdown
tags: Markdown
categories: 技术
comments: true
date: 2017-11-08 10:32:43
---


### Markdown

Markdown的目标是实现“易读易写”，且慢慢发展为一种适用于网络的书写语言。所以花点时间学习一下还是有必要的。

<!-- more -->

#### 1. 标题（Headers）

```
# H1
## H2
### H3
#### H4
##### H5
###### H6

对于H1和H2标题还可以这样写：

Alt-H1
======

Alt-H2
------
```

下面是上述语法的展示：

# H1
## H2
### H3
#### H4
##### H5
###### H6

H1和H2的另外写法展示：

Alt-H1
======
Alt-H2
------

---

#### 2. 强调（Emphasis）

```
斜体：　*斜体（星号）*　或者　_斜体（下划线）_
加粗：　**加粗（星号）**　或者　__加粗（下划线）__
混合：　**加粗_斜体_**
删除线：　~~删除线~~
```
斜体：　*斜体（星号）*　或者　_斜体（下划线）_

加粗：　**加粗（星号）**　或者　__加粗（下划线）__

混合：　**加粗_斜体_**

删除线：　~~删除线~~

---

#### 3. 列表（Lists）

在这个例子中，句首和句尾的空格用“%”表示。

```
1. 第一个有序的列表项
2. 第二个有序的列表项
%%* 无序的子列表
1. 第三个有序的列表项（前面的数字没用，因为有序）
%%1. 第一个有序的子列表
4. 第四个有序的列表项

%%%你可以正确的缩进列表项目中的段落，但是需要注意的是上面的一个空白行，和行首的空格（至少一个，我们这里用了三个）。

%%%在Markdown中实现换行，可以在行尾添加两个空格，然后在换行。%%
%%%注意，这里实现的换行是在一个段落中，平时我们换行直接敲两个回车即可。


* 无序列表项
- 无序列表项
+ 无序列表项
```

1. 第一个有序的列表项
2. 第二个有序的列表项
  * 无序的子列表项
1. 有序的列表项
  1. 第三个有序的列表项（前面的数字没用，因为有序）
4. 第四个有序的列表项

   你可以正确的缩进列表项目中的段落，但是需要注意的是上面的一个空白行，和行首的空格（至少一个，我们这里用了三个）。

   在Markdown中实现换行，可以在行尾添加两个空格，然后在换行。  
   注意，这里实现的换行是在一个段落中，平时我们换行直接敲两个回车即可。


* 无序列表项
- 无序列表项
+ 无序列表项

---

#### 4. 链接（links）

在Markdown中有两种方式实现链接。

```
[内嵌式链接](www.enningxie.com)

[带标题（title）的内嵌式链接](www.enningxie.com "Coolixz")

[参考式链接][Arbitrary case-insensitive reference text]

[存储文件的相对引用](../../test)

[可以用数字作为参考样式的链接定义][1]

或者可以直接用[link text itself]

上述参考链接的链接地址：

[Arbitrary case-insensitive reference text]: www.enningxie.com
[1]: www.enningxie.com
[link text itself]: www.enningxie.com

```

[内嵌式链接](www.enningxie.com)

[带标题（title）的内嵌式链接](www.enningxie.com "Coolixz")

[参考式链接][Arbitrary case-insensitive reference text]

[存储文件的相对引用](../../test)

[可以用数字作为参考样式的链接定义][1]

或者可以直接用[link text itself]

上述参考链接的链接地址：

[Arbitrary case-insensitive reference text]: www.enningxie.com
[1]: www.enningxie.com
[link text itself]: www.enningxie.com

---

#### 5. 图片（Images）

```
内嵌式：

![text](http://oslivcbny.bkt.clouddn.com/apple-touch-icon-next2.png)

参考式：

![text][logo]

[logo]: http://oslivcbny.bkt.clouddn.com/apple-touch-icon-next2.png
```

内嵌式：

![text](http://oslivcbny.bkt.clouddn.com/apple-touch-icon-next2.png)

参考式：

![text][logo]

[logo]: http://oslivcbny.bkt.clouddn.com/apple-touch-icon-next2.png

---

#### 6. 代码块（Code）

```
内嵌的代码块，通常用`code`。

一般代码块，用三个`表示，或者缩进４个空格。我觉得还是前一种方式简单。  
第一种还支持语法高亮。具体写法就是在代码块开头的三个`之后写明代码类型。
```

```javascript
var s = "JavaScript"
alert(s);
```

```python
s = "Python"
print("s")
```

```
没有指定代码类型的代码块
print("0-0");
```

---

#### 7. 表格（Tables）

```
冒号(:)可以用来对齐列

|表头１|表头２|表头３|
|:-----|:-----:|-----:|
|左对齐|居中|右对齐|
|sdasda|dasdad|dasdasd|
|a|b|c|

表格中至少三个(-)用于制表。　　
最外侧的(|)可以省略。　　
可以在表格内使用强调语法。

表头１|表头２|表头３
:-----|:-----:|-----:
左对齐|居中|右对齐
*斜体*|`代码块`|**加粗**
a|b|c
```

冒号(:)可以用来对齐列

|表头１|表头２|表头３|
|:---|:---:|---:|
|左对齐|居中|右对齐|
|sdasda|dasdad|dasdasd|
|a|b|c|

表格中至少三个(-)用于制表。　　
最外侧的(|)可以省略。　　
可以在表格内使用强调语法。

表头１|表头２|表头３
:-----|:-----:|-----:
左对齐|居中|右对齐
*斜体*|`代码块`|**加粗**
a|b|c

---

#### 8. 引用文字（Blockquotes）

```
> 引用文字用“>”标记。
> 这是同一个引用。

引用结束需要两个回车。

> 你可以加入强调语法在引用中，*斜体*，~~删除线~~
```

> 引用文字用“>”标记。
> 这是同一个引用。

引用结束需要两个回车。

> 你可以加入强调语法在引用中，*斜体*，~~删除线~~

---

#### 9. 内联HTML

你可以使用HTML在Markdown中，像这样：

```
<dl>
  <dt>Definition list</dt>
  <dd>Is something people use sometimes.</dd>

  <dt>Markdown in HTML</dt>
  <dd>Does *not* work **very** well. Use HTML <em>tags</em>.</dd>
</dl>
```

<dl>
  <dt>Definition list</dt>
  <dd>Is something people use sometimes.</dd>

  <dt>Markdown in HTML</dt>
  <dd>Does *not* work **very** well. Use HTML <em>tags</em>.</dd>
</dl>

#### 10. 水平分隔线

```
Markdown中的水平分隔线有三种：

---

连字符

***

星号

___

下划线
```

Markdown中的水平分隔线有三种：

---

连字符

***

星号

___

下划线

Markdown最基本的用法就是这些。

---

#### 11. Markdown编辑器

目前我在用的Markdown的编辑器是GitHub出品的[Atom](https://atom.io).

![Atom_01](http://oslivcbny.bkt.clouddn.com/Atom01.png)

Atom可以实现一边编辑`.md`文件一边预览的效果。

`ctrl+shift+m`快捷键用于呼出当前`.md`文件的预览界面。

![Atom_02](http://oslivcbny.bkt.clouddn.com/Atom_02.png)

最后，玩的开心:)
