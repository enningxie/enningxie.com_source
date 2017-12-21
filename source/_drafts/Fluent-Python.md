---
title: Fluent Python
tags: Python
categories: 技术
comments: true
---

## 流畅的python

读书记录。

Python是一门既容易上手又强大的编程语言。

<!--more-->

### 第一章：Python数据模型

Python解释器碰到特殊的句法时，会使用**特殊方法**去激活一些基本的对象操作，这些特殊方法的名字以两个下划线开头，以两个下划线结尾（如`__getitem__`）。如`obj[key]`的背后就是`__getitem__`方法。

接下来的例子展示了如何实现`__getitem__`和`__len__`这两个特殊方法。

```python
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for rank in self.ranks for suit in self.suits]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
```

`collections.namedtuple`构建了一个简单的类来表示一张纸牌。`namedtuple`用来构建只有少数属性但是没有方法的对象，比如数据库条目。

我们可以通过`my_card = Card('4', 'spades')`轻松得到一个纸牌对象。

这里我们重点关注的是`FrenchDeck`类，可以通过

```python
>>> deck = FrenchDeck()
>>> len(deck)
```
查看一副牌的张数。

抽牌操作是通过`__getitem__`方法提供的，可以通过`deck[0]`调用。

`random`包中提供了从一个序列中随机选出一个元素的函数`random.choice`，２２

`__repr__`所返回的字符串应该准确、无歧义，并且尽可能表达出如何用代码创建出这个被打印的对象。

`__repr__`和`__str__`的区别在于，后者是`str()`函数被使用，或是在用`print`函数打印一个对象的时候才被调用的，并且返回的字符串对终端用户友好。简单而言，前者方便我们调试和记录日志，后者则是给终端用户看的。

`list.sort`方法会就地排序列表，也就是说不会把原列表复制一份。

Python的一个惯例：如果一个函数或者方法对对象进行的是就地改动，那它就应该返回None，好让调用者知道传入的参数发生了变动，而且并未产生新的对象。

与`list.sort`相反的是内置函数`sorted`，它会新建一个列表作为返回值。

`collections.deque`类（双向队列）是一个线程安全、可以快速从两端添加或删除元素的数据类型。

`sorted`和`list.sort`背后的排序算法是`Timsort`，它是一种自适应算法，会根据原始数据的特点交替使用插入排序和归并排序，以达到最佳效率。
