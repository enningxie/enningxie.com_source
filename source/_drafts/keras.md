---
title: keras
tags: keras
categories: DeepLearning
comments: true
---

### Keras:基于Python的深度学习库

首先是对keras的一个简单的介绍，来自[官方文档](https://keras-cn.readthedocs.io)。

> Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow、Theano以及CNTK后端。Keras 为支持快速实验而生，能够把你的idea迅速转换为结果。

<!--more-->

当前的版本号是2.0.9，对应于官方的2.0.9 release版本。

---

#### 安装

关于keras的安装参考：

- [Linux](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/)

- [Windows](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/)

---

#### 快速上手

Keras的核心数据结构是“模型”，模型是一种组织网络层的方式。keras中主要的模型是Sequential模型，Sequential是一系列网络层按顺序构成的栈。

首先导入Sequential模型：

```Python
from keras.models import Sequential
model = Sequential()
```

将一些网络层通过`.add()`堆叠起来，用于构建模型：

```Python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
```

完成模型的搭建后，我们需要使用`.compile()`方法来编译模型：

```Python
model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
```

从上述编译模型的代码中，我们可以看到，我们需要指定损失函数（loss）和优化方法（optimizer）。当然如果需要的话，你也可以自己定制loss函数。

keras的一个核心的理念就是简明易用，保证了用户对keras的绝对的控制力度，用户可以根据自己的需要定制自己的模型、网络层，甚至修改源代码。

定制：

```Python
from keras.optimizers import SGD
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

在对模型完成编译后，我们在训练数据上按batch进行一定次数的迭代来训练模型：

```Python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

当然我们也可以手动将一个个batch的数据送入网络中训练：

```Python
model.train_on_batch(x_batch, y_batch)
```

最后，我们用一行代码对我们的模型进行评估，用于查看模型的指标是否满足要求：

```Python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

或者，我们可以用我们的模型，对新的数据进行预测：

```Python
classes = model.predict(x_test, batch_size=128)
```

以上就是一个模型的搭建过程。

keras默认使用Tensorflow作为后端来进行张量操作的。当然也可以切换成Theano，在这不做拓展。

在[keras代码包](https://github.com/fchollet/keras)的examples文件夹下，我们提供了一些更高级的模型，可以去看看。

#### 初步了解
