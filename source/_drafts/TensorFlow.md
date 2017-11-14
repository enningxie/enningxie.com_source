---
title: TensorFlow
tags: TensorFlow
categories: DeepLearning
comments: true
---

### TensorFlow 教程

TensorFlow是一个使用数据流图进行数值计算的开源软件库。图中的节点代表数学运算，而图中的边则代表在这些节点之间传递的多维数组（张量）。它可以用于进行机器学习和深度神经网络研究，但它是一个非常基础的系统，因此也可以应用于众多其他领域。

<!--more-->

#### TensorFlow的安装

[Linux](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/)（推荐）

[Windows](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/)

以上给出的外链是安装和配置keras所用的，但其实keras可以是以TensorFlow为后端的，所以说安装至TensorFlow完毕处即可，若想安装keras继续教程均可。

到目前，最新的TensorFlow版本为1.4。

#### TensorFlow的基础架构

##### 处理结构

**计算图纸**

TensorFlow首先是要定义神经网络的结构，然后向其中输入数据，进行运算和训练。

![](http://oslivcbny.bkt.clouddn.com/tensorflow_01.png)

TensorFlow中是采用数据流图进行计算的，所以我们先要创建一个数据流图(data flow graphs)，然后将我们的数据以张量的形式输入定义好的数据流图中去计算。数据流图中的节点（nodes）在图中表示数学操作，图中的线（edges）则表示节点之间相互联系的多维数据数组，即张量（tensor）。训练模型时tensor会不断从数据流图中的一个节点flow到另一个节点，这就是TensorFlow名字的由来。

**Tensor张量的意义**

张量（Tensor）:

- 张量有很多种。零阶张量为纯量或为标量（scalar）也就是一个数值。比如`1`

- 一阶张量为向量（vector），比如一维的`[1, 2, 3]`

- 二阶张量为矩阵（matrix），`[[1, 2, 3], [4, 5, 6]]`

等等。

##### 代码实例

记住，TensorFlow的工作是帮助你构建你所需要的神经网络的模型结构，然后接受你传入的tensor，进行处理运算。

**创建数据**

```Python
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3
```

接着, 我们用 `tf.Variable` 来创建描述 y 的参数. 我们可以把 `y_data = x_data*0.1 + 0.3` 想象成 `y=Weights * x + biases`, 然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3.

**搭建模型**

```Python
Weights = tf.Variable(tf.random_uniform([1], -1, 1.))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases
```

**计算误差**

计算真实值与预测值之间的误差loss

```Python
loss = tf.reduce_mean(tf.square(y-y_data))
```

**传播误差**

反向传递误差的工作就教给optimizer了, 我们使用的误差传递方法是梯度下降法: Gradient Descent 让后我们使用 optimizer 来进行参数的更新.

```Python
optimizer = tf.train.GradientDescentOptimizer(0.5)
train_op = optimizer.minimize(loss)
```

**训练**

到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构. 在使用这个结构之前, 我们必须先初始化所有之前定义的Variable, 所以这一步是很重要的!

```Python
init_op = tf.global_variables_initializer()
```

接着,我们再创建会话 Session. 我们会在下一节中详细讲解 Session. 我们用 Session 来执行 init_op 初始化步骤. 并且, 用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.

```Python
sess = tf.Session()
sess.run(init_op)

for step in range(201):
    sess.run(train_op)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
```

最终的结果：

```python
0 [-0.37600309] [ 0.87014645]
20 [-0.08786586] [ 0.40986526]
...
160 [ 0.09994658] [ 0.30003124]
180 [ 0.09998336] [ 0.30000973]
200 [ 0.09999482] [ 0.30000305]
```

经过我们定义的模型，最终 Weights收敛到`0.09999482`，biases收敛到`0.30000305`。符合预期。

##### Session 会话控制

TensorFlow中的`Session`用于控制和输出文件的执行语句。运行`session.run()`或者是`object_op.eval()`均可以获得目标的运算结果。

直接来看代码。

```Python
import tensorflow as tf

# create two matrixes
matrix1 = tf.constant([[3, 3]])  # shape (1, 2)
matrix2 = tf.constant([[2], [2]])  # shape (2, 1)

product_op = tf.matmul(matrix1, matrix2)
```

因为`product_op`不是直接计算的步骤，所以我们会用`Session`来激活`product`并得到计算结果，有两种方式使用`Session`:

```Python
# method 1
sess = tf.Session()
result = sess.run(product_op)
print(result)
sess.close()
```

```Python
# method 2
with tf.Session() as sess:
    result = sess.run(product_op)
    print(result)
```

以上两种使用`Session`的区别在于，第一种方法需要，手动关闭`sess.close()`，这种方式适合交互式编程环境，比如使用ipython的时候；第二种方法，则不需要手动关闭`Session`，这种方式适合写在某个文件中（系统编程）。

##### Variable 变量 
