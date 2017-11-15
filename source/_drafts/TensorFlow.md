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

**简单运用**

 在TensorFlow中，定义了某字符串是变量，它才是变量，定义的语法：`state = tf.Variable()`

 ```Python
import tensorflow as tf

state = tf.Variable(0, name="counter")

# 定义常量one
one = tf.constant(1)

# 定义加法步骤
add_op = tf.add(state, one)

# 将state 更新成add_op
update = tf.assign(state, add_op)
 ```

 如果你在TensorFlow中设定了变量，那么初始化变量是相当重要的，所以定义了变量后，一定要记得定义变量初始化操作哦，`init_op = tf.global_variables_initializer()`。

 光定义变量初始化操作还不行，还需要在`tf.Session()`中`sess.run(init_op)`，才算变量初始化成功。

 ```Python
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
 ```

注意：直接调用`sess.run(state)`不起作用，需要先执行update操作才可以。

##### Placeholder 传入值

关于TensorFlow中的`placeholder`，类似占位符的一个概念，可以用来暂时储存变量。

在TensorFlow中，如果想要从外部传入data，那就需要用到`tf.placeholder()`，然后以`sess.run(****, feed_dict={input: ****})`的形式向运算中传入数据data。

```Python
import tensorflow as tf
# 在tensorflow 中需要定义placeholder的type，一般为tf.float32
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 乘法操作
mul_op = tf.multiply(input1, input2)
```

在定义好操作的结构之后，我们就可以在`tf.Session()`中去`run()`了。将需要传入进行运算的值放在`feed_dict={}`中（传入的值以字典的形式传入，一一对应）。

```Python
with tf.Session() as sess:
    result = sess.run(mul_op, feed_dict={input1: 3, input2: 4})
    print(result)
```

##### 代码实例

**定义add_layer()**

```Python
import tensorflow as tf


def add_layer(input, input_size, output_size, activation_func=None):
```

定义了`add_layer()`函数，用于向神经网络中添加层；参数分别是输入值、输入值的大小、输出值的大小、激活函数，默认是`None`。

接下来，我们要定义`Weights`和`biases`，并初始化它们。

```Python
Weights = tf.Variable(tf.random_normal([input_size, output_size]))
biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
```

从上面代码，我们可以看出，权重`Weights`的初始化是随机正态分布初始化的，注意一下初始化时传入的shape`[input_size, output_size]`。`biases`本来是用全零初始化的，但是在深度学习领域，最好不要用全零初始化，所以最后加上`0.1`。

```Python
Wx_plus_b = tf.matmul(input, Weights) + biases
if activation_func is None:
    output = Wx_plus_b
else:
    output = activation_func(Wx_plus_b)
return output
```

`Wx_plus_b`是简单的矩阵运算。在`activation_func`为`None`的情况下，我们的`output`就是`Wx_plus_b`，否则，将`Wx_plus_b`值传入`activation_func`中，返回最后的结果。
