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

#### 建造我们的神经网络

##### 添加层

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


##### 建造神经网络

在上一步的基础上，首先来导入数据。

```Python
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# print(x_data.shape)
noise = np.random.normal(0, 0.005, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5, + noise
```

构建所需的数据。 这里的x_data和y_data并不是严格的一元二次函数的关系，因为我们多加了一个noise,这样看起来会更像真实情况。

```Python
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
```

利用占位符定义我们所需的神经网络的输入。 `tf.placeholder()`就是代表占位符，这里的`None`代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1。

接下来，我们就可以开始定义神经层了。 通常神经层都包括输入层、隐藏层和输出层。这里的输入层只有一个属性， 所以我们就只有一个输入；隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元； 输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。 所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。

**搭建网络**

下面，我们开始定义隐藏层,利用之前的`add_layer()`函数，这里使用 Tensorflow 自带的激励函数`tf.nn.relu`。

```Python
l1 = add_layer(xs, 1, 10, activation_func=tf.nn.relu)
```

接着，定义输出层。此时的输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层。

```Python
prediction = add_layer(l1, 10, 1, activation_func=None)
```

计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。

```Python
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
```

接下来，是很关键的一步，如何让机器学习提升它的准确率。定义优化方法`tf.train.GradientDescentOptimizer()`，　其中的值通常都小于1，这里取的是0.1，代表以0.1的效率来最小化误差loss。

```Python
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```

使用变量时，需要初始化变量操作：

```Python
init_op = tf.global_variables_initializer()
```

最后在`tf.Session()`中运行就好了：

```Python
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(1000):
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: x_data, ys: y_data})
        if (step+1) % 50 == 0 or step == 0:
            print("Step: ", step+1, " loss: {:.4f}".format(loss_))
```

##### matplotlib 可视化

构建图形，用散点图描述真实数据之间的关系。（`plt.ion()`用于连续显示）

```Python
# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()
```

散点图的结果显示为：

![](http://oslivcbny.bkt.clouddn.com/tensorflow_02.png)

接下来，我们来显示预测数据。

每隔50次训练刷新一次图形，用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.1s。

```Python
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(1000):
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: x_data, ys: y_data})
        if (step+1) % 50 == 0 or step == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_ = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
            lines = ax.plot(x_data, prediction_, 'r-', lw=1)
            plt.pause(0.1)
            print("Step: ", step+1, " loss: {:.4f}".format(loss_))
```

最后，可视化结果为：

![](http://oslivcbny.bkt.clouddn.com/tensorflow_03.png)

##### 加速神经网络训练

加速神经网络训练一般会涉及到:

- Stochastic Gradient Descent (SGD)

- Momentum

- AdaGrad

- RMSProp

- Adam

**Stochastic Gradient Descent (SGD)**

![](http://oslivcbny.bkt.clouddn.com/tensorflow_04.png)

最基础的方法就是 SGD 啦, 想像红色方块是我们要训练的 data, 如果用普通的训练方法, 就需要重复不断的把整套数据放入神经网络 NN训练, 这样消耗的计算资源会很大.

我们换一种思路, 如果把这些数据拆分成小批小批的, 然后再分批不断放入 NN 中计算, 这就是我们常说的 SGD 的正确打开方式了. 每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程, 而且也不会丢失太多准确率.如果运用上了 SGD, 你还是嫌训练速度慢, 那怎么办?

![](http://oslivcbny.bkt.clouddn.com/tensorflow_05.png)

没问题, 事实证明, SGD 并不是最快速的训练方法, 红色的线是 SGD, 但它到达学习目标的时间是在这些方法中最长的一种. 我们还有很多其他的途径来加速训练.

**Momentum 更新方法**

![](http://oslivcbny.bkt.clouddn.com/tensorflow_06.png)

大多数其他途径是在更新神经网络参数那一步上动动手脚. 传统的参数 W 的更新是把原始的 W 累加上一个负的学习率(learning rate) 乘以校正值 (dx). 这种方法可能会让学习过程曲折无比, 看起来像 喝醉的人回家时, 摇摇晃晃走了很多弯路.

![](http://oslivcbny.bkt.clouddn.com/tensorflow_07.png)

所以我们把这个人从平地上放到了一个斜坡上, 只要他往下坡的方向走一点点, 由于向下的惯性, 他不自觉地就一直往下走, 走的弯路也变少了. 这就是 Momentum 参数更新. 另外一种加速方法叫AdaGrad.

**AdaGrad 更新方法**

![](http://oslivcbny.bkt.clouddn.com/tensorflow_08.png)

这种方法是在学习率上面动手脚, 使得每一个参数更新都会有自己与众不同的学习率, 他的作用和 momentum 类似, 不过不是给喝醉酒的人安排另一个下坡, 而是给他一双不好走路的鞋子, 使得他一摇晃着走路就脚疼, 鞋子成为了走弯路的阻力, 逼着他往前直着走. 他的数学形式是这样的. 接下来又有什么方法呢? 如果把下坡和不好走路的鞋子合并起来, 是不是更好呢? 没错, 这样我们就有了 RMSProp 更新方法.

**RMSProp 更新方法**

![](http://oslivcbny.bkt.clouddn.com/tensorflow_09.png)

有了 momentum 的惯性原则 , 加上 adagrad 的对错误方向的阻力, 我们就能合并成这样. 让 RMSProp同时具备他们两种方法的优势. 不过细心的同学们肯定看出来了, 似乎在 RMSProp 中少了些什么. 原来是我们还没把 Momentum合并完全, RMSProp 还缺少了 momentum 中的 这一部分. 所以, 我们在 Adam 方法中补上了这种想法.

**Adam 更新方法**

![](http://oslivcbny.bkt.clouddn.com/tensorflow_10.png)

计算m 时有 momentum 下坡的属性, 计算 v 时有 adagrad 阻力的属性, 然后再更新参数时 把 m 和 V 都考虑进去. 实验证明, 大多数时候, 使用 adam 都能又快又好的达到目标, 迅速收敛. 所以说, 在加速神经网络训练的时候, 一个下坡, 一双破鞋子, 功不可没.

以上所提到的优化器在`tf.train`中均有实现。

#### TensorFlow的可视化操作

学会用 Tensorflow 自带的 tensorboard 去可视化我们所建造出来的神经网络是一个很好的学习理解方式. 用最直观的流程图告诉你你的神经网络是长怎样,有助于你发现编程中间的问题和疑问.

首先，我们来搭建网络结构，用于接下来的可视化展示：

将之前的`input`代码修改为：

```python
xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
ys = tf.placeholder(tf.float32, [None, 1], name='y_in')
```

对输入数据指定名称，以方便接下来展示的时候更好的观测。

使用`with tf.name_scope('inputs')`可以将xs和ys包含进来，形成一个大的图层，图层的名字就是`with tf.name_scope()`方法里的参数。

将`layer`修改为：

```python
def add_layer(input, input_size, output_size, activation_func=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([input_size, output_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input, Weights) + biases
        if activation_func is None:
            output = Wx_plus_b
        else:
            output = activation_func(Wx_plus_b)
        return output
```

最后编辑loss部分：将`with tf.name_scope()`添加在loss上方，并为它起名为loss.

```python
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
```

使用with tf.name_scope()再次对train_step部分进行编辑,如下：

```python
with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```

我们需要使用 `tf.summary.FileWriter()`  将上面‘绘画’出的图保存到一个目录中，以方便后期在浏览器中可以浏览。 这个方法中的第二个参数需要使用sess.graph ， 因此我们需要把这句话放在获取session的后面。 这里的graph是将前面定义的框架信息收集起来，然后放在logs/目录下面。

```python
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter("/home/enningxie/logs", sess.graph)
```

最后在你的terminal（终端）中 ，使用以下命令

```python
tensorboard --logdir logs
```

在浏览器地址栏输入`http://localhost:6006`，即可进入tensorboard进入进一步了解。

![](http://oslivcbny.bkt.clouddn.com/tensorflow_11.png)

其实tensorboard还可以可视化训练过程（biases变化过程）

我们从输入部分开始，

```python
import tensorflow as tf
import numpy as np

## make up some data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
```

接下来制作`weights`和`biases`的变化图表





































### 杂记
