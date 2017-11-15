---
title: keras
tags: keras
categories: DeepLearning
comments: true
date: 2017-11-15 16:33:05
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

在进一步学习keras之前，我们先给出一些相关的概念，方便后续的学习。

##### 符号计算

Keras的底层库使用Theano或TensorFlow，这两个库也称为Keras的后端。无论是Theano还是TensorFlow，都是一个“符号式”的库。

因此，这也使得Keras的编程与传统的Python代码有所差别。笼统的说，符号主义的计算首先定义各种变量，然后建立一个“计算图”，计算图规定了各个变量之间的计算关系。建立好的计算图需要编译以确定其内部细节，然而，此时的计算图还是一个“空壳子”，里面没有任何实际的数据，只有当你把需要运算的输入放进去后，才能在整个模型中形成数据流，从而形成输出值。

Keras的模型搭建形式就是这种方法，在你搭建Keras模型完毕后，你的模型就是一个空壳子，只有实际生成可调用的函数后（K.function），输入数据，才会形成真正的数据流。

##### 张量

张量，或tensor，可以看作是向量、矩阵的自然推广，我们用张量来表示广泛的数据类型。

规模最小的张量是0阶张量，即标量，也就是一个数。

当我们把一些数有序的排列起来，就形成了1阶张量，也就是一个向量。

如果我们继续把一组向量有序的排列起来，就形成了2阶张量，也就是一个矩阵。

把矩阵摞起来，就是3阶张量，我们可以称为一个立方体，具有3个颜色通道的彩色图片就是一个这样的立方体。

把立方体摞起来，好吧这次我们真的没有给它起别名了，就叫4阶张量了，不要去试图想像4阶张量是什么样子，它就是个数学上的概念。

张量的阶数有时候也称为维度，或者轴，轴这个词翻译自英文axis。譬如一个矩阵[[1,2],[3,4]]，是一个2阶张量，有两个维度或轴，沿着第0个轴（为了与python的计数方式一致，本文档维度和轴从0算起）你看到的是[1,2]，[3,4]两个向量，沿着第1个轴你看到的是[1,3]，[2,4]两个向量。

要理解“沿着某个轴”是什么意思，不妨试着运行一下下面的代码：

```Python
import numpy as np

a = np.array([[1, 2], [3, 4]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)

print(sum0)
print(sum1)
```

关于张量，目前知道这么多就足够了。事实上我也就知道这么多.

##### data_format

这是一个无可奈何的问题，在如何表示一组彩色图片的问题上，Theano和TensorFlow发生了分歧，'th'模式，也即Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），Caffe采取的也是这种方式。第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。这种theano风格的数据组织方法，称为“channels_first”，即通道维靠前。

而TensorFlow，的表达形式是（100,16,32,3），即把通道维放在了最后，这种数据组织方式称为“channels_last”。

Keras默认的数据组织形式在~/.keras/keras.json中规定，可查看该文件的image_data_format一项查看，也可在代码中通过K.image_data_format()函数返回，请在网络的训练和测试中保持维度顺序一致。

##### 函数式模型

在Keras 0.x中，模型其实有两种，一种叫Sequential，称为序贯模型，也就是单输入单输出，一条路通到底，层与层之间只有相邻关系，跨层连接统统没有。这种模型编译速度快，操作上也比较简单。第二种模型称为Graph，即图模型，这个模型支持多输入多输出，层与层之间想怎么连怎么连，但是编译速度慢。可以看到，Sequential其实是Graph的一个特殊情况。

在Keras1和Keras2中，图模型被移除，而增加了了“functional model API”，这个东西，更加强调了Sequential是特殊情况这一点。一般的模型就称为Model，然后如果你要用简单的Sequential，OK，那还有一个快捷方式Sequential。

由于functional model API在使用时利用的是“函数式编程”的风格，我们这里将其译为函数式模型。总而言之，只要这个东西接收一个或一些张量作为输入，然后输出的也是一个或一些张量，那不管它是什么鬼，统统都称作“模型”。

##### batch

深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。

另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。

为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

基本上现在的梯度下降都是基于mini-batch的，所以Keras的模块中经常会出现batch_size，就是指这个。

顺便说一句，Keras中用的优化器SGD是stochastic gradient descent的缩写，但不代表是一个样本就更新一回，还是基于mini-batch的。

##### epochs

epochs指的就是训练过程中数据将被“轮”多少次，就这样。

##### batch, epochs, sample概念解析

- Sample：样本，数据集中的一条数据。例如图片数据集中的一张图片，语音数据中的一段音频。

- Batch：中文为批，一个batch由若干条数据构成。batch是进行网络优化的基本单位，网络参数的每一轮优化需要使用一个batch。batch中的样本是被并行处理的。与单个样本相比，一个batch的数据能更好的模拟数据集的分布，batch越大则对输入数据分布模拟的越好，反应在网络训练上，则体现为能让网络训练的方向“更加正确”。但另一方面，一个batch也只能让网络的参数更新一次，因此网络参数的迭代会较慢。在测试网络的时候，应该在条件的允许的范围内尽量使用更大的batch，这样计算效率会更高。

- Epoch，epoch可译为“轮次”。如果说每个batch对应网络的一次更新的话，一个epoch对应的就是网络的一轮更新。每一轮更新中网络更新的次数可以随意，但通常会设置为遍历一遍数据集。因此一个epoch的含义是模型完整的看了一遍数据集。 设置epoch的主要作用是把模型的训练的整个训练过程分为若干个段，这样我们可以更好的观察和调整模型的训练。Keras中，当指定了验证集时，每个epoch执行完后都会运行一次验证集以确定模型的性能。另外，我们可以使用回调函数在每个epoch的训练前后执行一些操作，如调整学习率，打印目前模型的一些信息等。

---

#### 保存keras模型

不推荐使用pickle或cPickle来保存keras模型

可以使用`model.save(filepath)`将keras模型和权重保存在一个HDF5文件中，该文件将包含：

- 模型的结构，以便重构该模型

- 模型的权重

- 训练配置（损失函数，优化器等）

- 优化器的状态，以便从上次训练中断的地方开始

使用`keras.models.load_model(filepath)`来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译

例子：

```Python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
def model

model = load_model('my_mdoel.h5')
```

如果你只是希望保存模型的结构，而不包含其权重或配置信息，可以使用：

```Python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```

这项操作将把模型序列化为json或yaml文件，这些文件对人而言也是友好的，如果需要的话你甚至可以手动打开这些文件并进行编辑。

当然，你也可以从保存好的json文件或yaml文件中载入模型：

```Python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```

如果需要保存模型的权重，可通过下面的代码利用HDF5进行保存。注意，在使用前需要确保你已安装了HDF5和其Python库h5py

```Python
model.save_weights('my_model_weights.h5')
```

如果你需要在代码中初始化一个完全相同的模型，请使用：

```Python
model.load_weights('my_model_weights.h5')
```

如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，你可以通过层名字来加载模型：

```Python
model.load_weights('my_model_weights.h5', by_name=True)
```

如：

```Python
"""
假如原模型为：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))
    model.add(Dense(3, name="dense_2"))
    ...
    model.save_weights(fname)
"""
# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
model.add(Dense(10, name="new_dense"))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

#### 获取中间层的输出

一种简单的方法是创建一个新的`Model`，使得它的输出是你想要的那个输出

```Python
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

此外，我们也可以建立一个Keras的函数来达到这一目的：

```Python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([X])[0]
```

当然，我们也可以直接编写Theano和TensorFlow的函数来完成这件事

### 开始上路

#### 序贯(Sequential)模型

序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”。

可以通过向`Sequential`模型传递一个layer的list来构造该模型：

```Python
from keras.models import  Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, units=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
```

也可以通过`.add()`方法一个个将layer加入模型中：

```Python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
```

##### 指定输入数据的shape

模型需要知道输入数据的shape，因此，`Sequential`的第一层需要接受一个关于输入数据shape的参数，后面的各个层则可以自动的推导出中间数据的shape，因此不需要为每个层都指定这个参数。有几种方法来为第一层指定输入数据的shape

- 传递一个`input_shape`的关键字参数给第一层，`input_shape`是一个tuple类型的数据，其中也可以填入None，如果填入None则表示此位置可能是任何正整数。数据的batch大小不应包含在其中。

- 有些2D层，如Dense，支持通过指定其输入维度`input_dim`来隐含的指定输入数据shape。一些3D的时域层支持通过参数`input_dim`和`input_length`来指定输入shape。

- 如果你需要为输入指定一个固定大小的batch_size（常用于stateful RNN网络），可以传递`batch_size`参数到一个层中，例如你想指定输入张量的batch大小是32，数据shape是（6，8），则你需要传递`batch_size=32`和`input_shape=(6,8)`。

```Python
model = Sequential()
model.add(Dense(32, input_dim=784))
```

```Python
model = Sequential()
model.add(Dense(32, input_shape=784))
```

##### 编译

在训练模型之前，我们需要通过`compile`来对学习过程进行配置。`compile`接收三个参数：

- 优化器optimizer：该参数可指定为已预定义的优化器名，如`rmsprop`、`adagrad`，或一个`Optimizer`类的对象。

- 损失函数loss：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如`categorical_crossentropy`、`mse`，也可以为一个损失函数。

- 指标列表metrics：对分类问题，我们一般将该列表设置为`metrics=['accuracy']`。指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.指标函数应该返回单个张量,或一个完成`metric_name - > metric_value`映射的字典.

```Python
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

##### 训练

Keras以Numpy数组作为输入数据和标签的数据类型。训练模型一般使用`fit`函数

```Python
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
```

```Python
# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

##### 例子

这里是一些帮助你开始的例子

**基于多层感知器的softmax多分类**：

```Python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units
# in the first layer you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 ;
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy']
              )
model.fit(x_train, y_train, epochs=20, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print("Test set's score: ", score)
```

**MLP的二分类**

```Python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

#### 函数式（Functional）模型

函数式模型称作Functional，但它的类名是Model，因此我们有时候也用Model来代表函数式模型。

Keras函数式模型接口是用户定义多输出模型、非循环有向模型或具有共享层的模型等复杂模型的途径。一句话，只要你的模型不是类似VGG一样一条路走到黑的模型，或者你的模型需要多于一个的输出，那么你总应该选择函数式模型。函数式模型是最广泛的一类模型，序贯模型（Sequential）只是它的一种特殊情况。

这部分的文档假设你已经对Sequential模型已经比较熟悉.

让我们从简单一点的模型开始

##### 第一个模型：全连接网络

`Sequential`当然是实现全连接网络的最好方式，但我们从简单的全连接网络开始，有助于我们学习这部分的内容。在开始前，有几个概念需要澄清：

- 层对象接受张量为参数，返回一个张量。

- 输入是张量，输出也是张量的一个框架就是一个模型，通过`Model`定义。

- 这样的模型可以被像keras的`Sequential`一样被训练。

```Python
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(data, labels)  # starts training
```

##### 所有的模型都是可调用的，就像层一样

利用函数式模型的接口，我们可以很容易的重用已经训练好的模型：你可以把模型当作一个层一样，通过提供一个tensor来调用它。注意当你调用一个模型时，你不仅仅重用了它的结构，也重用了它的权重。

```Python
x = Input(shape=(784,))
y = model(x)
```

这种方式可以允许你快速的创建能处理序列信号的模型，你可以很快将一个图像分类的模型变为一个对视频分类的模型，只需要一行代码：

```Python
from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```

##### 多输入和多输出模型

使用函数式模型的一个典型场景是搭建多输入、多输出的模型。

考虑这样一个模型。我们希望预测Twitter上一条新闻会被转发和点赞多少次。模型的主要输入是新闻本身，也就是一个词语的序列。但我们还可以拥有额外的输入，如新闻发布的日期等。这个模型的损失函数将由两部分组成，辅助的损失函数评估仅仅基于新闻本身做出预测的情况，主损失函数评估基于新闻和额外信息的预测的情况，即使来自主损失函数的梯度发生弥散，来自辅助损失函数的信息也能够训练Embeddding和LSTM层。在模型中早点使用主要的损失函数是对于深度网络的一个良好的正则方法。总而言之，该模型框图如下：

![](http://oslivcbny.bkt.clouddn.com/multi-input-multi-output-graph.png)

让我们用函数式模型来实现这个框图

主要的输入接收新闻本身，即一个整数的序列（每个整数编码了一个词）。这些整数位于1到10，000之间（即我们的字典有10，000个词）。这个序列有100个单词。

```Python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
```

然后，我们插入一个额外的损失，使得即使在主损失很高的情况下，LSTM和Embedding层也可以平滑的训练。

```Python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```

再然后，我们将LSTM与额外的输入数据串联起来组成输入，送入模型中：

```Python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```

最后，我们定义整个2输入，2输出的模型：

```Python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```

模型定义完毕，下一步编译模型。我们给额外的损失赋0.2的权重。我们可以通过关键字参数`loss_weights`或`loss`来为不同的输出设置不同的损失函数或权值。这两个参数均可为Python的列表或字典。这里我们给`loss`传递单个损失函数，这个损失函数会被应用于所有输出上。

```Python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```

编译完成后，我们通过传递训练数据和目标值训练该模型：

```Python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```

因为我们输入和输出是被命名过的（在定义时传递了“name”参数），我们也可以用下面的方式编译和训练模型：

```Python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

##### 共享层

另一个使用函数式模型的场合是使用共享层的时候。

考虑微博数据，我们希望建立模型来判别两条微博是否是来自同一个用户，这个需求同样可以用来判断一个用户的两条微博的相似性。

一种实现方式是，我们建立一个模型，它分别将两条微博的数据映射到两个特征向量上，然后将特征向量串联并加一个logistic回归层，输出它们来自同一个用户的概率。这种模型的训练数据是一对对的微博。

因为这个问题是对称的，所以处理第一条微博的模型当然也能重用于处理第二条微博。所以这里我们使用一个共享的LSTM层来进行映射。

首先，我们将微博的数据转为（140，256）的矩阵，即每条微博有140个字符，每个单词的特征由一个256维的词向量表示，向量的每个元素为1表示某个字符出现，为0表示不出现，这是一个one-hot编码。

之所以是（140，256）是因为一条微博最多有140个字符，而扩展的ASCII码表编码了常见的256个字符。原文中此处为Tweet，所以对外国人而言这是合理的。如果考虑中文字符，那一个单词的词向量就不止256了。

```Python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
```

若要对不同的输入共享同一层，就初始化该层一次，然后多次调用它

```Python
# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

##### 层“节点”的概念

无论何时，当你在某个输入上调用层时，你就创建了一个新的张量（即该层的输出），同时你也在为这个层增加一个“（计算）节点”。这个节点将输入张量映射为输出张量。当你多次调用该层时，这个层就有了多个节点，其下标分别为0，1，2...

在上一版本的Keras中，你可以通过`layer.get_output()`方法来获得层的输出张量，或者通过`layer.output_shape`获得其输出张量的shape。这个版本的Keras你仍然可以这么做（除了`layer.get_output()`被output替换）。但如果一个层与多个输入相连，会出现什么情况呢?

如果层只与一个输入相连，那没有任何困惑的地方。`.output`将会返回该层唯一的输出

```Python
a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

但当层与多个输入相连时，会出现问题

```Python
a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```

这段代码就会报错

```Python
>> AssertionError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```

可以通过下面这种调用方式解决

```Python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

对于`input_shape`和`output_shape`也是一样，如果一个层只有一个节点，或所有的节点都有相同的输入或输出shape，那么`input_shape`和`output_shape`都是没有歧义的，并也只返回一个值。但是，例如你把一个相同的Conv2D应用于一个大小为(32,32,3)的数据，然后又将其应用于一个(64,64,3)的数据，那么此时该层就具有了多个输入和输出的shape，你就需要显式的指定节点的下标，来表明你想取的是哪个了

```Python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# Only one input so far, the following will work:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# now the `.input_shape` property wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

#### 关于keras模型

Keras有两种类型的模型，序贯模型（Sequential）和函数式模型（Model），函数式模型应用更为广泛，序贯模型是函数式模型的一种特殊情况。

两类模型有一些方法是相同的：

- `model.summary()`：打印出模型概况，它实际调用的是keras.utils.print_summary

- `model.get_config()`:返回包含模型配置信息的Python字典。模型也可以从它的config信息中重构回去

```Python
config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)
```

- `model.get_layer()`:依据层名或下标获得层对象

- `model.get_weights()`:返回模型权重张量的列表，类型为numpy array

- `model.set_weights()`:从numpy array里将权重载入给模型，要求数组具有与`model.get_weights()`相同的形状。

- `model.to_json`:返回代表模型的json字符串，仅包含网络结构，不包含权值。可以从JSON字符串中重构原模型：

```Python
from models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

- `model.to_yaml`：与`model.to_json`类似，同样可以从产生的YAML字符串中重构模型

```Python
from models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)`：将模型权重保存到指定路径，文件类型是HDF5（后缀是.h5）

- `model.load_weights(filepath, by_name=False)`：从HDF5文件中加载权重到当前模型中, 默认情况下模型的结构将保持不变。如果想将权重载入不同的模型（有些层相同）中，则设置by_name=True，只有名字匹配的层才会载入权重

#### 关于keras的“层”（Layer）

所有的keras层对象都有如下的方法：

- `layer.get_weights()`：返回层的权重（numpy array）

- `layer.set_weights(weights)`：从numpy array中将权重加载到该层中，要求numpy array的形状与`layer.get_weights()`的形状相同

- `layer.get_config()`：返回当前层配置信息的字典，层也可以借由配置信息重构：

```Python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

或者：

```Python
from keras import layers
config = layer.get_config()
layer = layers.deserialize({
                            'class_name': layer.__class__.__name__,
                            'config': config
})
```

如果层仅有一个计算节点（即该层不是共享层），则可以通过下列方法获得输入张量、暑促和张量、输入形状和输出数据的形状：

- `layer.input`

- `layer.output`

- `layer.input_shape`

- `layer.output_shape`

如果该层有多个计算节点，则可以使用下面的方法：

- `layer.get_input_at(node_index)`

- `layer.get_output_at(node_index)`

- `layer.get_input_shape_at(node_index)`

- `layer.get_output_shape_at(node_index)`

#### 常用层

常用层对应于core模块，core内部定义了一系列常用的网络层，包括全连接层、激活层等

##### Dense层

```Python
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

Dense就是常用的全连接层，所实现的运算是`output = activation(dot(input, kernel) + bias)`。其中`activation`是逐元素计算的激活函数，`kernel`是本层的权值矩阵，`bias`为偏置向量，只有当`use_bias=True`才会添加。


其他详细API请参考[keras官方文档](https://keras.io/)
