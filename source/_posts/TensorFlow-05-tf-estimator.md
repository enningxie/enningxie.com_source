---
title: 'TensorFlow_05: tf.estimator'
tags: TensorFlow
categories: TensorFlow
comments: true
date: 2018-01-09 19:42:48
---


### tf.estimator

what's `tf.estimator`?

> TensorFlow’s high-level machine learning API (tf.estimator) makes it easy to configure, train, and evaluate a variety of machine learning models.

<!--more-->

Dataset: Iris data set.

> The Iris data set contains 150 rows of data, comprising 50 samples from each of three related Iris species: Iris setosa, Iris virginica, and Iris versicolor.

the Iris data has been randomized and split into two separate CSVs:

- A training set of 120 samples(iris_training.csv)

- A test set of 30 samples(iris_test.csv).

```python
# about tf.estimator
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from urllib.request import urlopen

import tensorflow as tf
import numpy as np

IRIS_TRAINING = '/home/enningxie/Documents/DataSets/iris/iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = '/home/enningxie/Documents/DataSets/iris/iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'


# download and store the dataset
def download(pwd, url):
    if not os.path.exists(pwd):
        print("Downloading Training Set.")
        raw = urlopen(url).read()
        with open(IRIS_TRAINING, 'wb') as f:
            f.write(raw)
    else:
        print("existed.")


# load dataset
def load(pwd):
    data_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=pwd,
        target_dtype=np.int,
        features_dtype=np.float32
    )
    return data_set


# construct a dnn classifier
def model():
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    # build 3 layer dnn with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/home/enningxie/Documents/Models/iris_model"
                                            )
    return classifier


# train
def train(model, input_data):
    # input
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(input_data.data)},
        y=np.array(input_data.target),
        num_epochs=None,
        shuffle=True
    )
    model.train(input_fn=train_input, steps=2000)


# test
def test(model, input_data):
    test_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(input_data.data)},
        y=np.array(input_data.target),
        num_epochs=1,
        shuffle=False
    )
    # evaluate accuracy.
    accuracy_score = model.evaluate(input_fn=test_input)['accuracy']
    print("Test set accuracy: ", accuracy_score)


# predict
def predict(model, input_data):
    predict_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": input_data},
        num_epochs=1,
        shuffle=False
    )
    predictions = list(model.predict(input_fn=predict_input))
    predicted_classes = [p['classes'] for p in predictions]
    print("New samples, class predictions: ", predicted_classes)


# main
def main():
    # download dataset
    download(IRIS_TRAINING, IRIS_TRAINING_URL)
    download(IRIS_TEST, IRIS_TEST_URL)

    # load dataset
    training_set = load(IRIS_TRAINING)
    test_set = load(IRIS_TEST)

    classifier = model()
    # train
    train(classifier, training_set)
    # test
    test(classifier, test_set)
    # predict
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict(classifier, new_samples)

    print("end.")


if __name__ == "__main__":
    main()
```

#### Building Input Functions with `tf.estimator`

Introduces you to creating input functions in `tf.estimator`.

You will get an overview of how to construct an `input_fn` to preprocess and feed data into your models.

The `input_fn` is used to pass feature and target data to the `train`, `evaluate`, and `predict` methods of `Estimator`.

The user can do feature engineering or pre-processing inside the `input_fn`.

The following code illustrates the basic skeleton for an input function:

```python
def my_input_fn():
    # Preprocess your data here ...

    # then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels
```

Input functions must return the following two values containing the final feature and label data to be fed into your model.

`feature_cols`:

A dict containing key/value pairs that map feature column names to Tensors (or SparseTensors) containing the corresponding feature data.

`labels`:

A Tensor containing your label (target) values: the values your model aims to predict.

#### Converting Feature Data to Tensors

if you feature/label data is a python array or stored in pandas dataframes or numpy arrays, you can use the following methods to construct `input_fn`:

```python
import numpy as np
# numpy input_fn
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_data)},
    y=np.array(y_data),
    ...
)
```

```python
import pandas as pd
# pandas input_fn
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": x_data}),
    y=pd.Series(y_data),
    ...
)
```

#### Passing input_fn Data to Your Model

To feed data to your model for training, you simply pass the input function you've created to your train operation as the value of the input_fn parameter, e.g.:

```python
classifier.train(input_fn=my_input_fn, steps=2000)
```

#### deal with pandas data

```python
# pandas input_fn
# DNNRegressor with custom input_fn for Housing dataset.
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


# input_fn
def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle
    )


def main(unused_argv):
    # load datasets
    training_set = pd.read_csv("/home/enningxie/Documents/DataSets/boston/boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("/home/enningxie/Documents/DataSets/boston/boston_test.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("/home/enningxie/Documents/DataSets/boston/boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    # Feature cols
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    regressor = tf.estimator.DNNRegressor(
        feature_columns=feature_cols,
        hidden_units=[10, 10],
        model_dir="/home/enningxie/Documents/Models/boston/"
    )

    # train
    regressor.train(input_fn=get_input_fn(training_set), steps=5000)

    # evaluate
    ev = regressor.evaluate(
        input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False)
    )
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    # prediction
    y = regressor.predict(
        input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False)
    )
    predictions = list(p["predictions"] for p in itertools.islice(y, 6))
    print("Predictions: {}".format(str(predictions)))


if __name__ == "__main__":
    tf.app.run()
```

#### Advantages of Estimators

- You can run Estimators-based models on a local host or on a distributed multi-server environment without changing your model. Furthermore, you can run Estimators-based models on CPUs, GPUs, or TPUs without recoding your model.

- Estimators simplify sharing implementations between model developers.

- You can develop a state of the art model with high-level intuitive code, In short, it is generally much easier to create models with Estimators than with the low-level TensorFlow APIs.

- Estimators are themselves built on tf.layers, which simplifies customization.

- Estimators build the graph for you. In other words, you don't have to build the graph.

- Estimators provide a safe distributed training loop that controls how and when to: build the graph, initialize variables, start queues, handle exceptions, create checkpoint files and recover from failures, save summaries for TensorBoard.

#### Structure of a pre-made Estimators program

1. **Write one or more dataset importing functions.** For example, you might create one function to import the training set and another function to import the test set. Each dataset importing function must return two objects:

- a dictionary in which the keys are feature names and the values are Tensors (or SparseTensors) containing the corresponding feature data

- a Tensor containing one or more labels

```python
def input_fn(dataset):
   ...  # manipulate dataset, extracting feature names and the label
   return feature_dict, label
```

2. **Define the feature columns.** Each tf.feature_column identifies a feature name, its type, and any input pre-processing. For example, the following snippet creates three feature columns that hold integer or floating-point data. The first two feature columns simply identify the feature's name and type. The third feature column also specifies a lambda the program will invoke to scale the raw data:

```python
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn='lambda x: x - global_education_mean')
```

3. **Instantiate the relevant pre-made Estimator.** For example, here's a sample instantiation of a pre-made Estimator named `LinearClassifier`:

```python
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.Estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
```

4. **Call a training, evaluation, or inference method.** For example, all Estimators provide a train method, which trains a model.

```python
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
```

#### Benefits of pre-made Estimators

- Best practices for determining where different parts of the computational graph should run, implementing strategies on a single machine or on a cluster.

- Best practices for event (summary) writing and universally useful summaries.

#### Recommended workflow

1. Assuming a suitable pre-made Estimator exists, use it to build your first model and use its results to establish a baseline.

2. Build and test your overall pipeline, including the integrity and reliability of your data with this pre-made Estimator.

3. If suitable alternative pre-made Estimators are available, run experiments to determine which pre-made Estimator produces the best results.

4. Possibly, further improve your model by building your own custom Estimator.

#### Creating Estimators in `tf.estimator`

Let's see how to create your own `Estimator` using the building blocks provided in `tf.estimator`.

you will learn:

- Instantiate an `Estimator`

- Construct a custom model function

- Configure a neural network using `tf.feature_column` and `tf.layers`

- Choose an appropriate loss function from `tf.losses`

- Define a training op for your model

- Generate and return predictions

#### Converting Data into Tensors

Each continuous column in the train or test data will be converted into a Tensor, which in general is a good format to represent dense data. For categorical data, we must represent the data as a SparseTensor. This data format is good for representing sparse data.

#### Base Categorical Feature Columns

To define a feature column for a categorical feature, we can create a `CategoricalColumn` using the tf.feature_column API. If you know the set of all possible feature values of a column and there are only a few of them, you can use `categorical_column_with_vocabulary_list`. Each key in the list will get assigned an auto-incremental ID starting from 0. For example, for the `relationship` column we can assign the feature string "Husband" to an integer ID of 0 and "Not-in-family" to 1, etc., by doing:

```python
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])
```

What if we don't know the set of possible values in advance? Not a problem. We can use `categorical_column_with_hash_bucket` instead:

```python
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
```

What will happen is that each possible value in the feature column occupation will be hashed to an integer ID as we encounter them in training.

If we want to learn the fine-grained correlation between income and each age group separately, we can leverage bucketization. Bucketization is a process of dividing the entire range of a continuous feature into a set of consecutive bins/buckets, and then converting the original numerical feature into a bucket ID (as a categorical feature) depending on which bucket that value falls into. So, we can define a `bucketized_column` over age as:

```python
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

where the `boundaries` is a list of bucket boundaries. In this case, there are 10 boundaries, resulting in 11 age group buckets (from age 17 and below, 18-24, 25-29, ..., to 65 and over).

Using each base feature column separately may not be enough to explain the data. For example, the correlation between education and the label (earning > 50,000 dollars) may be different for different occupations. Therefore, if we only learn a single model weight for education="Bachelors" and education="Masters", we won't be able to capture every single education-occupation combination (e.g. distinguishing between education="Bachelors" AND occupation="Exec-managerial" and education="Bachelors" AND occupation="Craft-repair"). To learn the differences between different feature combinations, we can add `crossed feature columns` to the model.

```python
education_x_occupation = tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)
```

In the previous section we've seen several types of base and derived feature columns, including:

- `CategoricalColumn`

- `NumericColumn`

- `BucketizedColumn`

- `CrossedColumn`

All of these are subclasses of the abstract `FeatureColumn` class, and can be added to the `feature_columns` field of a model:

```python
base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]
crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]

model_dir = tempfile.mkdtemp()
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns)
```

#### Training and Evaluating Our Model

After adding all the features to the model, now let's look at how to actually train the model. Training a model is just a single command using the tf.estimator API:

```python
model.train(input_fn=lambda: input_fn(train_data, num_epochs, True, batch_size))
```

After the model is trained, we can evaluate how good our model is at predicting the labels of the holdout data:

```python
results = model.evaluate(input_fn=lambda: input_fn(
    test_data, 1, False, batch_size))
for key in sorted(results):
  print('%s: %s' % (key, results[key]))
```

The first line of the final output should be something like `accuracy: 0.83557522`, which means the accuracy is 83.6%. Feel free to try more features and transformations and see if you can do even better!

#### Adding Regularization to Prevent Overfitting

Regularization is a technique used to avoid overfitting. Overfitting happens when your model does well on the data it is trained on, but worse on test data that the model has not seen before, such as live traffic. Overfitting generally occurs when a model is excessively complex, such as having too many parameters relative to the number of observed training data. Regularization allows for you to control your model's complexity and makes the model more generalizable to unseen data.

In the Linear Model library, you can add L1 and L2 regularizations to the model as:

```python
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0))
```

One important difference between L1 and L2 regularization is that L1 regularization tends to make model weights stay at zero, creating sparser models, whereas L2 regularization also tries to make the model weights closer to zero but not necessarily zero. Therefore, if you increase the strength of L1 regularization, you will have a smaller model size because many of the model weights will be zero. This is often desirable when the feature space is very large but sparse, and when there are resource constraints that prevent you from serving a model that is too large.

In practice, you should try various combinations of L1, L2 regularization strengths and find the best parameters that best control overfitting and give you a desirable model size.

Model training is an optimization problem: The goal is to find a set of model weights (i.e. model parameters) to minimize a loss function defined over the training data, such as logistic loss for Logistic Regression models. The loss function measures the discrepancy between the ground-truth label and the model's prediction. If the prediction is very close to the ground-truth label, the loss value will be low; if the prediction is very far from the label, then the loss value would be high.

#### construct your own estimator

when you're creating your own estimator from scratch, the constructor accepts just two high-level parameters for model configuration, `model_fn` and `params`:

```python
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
```

- `model_fn`: A function object that contains all the aforementioned logic to support training, evaluation, and prediction. You are responsible for implementing that functionality.

- `params`: An optional dict of hyperparameters (e.g., learning rate, dropout) that will be passed into the `model_fn`.

```python
# Set model params
model_params = {"learning_rate": LEARNING_RATE}

# Instantiate Estimator
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
```

#### Constructing the `model_fn`

The basic skeleton for an `Estimator` API model function looks like this:

```python
def model_fn(features, labels, mode, params):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
   return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
```

The `model_fn` must accept three arguments:

- `features`: A dict containing the features passed to the model via `input_fn`.

- `labels`: A `Tensor` containing the labels passed to the model via `input_fn`. Will be empty for `predict()` calls, as these are the values the model will infer.

- `mode`: One of the following `tf.estimator.ModeKeys` string values indicating the context in which the `model_fn` was invoked:

- `tf.estimator.ModeKeys.TRAIN` The `model_fn` was invoked in training mode, namely via a `train()` call.

- `tf.estimator.ModeKeys.EVAL`. The `model_fn` was invoked in evaluation mode, namely via an `evaluate()` call.

- `tf.estimator.ModeKeys.PREDICT`. The `model_fn` was invoked in predict mode, namely via a `predict()` call.

`model_fn` may also accept a `params` argument containing a dict of hyperparameters used for training

The body of the function performs the following tasks (described in detail in the sections that follow):

- Configuring the mode

- Defining the loss function used to calculate how closely the model's predictions match the target values.

- Defining the training operation that specifies the `optimizer` algorithm to minimize the loss values calculated by the loss function.

The `model_fn` must return a `tf.estimator.EstimatorSpec` object, which contains the following values:

- `mode` (required). The mode in which the model was run. Typically, you will return the mode argument of the `model_fn` here.

- `predictions` (required in `PREDICT` mode). A dict that maps key names of your choice to `Tensors` containing the predictions from the model, e.g.:  
`python predictions = {"results": tensor_of_predictions}`  
In `PREDICT` mode, the dict that you return in `EstimatorSpec` will then be returned by predict(), so you can construct it in the format in which you'd like to consume it.

- `loss` (required in EVAL and TRAIN mode). A Tensor containing a scalar loss value: the output of the model's loss function calculated over all the input examples. This is used in TRAIN mode for error handling and logging, and is automatically included as a metric in EVAL mode.

- `train_op` (required only in `TRAIN` mode). An Op that runs one step of training.

- `eval_metric_ops` (optional). A dict of name/value pairs specifying the metrics that will be calculated when the model runs in `EVAL` mode. The name is a label of your choice for the metric, and the value is the result of your metric calculation. The `tf.metrics` module provides predefined functions for a variety of common metrics. The following `eval_metric_ops` contains an `"accuracy"` metric calculated using `tf.metrics.accuracy`:  
`python eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels, predictions) }`  
If you do not specify `eval_metric_ops`, only `loss` will be calculated during evaluation.

#### Configuring a neural network with `tf.feature_column` and `tf.layers`

Constructing a neural network entails creating and connecting the input layer, the hidden layers, and the output layer.

The input layer is a series of nodes (one for each feature in the model) that will accept the feature data that is passed to the `model_fn` in the features argument. If features contains an n-dimensional `Tensor` with all your feature data, then it can serve as the input layer. If features contains a dict of feature columns passed to the model via an input function, you can convert it to an input-layer Tensor with the `tf.feature_column.input_layer` function.

```python
input_layer = tf.feature_column.input_layer(
    features=features, feature_columns=[age, height, weight])
```

As shown above, `input_layer()` takes two required arguments:

- `features`. A mapping from string keys to the `Tensors` containing the corresponding feature data. This is exactly what is passed to the `model_fn` in the `features` argument.

- `feature_columns`. A list of all the `FeatureColumns` in the model—age, height, and weight in the above example.

The input layer of the neural network then must be connected to one or more hidden layers via an `activation function` that performs a nonlinear transformation on the data from the previous layer. The last hidden layer is then connected to the output layer, the final layer in the model. `tf.layers` provides the `tf.layers.dense` function for constructing fully connected layers. The activation is controlled by the activation argument. Some options to pass to the activation argument are:

- `tf.nn.relu`. The following code creates a layer of `units` nodes fully connected to the previous layer `input_layer` with a ReLU activation function (`tf.nn.relu`):  
`python hidden_layer = tf.layers.dense( inputs=input_layer, units=10, activation=tf.nn.relu)`

- `tf.nn.relu6`. The following code creates a layer of `units` nodes fully connected to the previous layer `hidden_layer` with a ReLU 6 activation function (`tf.nn.relu6`):  
`python second_hidden_layer = tf.layers.dense( inputs=hidden_layer, units=20, activation=tf.nn.relu)`

- `None`. The following code creates a layer of units nodes fully connected to the previous layer `second_hidden_layer` with no activation function, just a linear transformation:  
`python output_layer = tf.layers.dense( inputs=second_hidden_layer, units=3, activation=None)`

Other activation functions are possible, e.g.:

```python
output_layer = tf.layers.dense(inputs=second_hidden_layer,
                               units=10,
                               activation_fn=tf.sigmoid)
```

The above code creates the neural network layer `output_layer`, which is fully connected to `second_hidden_layer` with a sigmoid activation function (`tf.sigmoid`).

Putting it all together, the following code constructs a full neural network for the abalone predictor, and captures its predictions:

```python
def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}
  ...
```

Here, because you'll be passing the Datasets using `numpy_input_fn` as shown below, features is a dict `{"x": data_tensor}`, so `features["x"]` is the input layer. The network contains two hidden layers, each with 10 nodes and a `ReLU` activation function. The output layer contains no activation function, and is tf.reshape to a one-dimensional tensor to capture the model's predictions, which are stored in `predictions_dict`.

#### Defining loss for the model

The `EstimatorSpec` returned by the `model_fn` must contain `loss`: a `Tensor` representing the loss value, which quantifies how well the model's predictions reflect the label values during training and evaluation runs. The `tf.losses` module provides convenience functions for calculating loss using a variety of metrics, including:

- `absolute_difference`(labels, predictions). Calculates loss using the absolute-difference formula (also known as L1 loss).

- `log_loss`(labels, predictions). Calculates loss using the logistic loss forumula (typically used in logistic regression).

- `mean_squared_error`(labels, predictions). Calculates loss using the mean squared error (MSE; also known as L2 loss).

The following example adds a definition for `loss` to the `model_fn` using `mean_squared_error()` (in bold):

```python
def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)
  ...
```

Supplementary metrics for evaluation can be added to an `eval_metric_ops` dict. The following code defines an rmse metric, which calculates the root mean squared error for the model predictions. Note that the labels tensor is cast to a float64 type to match the data type of the predictions tensor, which will contain real values:

```python
eval_metric_ops = {
    "rmse": tf.metrics.root_mean_squared_error(
        tf.cast(labels, tf.float64), predictions)
}
```

#### Defining the training op for the model

The training op defines the optimization algorithm TensorFlow will use when fitting the model to the training data. Typically when training, the goal is to minimize loss. A simple way to create the training op is to instantiate a `tf.train.Optimizer` subclass and call the minimize method.

The following code defines a training op for the model_fn using the loss value calculated in `Defining Loss for the Model`, the learning rate passed to the function in params, and the gradient descent optimizer. For `global_step`, the convenience function `tf.train.get_global_step` takes care of generating an integer variable:

```python
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=params["learning_rate"])
train_op = optimizer.minimize(
    loss=loss, global_step=tf.train.get_global_step())
```

#### The complete model_fn

Here's the final, complete `model_fn`. The following code configures the neural network; defines loss and the training op; and returns a `EstimatorSpec` object containing `mode`, `predictions_dict`, `loss`, and `train_op`:

```python
def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"ages": predictions})

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), predictions)
  }

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)
```

#### Running the model

You've instantiated an `Estimator` and defined its behavior in `model_fn`; all that's left to do is `train`, `evaluate`, and make `predictions`.

Add the following code to the end of `main()` to fit the neural network to the training data and evaluate accuracy:

```python
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

# Train
nn.train(input_fn=train_input_fn, steps=5000)

# Score accuracy
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

ev = nn.evaluate(input_fn=test_input_fn)
print("Loss: %s" % ev["loss"])
print("Root Mean Squared Error: %s" % ev["rmse"])
```

To predict ages for the `ABALONE_PREDICT` data set, add the following to `main()`:

```python
# Print out predictions
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": prediction_set.data},
    num_epochs=1,
    shuffle=False)
predictions = nn.predict(input_fn=predict_input_fn)
for i, p in enumerate(predictions):
  print("Prediction %s: %s" % (i + 1, p["ages"]))
```

Congrats! You've successfully built a tf.estimator `Estimator` from scratch.

---

step by step:

[step 1](https://www.tensorflow.org/get_started/estimator)

[step 2](https://www.tensorflow.org/get_started/input_fn)

[step 3](https://www.tensorflow.org/programmers_guide/estimators)

[step 4](https://www.tensorflow.org/tutorials/wide)

[step 5](https://www.tensorflow.org/extend/estimators)

---

xz

:)
