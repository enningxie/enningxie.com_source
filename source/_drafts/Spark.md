---
title: Spark
tags: Spark
categories: 技术
comments: true
---

### draft

Spark是一个用来实现快速而通用的集群计算的平台。

Spark比MapReduce高效。

Spark Core实现了Spark的基本功能，包括任务调度、内存管理、错误恢复、与存储系统交互等模块。

RDD:弹性分布式数据集，是Spark主要的编程抽象。

Spark SQL是Spark用来操作结构化数据的程序包。

Spark Streaming是Spark提供的对实时数据进行流式计算的组件。

MLlib是Spark提供的机器学习程序库。

GraphX是用来操作图（如社交网络的朋友关系图）的程序库，可以进行并行的图计算。

Spark支持在各种集群管理器上运行，包括Hadoop YARN、Apache Mesos，以及Spark自带的一个简易调度器，叫做独立调度器。

Spark本身是用Scala写的，运行在Java虚拟机（JVM）上。

RDD(弹性分布式数据集)是Spark对分布式数据和计算的基本抽象。

Maven是一个流行的包管理工具，可以用于任何基于Java的语言，让你可以连接公共仓库中的程序库。

在python中，你可以把应用写成python脚本，但是需要使用Spark自带的`bin/spark-submit`脚本来运行。`spark-submit`脚本会帮我们引入python程序的Spark依赖。这个脚本为Spark的pythonAPI配置好了运行环境。
