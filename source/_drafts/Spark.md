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
