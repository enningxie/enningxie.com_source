---
title: Recommender system based deeplearning(papers)
tags: recsys
categories: recsys
comments: true
---

### 基于深度学习推荐系统领域感兴趣的论文

<!--more-->

众所周知，深度学习的发展在语音、图像、自然语言处理领域取得了很多成果。而且，深度学习在推荐系统和信息检索领域里也得到了应用。推荐系统领域结合深度学习是未来发展的趋势所在。

[1] 综述性文章，给出了基于深度学习推荐系统近年来的发展情况。

对深度学习整合进推荐系统的通常看法是显著优于传统模型。

deep learning based recommender system.

cnn: Deep neural networks for youtube recommendations. Image-based recommendations on styles and
substitutes. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback.

ae: Collaborative deep learning for recommender systems. Collaborative denoising auto-encoders for top-n recommender
systems. Autorec: Autoencoders meet collaborative ltering.

mlp: Deep neural networks for youtube recommendations. Wide & deep learning for recommender systems.

rnn: Session-based recommendations with recurrent neural networks.

dssm: A multi-view deep learning approach for cross domain user modeling
in recommendation systems.

rbm: Improving content-based and hybrid music recommendation using deep learning.

### re-update in 04-09

题目：一份对基于深度学习的推荐系统的调查和批判

摘要：近年来，推荐系统已经变得十分常见。类似Amazon或者eBay这样的公司，开发了数量庞大的产品为了满足各种客户的需要。客户们在电子商务领域内的选择也与日俱增。也就是说，在当今社会下，为了去寻找客户真正想要的，客户需要处理来自商家提供的海量的信息数据。推荐系统就是缓解如此的信息过量的方法之一。一方面来说，传统的推荐系统推荐商品基于不同的标准，像之前偏好的基于客户的个人信息。另一方面，深度学习技术在计算机视觉、语音识别和自然语言处理领域取得了瞩目的成就。然而，将深度学习技术应用于推荐系统领域的产品还不是很多。这篇文章中，我们在第一章节会介绍推荐系统领域传统的技术和深度学习技术，接下来会对几个最先进的基于深度学习的推荐系统进行进一步探索。

章节1

介绍

1.1 推荐系统

在过去的十年里，各种公司推出了大量各式各样的产品和服务。公司为了应对各种客户的需要推出了对应的产品。虽然某种程度上来说，客户能够获得更多的选择，但是无形中也增加了客户处理公司提供的大量数据带来的负担。推荐系统这时候被设计出来，用于帮助客户推荐相应的产品和优质的服务。**这些产品和服务很大程度上是被用户所偏爱的，原因是其基于客户的偏好、需要和购买历史等被推荐的。** 目前，在人们的日常生活中，很多人会用到推荐系统来网上购物、看新闻、看电影等等。

推荐系统被设计根据用户的购买历史和历史评分来推荐商品给用户，只需要提供一个关于用户（users）的集合 U 和一个关于商品（items）的集合 V。通常，一个推荐系统推荐商品给用户，不是根据预测出来的评分（ratings）就是为用户提供了一个排好序的商品列表。这里涉及到推荐系统领域内的两种技术：协同过滤算法（Collaborative Filtering）、基于内容的推荐（Content-based recommendations）。

1.1.1 协同过滤算法（Collaborative Filtering）

协同过滤算法（CF）在推荐系统领域可谓是家喻户晓。很多杰出的推荐方法[1-3]都是基于协同过滤算法[4]。

协同过滤算法源自一个很普遍的现象：每个用户是有自己独特的品味的，所以他购买的商品同样符合他的品味。例如：图1.1中，我们可以看出用户U1趋向于去购买商品I2，原因是用户U1和U4同样喜欢商品I1，而U4对商品I2给出了很高的分数（rating）。

![](http://oslivcbny.bkt.clouddn.com/20180410000238.png)

在一般的协同过滤算法的设置里，都会有一个关于用户偏好的集合存在。打个比方，现在有一个M个用户{u1, u2, ..., uM}和一个N个商品的{i1, i2, ..., iN}的列表。商品列表中的iu_i代表被用户u_i评过分，这般的评分可以是明确的指示（explicit indications），1-5范围内，或者是隐含的指示（implicit indications）。隐含的指示一般是隐含的反馈（implicit feedback），如来自用户的历史购买记录或是点击记录等。协同过滤算法可以是基于内存的（memory-based）也可以是基于模型的（model-based）。

1.1.1.1 基于内存的协同过滤

基于内存的协同过滤系统，推荐和预测都是基于相似度。评分数据往往都是用来计算用户和商品之间的相似度或者是权重信息。

对于基于内存的协同过滤有这样几个有点，首先，我们只需要计算相似度，这很简单，同样也容易实现。第二，基于内存的协同过滤系统能够处理大规模的数据。第三，大多数基于内存的系统都是在线学习的模型（online learning
models）。也就是说，对于新来的数据，系统处理起来也很简单。最后，推荐结果可以被理解，同样能够提供反馈用于解释为什么做出这样地推荐。然而，基于内存的推荐系统同样还是有几方面的局限性的。例如，由于相似度的值是基于商品的，对于稀疏的数据，被评过分的商品很少的情况，最终的推荐结果准确率就会很低。

最近邻协同过滤（Neighbor-based CF）就是基于内存协同过滤系统中的一个代表。一般最近邻协同过滤涉及两步，即：相似度的计算和预测。这里给出皮尔逊相似度的计算公式：

![](http://oslivcbny.bkt.clouddn.com/20180410003316.png)

基于向量余弦相似度（Vector Cosine-based Similarity）一般用来度量不同文本之间的不同。文本可以被表达成一个词频向量。在最近邻协同过滤中，基于向量余弦相似度被用来计算用户和商品之间的相似度。公式如下：

![](http://oslivcbny.bkt.clouddn.com/20180410003846.png)

在最近邻协同过滤系统中，我们用相似度来找到同目标用户u最相似的用户集合，从而来生成推荐或者预测。即，对目标用户的推荐和预测都可以通过集合中的用户的评分来做计算。

1.1.1.2 基于模型的协同过滤
