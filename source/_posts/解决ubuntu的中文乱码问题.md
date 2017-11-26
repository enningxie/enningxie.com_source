---
title: 解决ubuntu的中文乱码问题
tags: 坑
categories: 技术
comments: true
date: 2017-11-24 23:17:29
---

### ubuntu中文支持，及中文乱码问题

该篇博文，是本人踩了一下午的坑的成果，亲测有效。对服务器同样有效。

**状况**：所用的ubuntu系统不支持中文，遇见中文就????。ORZ...

<!--more-->

**目标**：使系统/服务器支持中文，能够正常显示。

首先，安装中文支持包`language-pack-zh-hans`：

```
$ sudo apt-get install language-pack-zh-hans
```

然后，修改`/etc/environment`（在文件的末尾追加）：

```
LANG="zh_CN.UTF-8"
LANGUAGE="zh_CN:zh:en_US:en"
```

再修改`/var/lib/locales/supported.d/local`(没有这个文件就新建，同样在末尾追加)：

```
en_US.UTF-8 UTF-8
zh_CN.UTF-8 UTF-8
zh_CN.GBK GBK
zh_CN GB2312
```

最后，执行命令：

```
$ sudo locale-gen
```

对于中文乱码是空格的情况，安装中文字体解决。

```
$ sudo apt-get install fonts-droid-fallback ttf-wqy-zenhei ttf-wqy-microhei fonts-arphic-ukai fonts-arphic-uming
```

以上，问题解决，中文显示正常。:)
