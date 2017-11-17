---
title: Linux for DeepLearning
tags: Linux
categories: 技术
comments: true
---

##　用于深度学习（机器学习）的Linux

这是一篇简短的Linux教程，主要用于应付，日常深度学习、机器学习项目里的工作。很多时候类似深度学习这种工程的开发是基于Linux的；所以说，熟悉一些有关的Linux知识还是很有必要的。因此，我们开始吧。

<!--more-->

首先，Linux是一个开放式的、开源系统，而且是免费的。世界上有很多优秀的程序员在共同维护和更新着这个系统。

Linux虽然是开源的系统，但并不会不安全。相反的，Linux由于其开源的特性，使得它比Windows更加安全。另外，Linux系统也不是特别吃硬件，以前老的机器上Windows带不动了，可以重新装个Linux试试，你的老机器，会焕然一新，不信动手试试。

关于Linux的版本选择和安装，本文就不做详细介绍。这部分，还请读者自行google。

本文的Linux命令，在Ubuntu上测试。其他版本类似。

系统、软件更新升级：

```
$ sudo apt-get update && sudo apt-get upgrade
```

### Working with vim

- type `yy` to copy a line

- type `p` to paste the line

- type `dd` to cut the line

- type `:w` to save any changes

- type `:q` to exit Vim

- `:wq` write and exit at the same time

- `:q!` exit without saving

### Basic regular expressions

- `grep "joe" file.txt` 从file.txt中查找`joe`

- `grep -i "joe" file.txt` 从file.txt中大小写不敏感的找出`joe`

- `.` matches any character

- `*` matches previous character multiple times

- `grep "^\s$" file.txt`　find empty lines.  
- `\s` this stands for space,
- `^` beginning of the line,
- `$` ending

- `sed "s/Joe/All/g" file.txt` find `Joe` word and replace with `All` word

### Pipes and subshells

- `pwd | wc -c` counting the length of the current path using the following command.

- `|` is the pipe symbol, and what it does is send the output of the command on the left to the command on the right. you can create a chain of any number of pipes.
