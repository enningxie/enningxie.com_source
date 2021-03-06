---
title: Memory
date: 2017-11-06 15:55:42
tags:
    - daily life
categories: 随笔
password: 110
---

记录一些日常所需的，不是什么重要的东西，但也不是不重要的东西。Locked.

<!--more-->

### 2017_10_23

- `sudo su - ennnigxie`  # '-'参数，切换用户的同时将其环境变量同时应用。'-l'参数也是这样的。

- 文件的压缩与解压(gzip/tar)  
`gzip test.txt` # 压缩文件  
`gzip -d test.txt.gz` # 解压文件  
\# `tar`打包压缩，通常有两种格式，分别以`.tar.bz2`和`.tar.gz`为后缀  
`tar jcvf test.tar.bz2 test.txt`  
\# 打包并调用`bzip2`压缩为`bz2`格式的文件  
`tar jxvf test.tar.bz2` # 解压`bz2` 文件  
`tar zcvf test.tar.gz test.txt` # 打包并调用`gzip`压缩为`gz`格式的文件  
`tar zxvf test.tar.gz` # 解压`gz`文件

---

### 2017_10_24

- `%f` 表示按浮点数的格式输出
- `%e` 表示按指数形式的浮点数的格式输出
- `%g` 表示自动选择合适的表示法输出

---

### 2017_10_25

 - 通配符：\*/?/[]  
 "*" 匹配０或多个任意字符  
 "?" 匹配单个字符  
 "[]" 可以匹配一组单个字符（例如，[12]），或者是匹配用连字符（“-”）指定的某一范围内的字符（例如，[1-3]）


 - `ls -R`  
 `-R`选项会递归地遍历指定目录，显示指定目录和它的每个子目录的内容。

 - Vim课堂  
 `x`  删除当前光标下的字符  
 `.`  重复上次修改  
 `u`  撤销操作  
 `dd`  删除操作，删除一整行  
 `>G`  会增加从当前行到末尾处的缩进层级  
 `a`  命令在当前光标之后添加内容  
 `A`  命令则在当前行的结尾添加内容  
 `$`  命令可以完成移动操作，光标移动至行尾


 ---

 ### 2017_10_26

 - `;`的命令是顺序执行的，但没有考虑它们的执行成功与否。分隔命令的一个更好的办法是使用`&&`，它同样也是依次顺序运行每条命令，但是只有当前面一条命令运行成功之后，才能执行下一条命令。如果一条命令运行失败，整个命令链就会停下来。

 - `||`只有第一条命令运行失败了，第二条才会执行。

 - `|`将第一条命令的输出作为第二条命令的输入。

 - `>`可以将输出从stdout重定向到文件。使用`>` 重定向时，如果文件不存在，就会创建一个新文件；如果文件已经存在，则会覆盖已有的文件。但是如果使用`>>`来替代，就会把输出追加到指定文件的底部（如果文件不存在就会创建它）。

 - `<` 将文件作为命令输入。ex:`$ echo < test.txt`

 - `>>` 追加

---

 *te**s**t*  
 __s__  
 * sd  
  * d  

1. d
2. df
3. ddasd
    * s d
    * sdas

![](./)

[xxx](sss)


> sdsad

- [ ] sdas

- [x] sdasd

sdasd|dsadas
--|---
saS|SAs

:+1:

:octocat:

:tada:

---

### 2017_10_27

- `cat`/`less`/`head`/`tail`。它们都能以只读模式显示文本文件的内容，但是显示的方法各不相同，`cat`命令一次显示整个文件的内容，而`less`命令则以分页方式一次一屏地显示内容。`head`命令和`tail`命令就好比一枚硬币的两面前者用于查看文件的开始部分，后者则显示文件结尾部分。总之，把这４个命令结合起来用，就可以很方便地查看文本文件中任何部分。

---

### 2017_10_30

- `!!`再次运行刚刚使用过的那条命令。

- `ps`和`top`跟踪计算机中正在运行的程序。

- `kill` 终止出错的程序。

- `lsof` 列出所有打开的文件。

- `free`　报告系统内存的占用情况。

- `df`和`du`磁盘空间占用情况。

- `ps aux`查看当前正在运行的所有进程。

- `ps U [username]`查看特定用户拥有的进程。

- `kill -9 [PID]` 停止任何正在进行的处理，马上关闭。

- `top`　查看正在运行的进程的动态更新列表。

- `free -h` 显示系统的内存信息。

- `df -h` 显示磁盘的使用情况。

- `du -h` 报告目录的使用空间，`du -hs` 报告目录使用的总空间。

---

### 2017_10_31

- `rpm -Uhv [packagename]` 为基于RPM的Linux系统安装软件。

- `rpm -ihv [packagename]` `-i` 安装软件包 `-h`安装过程中显示hash标记 `-v`显示命令的执行过程 `-U` update升级。

- `rpm -e` 删除基于RPM的linux系统中的软件。

- `dpkg -i [packagename]` 为Debian安装软件包。

- `dpkg -r [packagename]` 删除软件包。

- "DNS and BIND" 用于了解DNS.

- `host` 用于快速获知某个域名关联的ip地址。也可以做相反的操作。

- `traceroute ` 跟踪路由

- `ifconfig` 查看网络信息

- `ifdown` `ifup` 网络开关

- `ssh-keygen -t dsa` 创建一个ssh身份验证密钥

- `ssh-copy-id -i ~/.ssh/id_dsa.pub enningxie@10.131.102.87` 将公钥复制到目标主机，实现免密登录。

- `sftp enningxie@10.131.102.87` sftp到远程主机

- 有用的sftp命令  

命令|含义  
---|---
`cd`|切换目录
`exit`|关闭远程ssh服务器的连接
`get`|将指定的文件复制到本机
`help`|获取命令相关的帮助
`lcd`|将目录切换到本机
`lls`|列出本机上的文件
`ls`|列出远程ssh服务器上当前工作目录中的文件
`put`|将指定的文件复制到远程ssh服务器
`rm`|将指定的文件从远程ssh服务器删除

- `scp ~/bin/backup.sh enningxie@10.131.102.87:/home/enningxie/code` 安全地将一个文件从一台计算机复制到另一台计算机中。

---

### 2017_11_01

- `tar zxvf xxxx.tgz -C /usr/local/` 　解压到指定路径

- `ln -s spark-1.5 spark`  设置软链接

- `/etc/profile`  设置环境变量


---

### 2017_11_03

- 通过实现`__init__()`方法来初始化一个对象。每当创建一个对象，Python会先创建一个空对象，然后调用该对象的`__init__()`函数。

- `# coding=utf-8`　用于指定代码编码。

---

### 2017_11_04

##### spark 核心概念简介

- 每个Spark应用都由一个驱动器程序来发起集群上的各种并行操作。驱动器程序包含应用的main函数，并定义了集群上的分布式数据集，还对这些分布式数据集应用了相关操作。

- 驱动程序通过一个SparkContext对象来访问Spark。这个对象代表对集群的一个连接

---

### 2017_11_09

- `wget -r -np -nd url` 下载全部文件

- `proxychains wget ` 使用代理，需要安装

---

### 2017_11_17

- np.r_,np.c_

用法：concatenation function

np.r_按row来组合array,

np.c_按colunm来组合array

```Python
>>> a = np.array([1,2,3])
>>> b = np.array([5,2,5])
>>> //测试 np.r_
>>> np.r_[a,b]
array([1, 2, 3, 5, 2, 5])
>>>
>>> //测试 np.c_
>>> np.c_[a,b]
array([[1, 5],
       [2, 2],
       [3, 5]])
>>> np.c_[a,[0,0,0],b]
array([[1, 0, 5],
       [2, 0, 2],
       [3, 0, 5]])
```

---

### 2017_11_23

- tar -zxvf test.tar.gz -C /tmp

将`.tar.gz`压缩文件解压到指定目录。

---

### 2017_11_27

- `usermod -a -G groupname username`

将已有的用户附加到已有的组中。

`-G` 指定用户的附加组。

`-a` append与`-G`联合使用。

---

### 2017_12_14

- `ssh myserver -N -L localhost:8888:localhost:8888`

在客户端链接服务器端服务。

---

### 2018_03_22

- 'channels_first' is typically faster on GPUs while 'channels_last' is typically faster on CPUs.

---

### 2018_03_24
