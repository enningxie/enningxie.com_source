---
title: Docker
tags: Docker
categories: 技术
comments: true
date: 2017-11-23 22:16:12
---


### 关于Docker的一些

关于Docker的介绍，一句话“一处配置，多处使用”，这里所说的使用是拿过来直接用，无须再次配置环境，而且保证了多处的环境的一致性。

Docker可以说是“一直折腾系统的”孩子的福音。以后只要做好一个Docker的镜像(image)，系统再炸的话，只要重装系统，拿来配置好的image就直接可以上手继续折腾了。:)

<!--more-->

#### Docker的安装

关于Docker的安装，考虑到各位看官手上的系统都不一样，所以...还是自行左转[google](www.google.com)好吧。ps:不高兴写:)

#### Docker的基本操作

在介绍Docker的基本操作之前，先给大家介绍一下关于Docker的三个概念：

- image(镜像)：最基本的东西，联想下自己的系统镜像（什么都没装的那种）。再一个需要注意的是Docker中的image类似，oop中的class的概念，然而它的实例化的东西就是我们接下来要提到的container了。

- container(容器)：image实例化之后的东西，可以在image的基础上，做特定的修改，有点个性化，专业化的意思。

- repository(仓库)：存放image的地方。有远程本地之分，该教程没有涉及到远程仓库。ps:为了供看官们玩耍，给大家提供了一个`docker pull`操作，从远程仓库pull指定镜像到本地玩耍:)

给出Docker官方提供的[API](https://docs.docker.com/engine/reference/run/)做为参考。

```
$ sudo docker pull ubuntu
```

`pull`后面跟`REPOSITORY:TAG`，上述的命令中只给出了`REPOSITORY`，所以默认的`TAG`就是`latest`。

```
$ sudo docker search ubuntu
```

用于在远程REPOSITORY中搜寻image。

```
$ sudo docker images
```

列出本地的镜像。

```
$ sudo docker inspect ubuntu:17.10
```

获得指定镜像的详细信息。

```
$ sudo docker rmi ubuntu:17.10
```

- `-f` 选项用于强制删除。

删除指定镜像。对于镜像来说，还有一个`image id`字段对镜像进行唯一的描述。所以一般操作用`image id`替换`ubuntu:17.10`效果一样。

```
$ sudo docker run -it --name anotherName ubuntu:17.10 /bin/bash
```

创建并运行容器，并执行`/bin/bash`命令。

- `-i`　打开交互模式；

- `-t`　为这个Docker容器分配一个伪终端。

- `-d` 将容器放到后台运行。

- `--name` 为容器分配一个别名。

```
$ sudo docker ps
```

不加参数，即列出正在运行的容器。

- `-l` 列出上一个运行的容器

- `-a` 列出所有容器

```
$ sudo docker start [anotherName] or [container id]
```

启动容器。

```
$ sudo docker stop [anotherName] or [container id]
```

停止。

```
$ sudo docker restart [anotherName] or [container id]
```

重启。

```
$ sudo docker rm [anotherName] or [container id]
```

删除容器。

```
$ sudo docker pause [anotherName] or [container id]
```

暂停。

```
$ sudo docker unpause [anotherName] or [container id]
```

恢复。

```
$ sudo docker top [anotherName] or [container id]
```

查看进程信息。

```
$ sudo docker inspect [anotherName] or [container id]
```

查看容器信息。

```
$ sudo docker attach [anotherName] or [container id]
```

进入容器（将后台运行的容器切换到前台运行）。

```
$ sudo docker exec [anotherName] or [container id] ps
```

在容器中执行另一条命令(`ps`)

```
$ sudo docker run -it -v /tmp:/data ubuntu:17.10 /bin/bash
```

上述的`-v`为容器和宿主机提供了一个共享数据卷(volume)的功能，`/tmp:/data`中`:`前一部分`/tmp`为宿主机上的目录，后面`/data`为要挂载到容器的目录。

#### 制作镜像

本文提供的Docker制作镜像是通过Dockerfile文件进行制作的。

`docker import`和`docker export`命令是用来导出和导入容器的。

`docker commit`命令用来创建容器的提交。

##### image的导入导出

**导出镜像**

```
$ sudo docker save ubuntu:17.10 ubuntu.tar
```

而一般导出镜像都要进行压缩处理，则

```
$ sudo docker save ubuntu:17.10 | gzip > ubuntu.tar.gz
```

**导入镜像**

```
$ sudo docker load --input centos7.tar
```

or

```
sudo docker load -i centos7.tar
```

##### 基础指令

**FROM**

作为Dockerfile文件中的第一条指令，

```
FROM <image>
```

指定从那个基础镜像构建。

例如：

```
FROM simpleware/bigdata-ubuntu:latest
```

**MAINTAINER**

提供镜像作者的信息

```
MAINTAINER enningxie <enningxie@163.com>
```

**RUN**

指定镜像构建过程中要的操作，这是重点，直接上例子，根据需要写命令：

```
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
&& /bin/bash /root/anaconda3.sh -b -p /opt/conda \
&& rm /root/anaconda3.sh \
&& tar -zxvf /root/pycharm.tar.gz -C /opt \
&& rm /root/pycharm.tar.gz \
&& apt-get clean
```

**WORKDIR**

用于切换构建过程中的工作目录。

```
WORKDIR /usr
```

**ADD**

用于将宿主机上的文件，在构建过程中载入容器

```
ADD <src> <dest>
```

将`<src>`中指定的文件、目录、乃至URL指定的远程文件复制到镜像<dest>目标路径中。

**COPY**

同`ADD`命令，引入外部文件进image，

```
COPY <src> <dest>
```

区别在于`COPY`指令不能识别网络地址，也不会自动对压缩文件进行解压操作。所以在不需要自动解压或者没有网络文件需求的时候，使用`COPY`指令是一个不错的选择。

**CMD**

`CMD`指令用于指定由镜像创建的容器中的主体程序。

```
CMD [ "/bin/bash" ]
```

与`RUN`命令的区别，`RUN`命令是在镜像构建的过程中执行，并将执行的结果提交到新的镜像层中；而`CMD`指令在镜像构建的过程中不能执行，它只是配置镜像的默认入口程序。

通过`CMD`指令指定了基于这个镜像所创建的容器在默认情况下运行的程序，当然这个程序是可以改变的。

```
$ sudo docker run -it ubuntu:17.10 pwd
```

这样就会被覆盖掉。

**ENV**

在Dockerfile中，用于指定环境变量

```
ENV PATH /opt/conda/bin:$PATH
```

通过`ENV`指令能够很轻松地设置Dockerfile中的环境变量。

##### 镜像的构建

```
$ sudo docker duild -t enningxie/ubuntu:python <src>
```

- `-t`指定构建后的image的`REPOSITORY:TAG`

- `<src>`指定用于构建镜像的Dockerfile文件的路径。

#### tips

在实际应用过程遇到，提供的镜像装有vnc服务，需要通过vnc进行gui界面的访问。

```
$ docker run --name chrome01 -P -d linfan/chrome
```

- `-P` Publish all exposed ports to random ports

- `-d` Run container in background and print container ID

```
$ docker port chrome01
```

运行结果：

```
5900/tcp -> 0.0.0.0:32775
```

查看端口，实际操作中vnc端口为5900，则下载[vnc客户端](https://www.realvnc.com/en/connect/download/viewer/)，配置`0.0.0.0:32775`，输入用户名，密码，即可访问运行着的容器中的gui界面程序。

以上。

have fun :)
