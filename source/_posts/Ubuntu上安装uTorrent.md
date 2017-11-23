---
title: Ubuntu上安装uTorrent
tags: Ubuntu
categories: Ubuntu
comments: true
date: 2017-11-22 19:24:42
---


### uTorrent安装

uTorrent是一款BT下载软件，作为Windows上的迅雷等类似下载工具的替代品。

<!--more-->

- 更新软件库，安装必要的依赖

```
$ sudo apt-get update
$ sudo apt-get install libssl1.0.0 libssl-dev
```

- 从官网下载最新的[uTorrent server](http://www.utorrent.com/downloads/linux):

```
$ wget http://download-new.utorrent.com/endpoint/utserver/os/linux-x64-ubuntu-13-04/track/beta/ -O utserver.tar.gz
```

- 解压下载的文件到指定目录：

```
$ sudo tar -zxvf utserver.tar.gz -C /opt/
```

- 更改目录权限：

```
$ sudo chmod 777 /opt/utorrent-server-alpha-v3_3/
```

- 创建`/usr/bin`的链接：

```
$ sudo ln -s /opt/utorrent-server-alpha-v3_3/utserver /usr/bin/utserver
```

- 启动uTorrent Server:

```
$ utserver -settingspath /opt/utorrent-server-alpha-v3_3/
```

运行之后，使用浏览器访问：`http://<your_ip>:8080/gui`。

浏览器会提示你输入用户名及密码：（用户名admin，密码为空）

这样就进入了uTorrent的管理界面了，尽情的下载吧。

:)
