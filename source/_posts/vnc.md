---
title: vnc
tags: vnc
categories: ubuntu
comments: true
date: 2017-12-06 14:00:59
---


## Ubuntu 16.04 安装VNC及gnome桌面环境

由于实验室服务器所需，通过vnc连接服务器，实现图形化界面显示。

<!--more-->

### 安装桌面环境

**使用SSH登录服务器**

使用root账户。

**更新源及系统**

```
$ apt update
$ apt upgrade -y
```

**安装桌面环境**

远程连接使用gnome2。

完整安装（不推荐）：

```
$ apt install ubuntu-desktop gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal -y
```

仅安装核心组件：

```
$ apt-get install --no-install-recommends ubuntu-desktop gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal -y
```

### 安装VNC server

```
$ apt install vnc4server -y
```

**测试连接**

注：用户名填写当前正使用的用户名，例如root。IP地址填写当前这台服务器的IP地址。

```
$ ssh -L 5901:127.0.0.1:5901 用户名@IP地址或网址
```

提示是否继续连接，输入 yes。输入密码后，可以登录成功，证明配置正确。否则需要检查防火墙是否开放 5901 端口。

### 配置VNC server

```
vncserver :1
```

首次启动会让输入两遍 VNC 的密码，并且密码不可见。假如后期需要更改 VNC 连接密码，只需要输入 `vncpassword` 即可。

**关闭vncserver**

```
vncserver -kill :1
```

**修改配置文件**

修改 `~/.vnc/xstartup`，在 `x-window-manager &` 的后面新增下面这 4 行：

```
gnome-panel &
gnome-settings-daemon &
metacity &
nautilus &
```

完整的配置文件如下：

```
#!/bin/sh

# Uncomment the following two lines for normal desktop:
# unset SESSION_MANAGER
# exec /etc/X11/xinit/xinitrc

[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
xsetroot -solid grey
vncconfig -iconic &
x-terminal-emulator -geometry 80x24+10+10 -ls -title "$VNCDESKTOP Desktop" &
x-window-manager &

gnome-panel &
gnome-settings-daemon &
metacity &
nautilus &
```

此时再启动　vncserver

```
vncserver :1
```

**配置开机启动：**

首先输入 `crontab` 命令。

会提示选择默认的编辑器。推荐使用 vim，选择 vim.basic 就可以了。

Select an editor. To change later, run ‘select-editor’.

在末尾另起一行输入如下命令：

```
@reboot /usr/bin/vncserver :1
```

:)
