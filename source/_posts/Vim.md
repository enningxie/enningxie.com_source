---
title: Vim
tags: Vim
categories: 技术
comments: true
date: 2017-11-10 11:03:48
---


### Vim 日常使用技巧

本文介绍了在使用Vim的过程中一些必要的指令和注意事项，以及一些插件。

<!--more-->

#### 安装

```
$ sudo apt-get install vim //Ubuntu
```

其他平台自行google。

#### 新手上路

移动光标:

```
# hjkl
# 2w 向前移动两个单词
# 3e 向前移动到第三个单词的末尾
# 0 移动到当前行的行首
# $ 移动到当前行的末尾
# gg 移动到当前文件的第一行行首
# G 移动到当前文件的最后一行行首
# 行号+G 移动到当前文件，指定行行首
# <ctrl>+o 跳转回之前的位置
# <ctrl>+i 返回跳转之前的位置
```

退出：

```
# <esc> 进入正常模式
# :q! 不保存退出
# :wq 保存后退出
```

删除：

```
# x 删除当前字符
# dw 删除至当前单词的末尾
# de 删除当前字符，及后一个单词
# d$ 删除至当前行尾
# dd 删除整行
# 2dd 删除两行
```

修改：

```
# i 插入文本
# A 当前行末尾添加
# r 替换当前字符
# o 打开新的一行并进入插入模式
```

撤销：

```
# u 撤销
# <ctrl>+r 取消撤销
```

复制、粘贴、剪切

```
# v 进入可视模式
# y 复制
# p 粘贴
# yy 复制当前行
# dd 剪切当前行
```

状态：

```
# <ctrl>+g 显示当前行以及文件信息
```

查找

```
# / 正向查找（n 继续查找，N 相反方向查找）
# ? 逆向查找
# :set ic 忽略大小写
# :set noic 取消忽略大小写
# :set hls 匹配项高亮显示
# :set nohls 取消匹配项高亮显示
# :set is 显示部分匹配
```

替换：

```
# :s/old/new 替换该行第一个匹配串
# :s/old/new/g 替换全行的匹配串
# :%s/old/new/g 替换整个文件的匹配串
```

执行外部命令

```
# :!shell 执行外部命令
```

#### .vimrc

`.vimrc`是Vim的配置文件，需要我们自己创建

```
cd ~  # 进入当前用户的根目录
touch .vimrc  # 新建配置文件

# 安装vim-plug来管理插件
# Unix
# vim-plug
# Vim
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
# Neovim
curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

#### 基本配置

取消备份

```
set nobackup
set noswapfile
```

文件编码

```
set encoding=utf-8
```

显示行号

```
set number
```

取消换行

```
set nowrap
```

显示光标当前位置

```
set ruler
```

设置缩进

```
set cindent

set tabstop=4
set shiftwidth=4
```

突出显示当前行

```
set cursorline
```

左下角显示当前 vim 模式

```
set showmode
```

代码折叠

```
# 启动vim时关闭折叠代码
set nofoldenable
```

待补充。。。
