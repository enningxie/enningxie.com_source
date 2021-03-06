---
title: Linux相关
tags: Linux
categories: Linux
comments: true
---

### Linux 相关

<!--more-->

#### 新手必须掌握的Linux命令

Shell是命令行工具

Bash是终端解释器

常见的命令格式：`命令名称 [命令参数] [命令对象]`

`echo`命令：

用于在终端输出字符串或变量提取后的值。

`date`命令：

用于显示及设置系统时间或日期。

`reboot`命令：

用于重启系统。

`poweroff`：

用于关闭系统。

`wget`：

用于在终端下载网络文件，格式为`wget [参数] 下载地址`。

参数|作用
---|---
`-b`|后台下载模式
`-P`|下载到指定目录
`-t`|最大尝试次数
`-c`|断点续传
`-p`|下载页面内所有资源，包括图片、视频等
`-r`|递归下载

`ps`：

用于查看系统中的进程状态，格式为`ps [参数]`。

参数|作用
---|---
`-a`|显示所有进程（包括其他用户的进程）
`-u`|用户以及其他详细信息
`-x`|显示没有控制终端的进程

Linux系统中，进程常见的状态有5种：

- `R`（运行）：进程正在运行或运行队列中等待。

- `S`（中断）：进程处于休眠中，当某个条件形成后或者接收到信号时，则脱离该状态。

- `D`（不可中断）：进程不响应系统异步信号，即便用`kill`命令也不能将其中断。

- `Z`（僵死）：进程已经终止，但进程描述符依然存在，直到父进程调用`wait4()`系统函数后将进程释放。

- `T`（停止）：进程收到停止信号后停止运行。

> Linux系统中，命令参数有长短格式之分，长格式和长格式（`--help`）不能合并，长格式和短格式（`-a`）之间也不能合并，但短格式和短格式之间是可以合并的，合并后仅保留一个`-`即可。另外`ps`命令可以允许参数不加`-`，因此可以直接写成`ps aux`的样子。

`top`:

用于动态地监视进程活动与系统负载等信息，其格式为`top`。

`top`命令相当强大，能够动态地查看系统运维状态，完全将它看作Linux中的“强化版的windows任务管理器”。

`pidof`:

用于查询某个指定服务进程的PID值，其格式为`pidof [参数] [服务名称]`。

每个进程的进程号码值（PID）是唯一的，因此可以通过PID来区分不同的进程。如，可以使用以下命令来查询本机上的`sshd`服务的PID:

```
$ pidof sshd
```

`kill`：

用于终止某个指定的PID的服务进程，格式为`kill [参数] [PID]`。

`killall`：

用于终止某个指定名称的服务所对应的全部进程，格式为`killall [参数] [进程名称]`。

通常来讲，复杂的软件的服务程序会有多个进程同为用户提供服务，如果逐个去结束这些进程会比较麻烦，此时可以使用`killall`命令来批量结束某个服务程序带有的全部进程。

#### 系统状态检测命令

`ifconfig`：

用于获取网卡配置与网络状态等信息，格式为`ifconfig [网络设备] [参数]`。

使用`ifconfig`命令来查看本机当前的网卡配置与网络状态等信息时，其实主要查看的就是网卡名称、inet参数后面的IP地址、ether参数后面的网卡物理地址（又称MAC地址），以及RX、TX的接收数据包与发送数据包的个数及累计流量。

`uname`：

用于查看系统内核与系统版本等信息，格式为`uname [-a]`。

在使用`uname`时，一般会同`-a`参数配合使用来完整地查看当前系统的内核名称、主机名、内核发行版本、节点名、系统时间、硬件名称、硬件平台、处理器类型以及操作系统名称等信息。

要查看当前系统版本的详细信息，则需要查看`redhat-release`文件，`cat /etc/redhat-release`

`uptime`:

用于查看系统的负载信息，格式为`uptime`，负载越低越好，尽量不要长期超过1，在生产环境中不要超过5。

`free`:

用于显示当前系统中内存的使用量信息，格式为`free [-h]`。

使用`-h`参数以更人性化的方式输出当前内存的实时使用量信息。

`who`:

用于查看当前登入主机的用户终端信息，格式为`who [参数]`。

可以快速显示出所有正在登录本机的用户的名称以及他们正在开启的终端信息。

`last`:

用于查看所有系统的登录记录，格式为`last [参数]`。

使用`last`命令可以查看本机的登录记录。但是这些信息都是以日志文件的形式保存在系统中，因此hacker可以任意对内容进行篡改。

`history`:

用于显示历史执行过的命令，格式为`history [-c]`。

历史命令会被保存到用户的home目录中的`.bash_history`文件中。

> Linux系统中以点（.）开头的文件均代表隐藏文件，这些文件大多数为系统服务文件，可以用`cat`命令查看其文件内容。

`sosreport`:

用于收集系统配置及架构信息并输出诊断文档。

#### 工作目录切换命令

`pwd`:

用于显示用户当前的工作目录。

`cd`:

用于切换工作目录。

`ls`:

用于显示目录中的文件信息。

#### 文本文件编辑命令

Linux 系统中'一切都是文件'。

`cat`:

用于查看纯文本文件（内容较少）。

`more`:

用于查看纯文本文件（内容较多）。

空格键，回车键用于翻页。

`head`:

用于查看纯文本文件的前N行。eg.`head -n 20 enning.txt`

`tail`:

用于查看纯文本文件后N行，或持续刷新内容。

`tail`命令最强悍的功能是可以持续刷新一个文件的内容，当想要实时查看最新日志文件时，`tail -f xz.log`.

`tr`:

用于替换文本文件中的字符。

`wc`:

用于统计指定文本的行数、字数、字节数。

`stat`:

用于查看文件的具体存储信息和时间等信息（查看文件的状态）。

`cut`:

用于按‘列’提取文本字符。

`diff`:

用于比较多个文本文件的差异。

#### 文件目录管理命令

`touch`:

用于创建空白文件或设置文件的时间。

参数|作用
---|---
`-a`|仅修改“读取时间”（atime）
`-m`|仅修改“修改时间”（mtime）
`-d`|同时修改atime与mtime

`mkdir`:

用于创建空白的目录。

参数`-p`用来递归创建出具有嵌套叠层关系的文件目录。

`cp`:

用于复制文件或目录。`cp [参数] 源文件 目标文件`

- 如果目标文件是目录，则会把源文件复制到该目录中；

- 如果目标文件也是普通文件，则会询问是否要覆盖它；

- 如果目标文件不存在，则执行正常的复制操作。

`mv`:

用于剪切文件或将文件重命名。

`rm`:

删除操作，`-f`强制删除，`-r`删除目录。

`dd`:

用于按照指定大小和个数的数据块来复制文件或转换文件。

`file`:

用于查看文件的类型。

#### 打包压缩与搜索命令

`tar`:

用于对文件进行打包压缩或解压，`tar [参数] 文件`

参数|作用
---|---
`-c`|创建压缩文件
`-x`|解开压缩文件
`-t`|查看压缩包内有哪些文件
`-z`|用Gzip压缩或解压
`-j`|用bzip2压缩或解压
`-v`|显示压缩或解压过程
`-f`|目标文件名
`-p`|保留原始的权限与属性
`-P`|使用绝对路径来压缩
`-C`|指定解压到的目录

- `tar -czvf package_name.tar.gz path_dir`, 把指定的文件进行打包压缩

- `tar -xzvf package_name.tar.gz`, 解压指定压缩文件

- `tar -xzvf package_name.tar.gz -C path_dir`, 解压到指定目录

`grep`

用于在文本中执行关键词搜索，并显示匹配的结果，格式`grep [选项] [文件]`。

参数|作用
---|---
`-c`|仅显示找到的行数
`-i`|忽略大小写
`-n`|显示行号
`-v`|反向选择--仅列出没有关键词的行

`find`

用于按照指定条件来查找文件，格式`find [查找路径] 寻找条件 操作`。

参数很多，推荐`man find`。

#### 管道符、重定向与环境变量

输入重定向是指把文件导入到命令中；

输出重定向是指把原本要输出到屏幕的数据信息写入到指定文件中。

输入重定向中用到的符号及其作用：

符号|作用
---|---
命令 < 文件|将文件作为命令的标准输入
命令 << 分界符|从标准输入中读入，直到遇到分界符才停止
命令 < 文件1 > 文件2|将文件1作为命令的标准输入并将标准输出到文件2

输出重定向中用到的符号及其作用：

符号|作用
---|---
命令 > 文件|将标准输出重定向到一个文件中（清空原有文件的数据）
命令 2> 文件|将错误输出重定向到一个文件中（清空原有文件的数据）
命令 >> 文件|将标准输出重定向到一个文件中（追加到原有内容的后面）
命令 2>> 文件|将错误输出重定向到一个文件中（追加到原有内容的后面）
命令 >> 文件 2>&1 或 命令 &>> 文件|将标准输出与错误输出共同写入到文件中（追加)

对于重定向中的标准输出模式，可以省略文件描述符1不写，而错误输出模式的文件描述符2是必须要写的。

---

管道命令符的作用可以用一句话来概况：“把前一个命令原本要输出到屏幕的数据当作是后一个命令的标准输入”。

---

变量是计算机系统中用于保存可变值的数据类型。在Linux系统中，变量名称一般都是大写的，这是一种约定俗成的规范。我们可以直接通过变量名称来提取到对应的变量值。Linux系统中的环境变量是用来定义系统运行环境的一些参数。

用户执行了一条命令之后，Linux执行分为4步：

- 第1步：判断用户是否以绝对路径或相对路径的方式输入命令（如`/bin/ls`），是的话就直接执行。

- 第2步：Linux系统检查用户输入的命令是否为“别名命令”，即用一个自定义的命令名称来替代原本的命令名称。可以用`alias`命令来创建一个属于自己的命令别名，格式为`alias 别名=命令`。若要取消一个命令别名，则是用`unalias 别名`。

- 第3步：Bash解释器判断用户输入的是内部命令还是外部命令。内部命令是解释器内部的指令，会直接执行；而用户在绝大部分时间输入的是外部命令，这些命令由步骤4继续处理。可以使用`type 命令名称`来判断用户输入的命令是内部命令还是外部命令。

- 第4步：系统在多个路径中查找用户输入的命令文件，而定义这些路径的变量叫做`PATH`，可以简单地把它理解成是“解释器的小助手”，作用是告诉Bash解释器待执行的命令可能存放的位置，然后Bash解释器就会在这些位置逐个查找。`PATH`是由多个路径值组成的变量，每个路径值之间用冒号间隔，对这些路径的增加和删除操作将影响到Bash解释器对Linux命令的查找。

在接手一台Linux系统后一定会在执行命令前先检查`PATH`变量中是否有可疑的目录，我们可以用`env`命令来查看Linux系统中所有的环境变量。

#### Vim编辑器与Shell命令脚本

Linux系统中一切都是文件，而配置一个服务就是在修改其配置文件的参数。

Vim是一款很棒的文本编辑器。

其中有三个模式，日常操作过程中会在这三个模式下进行切换：

- 命令模式：控制光标移动，可对文本进行复制、粘贴、删除和查找等工作。

- 输入模式：正常的文本录入。

- 末行模式：保存或退出文档，以及设置编辑环境。

Vim中常用的命令：

命令|作用
---|---
`dd`|删除（剪切）光标所在的整行
`5dd`|删除（剪切）从光标处开始的5行
`yy`|复制光标所在的整行
`5yy`|复制从光标处开始的5行
`n`|显示搜索命令定位到的下一个字符串
`N`|显示搜索命令定位到的上一个字符串
`u`|撤销上一步的操作
`p`|将之前删除（`dd`）或复制（`yy`）过的数据粘贴到光标后面

末行模式主要用于保存或退出文件，以及设置Vim编辑器的工作环境，还可以让用户执行外部的Linux命令或跳转到所编写文档的特定行数。

末行模式中可用的命令：

命令|作用
---|---
`:w`|保存
`:q`|退出
`:q!`|强制退出（放弃对文档的修改内容）
`:wq!`|强制保存退出
`:set nu`|显示行号
`:set nonu`|不显示行号
`:命令`|执行该命令
`:整数`|跳转到该行
`:s/one/two`|将当前光标所在行的第一个one替换成two
`:s/one/two/g`|将当前光标所在行的所有one替换成two
`:%s/one/two/g`|将全文中的所有one替换成two
`?字符串`|在文本中从下至上搜索该字符串
`/字符串`|在文本中从上至下搜索该字符串

##### 配置主机名称

修改`/etc/hostname`文件中的内容即可。

##### 编写Shell脚本

Shell脚本命令的工作方式有两种：交互式和批处理。

- 交互式（Interactive）：用户每输入一条命令就立即执行。

- 批处理（Batch）：由用户事先编好一个完整的Shell脚本，Shell会一次性执行脚本中诸多的命令。

查看命令行终端解释器：`# echo $SHELL`.

Shell脚本文件的名称可以任意，但为了避免被误以为是普通文件，建议将`.sh`后缀加上。

给出一个`example.sh`:

```
#!/bin/bash
#example
pwd
ll
```

解释：第一行的（`#!`）脚本声明，用来告诉系统使用哪个Shell解释器来执行该脚本；第二行是介绍说明。

运行脚本的方式：

- `bash example.sh`

- `./example.sh`

**接收用户的参数：**`$0`对应的是当前Shell脚本程序的名称，`$#`对应的是总共有几个参数，`$*`对应的是所有位置的参数值，`$?`对应的是显示上一次命令的执行返回值，而`$1`/`$2`...分别对应的是第N个位置的参数值。

`example_01.sh`:

```
#!/bin/bash
echo "脚本名称：$0"
echo "总共参数：$#，分别是$*"
echo "第一个参数$1"
```
### shell 脚本、 定时任务 todo p114

#### 用户身份与文件权限
