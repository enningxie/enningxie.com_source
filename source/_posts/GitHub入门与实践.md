---
title: GitHub入门与实践
date: 2017-11-06 15:57:56
tags:
    - GitHub
categories: 技术
---

### GitHub入门与实践

欢迎加入[GitHub](https://github.com),组织需要你 :)

<!--more-->

#### 第一章　欢迎来到GitHub的世界

##### 1.1 什么是GitHub

GitHub是为开发者提供Git仓库的托管服务。这是一个让开发者与朋友、同事、同学、及陌生人共享代码的完美场所。

> **GitHub与Git的区别**  
> 首先，GitHub与Git是完全不同的两个东西。  
> 在Git中，开发者将源代码存入名为“Git仓库”的资料库中并加以使用。而GitHub则是在网络上提供Git仓库的一项服务。也就是说GitHub上公开的软件源代码全部由Git进行管理。


##### 1.2 使用GitHub会带来哪些变化

Pull Request功能：是指开发者在本地对源码进行修改后，向GitHub中托管的Git仓库请求合并的功能。

开发者可以在Pull Request上通过评论交流。通过这个功能，开发者可以轻松更改源码，并公开更改的细节，然后向仓库提交合并请求。而且，如果请求的更改与项目的初衷相违背，也可以选择拒绝合并。

GitHub的Pull Request不但能轻松查看源代码的前后差别，还可以对指定的一行代码进行评论。

任务管理和BUG报告可以通过Issue进行交互。

##### 1.3 GitHub提供的主要功能

- Git仓库  
一般情况下，我们可以免费建立任意个GitHub提供的Git仓库。但如果需要建立只对特定人或只对自己公开的私有仓库，则需要付费。

- Organization  
通常来说，个人使用时只要使用个人账户就足够了，但如果是公司，建议使用Organization账户。它的优点在于可以统一管理账户和权限，还能统一支付一些费用。如果只使用公开仓库，是可以免费创建Organization账户的。

- Issue  
Issue功能，是将一个任务或问题分配给一个Issue进行追踪和管理的功能。可以像BUG管理系统一样使用。在GitHub上，每当我们Pull Request的同时，往往都会创建一个Issue。每一个功能更改或修正都对应一个Issue，讨论或修改都以这个Issue为中心进行。只要查看这个Issue，就能知道和这个更改相关的一切信息，并以此进行管理。在Git的提交信息中写上Issue的ID（例如“#7”），GitHub就会自动生成从Issue到对应提交的链接。另外，只要按照特定的格式描述提交信息，还可以关闭Issue。

- Wiki  
通过Wiki功能，任何人都能随时对一篇文章进行更改并保存，因此可以多人共同完成一篇文章。该功能常用于编写开发文档或手册中。

- Pull Request　　
开发者向GitHub的仓库推送更改或功能添加后，可以通过Pull Request功能向别人的仓库提出申请，请求对方合并。Pull Request送出后，目标仓库的管理者等人将能够查看Pull Request的内容及其中包含的代码更改。同时，GitHub还提供了对Pull Request和源代码前后差别进行讨论的功能。通过此功能，可以以行为单位对源代码添加评论，让程序员之间可以高效的交流。

#### 第二章　Git的导入

Git仓库管理功能是GitHub的核心。

##### 2.1 版本管理

Git属于分散型版本管理系统，是为版本管理而设计的软件。

GitHub将仓库Fork给了每一个用户，Fork就是将GitHub的某个特定仓库复制到自己的账户下。Fork出的仓库与原仓库是两个不同的仓库，开发者可以随意编辑。

分散型拥有多个仓库，由于本地的开发环境中就有仓库，所以开发者不必连接远程仓库就可以进行开发。

##### 2.2 Git的安装

该部分请自行google。

##### 2.3 初始设置

- 设置姓名和邮箱地址  
``` Python
$ git config --global user.name "Firstname Lastname"
$ git config --global user.email "your_email@example.com"
```
这个命令，会在“~/.gitconfig”中以如下的形式输出设置文件。
```Python
[user]
  name = Firstname Lastname
  email = your_email@example.com
```
想更改这些信息时，可以直接编辑这个设置文件。这里设置的姓名和邮箱地址会用在Git的提交日志中。由于在GitHub上公开仓库时，这里的姓名和邮箱地址也会随着日志一同被公开，所以请不要使用不便公开的隐私信息。

- 提高代码可读性  
将`color.ui`设置为`auto`可以让命令的输出拥有更高的可读性。
```Python
$ git config --global color.ui auto
```
`~/.gitconfig`中会增加下面一行。
```
[color]
ui = auto
```
这样一来，各种命令的输出就会变得更容易分辨。

#### 第三章 使用GitHub的前期准备

##### 3.1 使用前的准备

- 创建账户  
自行创建。

- 设置SSH Key  
GitHub上连接已有仓库时的认证，是通过使用了SSH的公开密匙认证方式进行的。  
创建SSH Key:`ssh-keygen -t rsa -C "your_email@example.com"`,`your_email@example.com`的部分请改成你在创建账户时用的邮箱地址。  
`id_rsa`文件是私有密钥，`id_rsa.pub`是公开密钥。

- 添加公开密钥  
在GitHub中添加公开密钥，今后就可以用私有密钥进行认证了。  
完成以上设置之后，就可以用手中的私人密钥与GitHub进行认证和通信了。  

##### 3.2 其它说明

- `.gitignore`文件  
帮助我们把不需要在Git仓库中进行版本管理的文件记录在`.gitignore`文件中，省去了每次根据框架进行设置的麻烦。

- `README.md`文件  
`README.md`文件的内容会自动显示在首页当中。因此，人们一般会在这个文件中标明本仓库所包含的软件的概要、使用流程、许可协议等信息。

- `git clone`  
clone已有的仓库源代码。

- 编写代码

- 提交  
通过`git add`命令将文件加入暂存区，再通过`git commit`命令提交。  
添加成功后，可以通过`git log`查看提交日志。

- 进行push  
最后进行`git push`之后，GitHub上的仓库就会被更新。

#### 第4章 通过实际操作学习Git

##### 4.1 基本操作

- `git init`初始化仓库  
要使用Git进行版本管理，必须先初始化仓库。Git使用`git init`命令进行初始化仓库操作。如果初始化成功，当前目录下就会生成`.git`目录。这个目录里存储着管理当前内容所需的仓库数据。

- `git status`查看仓库的状态  
`git status`命令用于显示Git仓库的状态。

- `git add`向暂存区中添加文件  
要想文件成为Git仓库的管理对象，就需要用`git add`命令将其加入暂存区中，暂存区是提交之前的一个临时区域。

- `git commit`保存仓库的历史记录  
`git commit`命令可以将当前暂存区中的文件实际保存到仓库的历史记录中。通过这些记录，我们就可以在工作树中复原文件。  
`git commit -m 'message'`其中`-m`参数后面记录了提交信息。

- `git log`查看提交后的日志  
`git log`命令可以查看以往仓库中提交的日志。包括可以查看什么人在什么时候进行了提交或合并，以及操作前后有怎样的差别。  
显示文件的改动：如果想查看提交所带来的改动，可以加上`-p`参数，文件的前后差别就会显示在提交信息之后。    
只显示指定目录、文件的日志： 只要在`git log`命令之后加上目录名，便会只显示该目录下的日志。如果加的是文件名，就会只显示与该文件相关的日志。  
`git log -p README.md`就可以只查看README.md文件的提交日志以及提交前后的差别。

- `git diff`查看更改前后的差别  
`git diff`命令可以查看工作树、暂存区、最新提交之间的差别。  
查看与最新提交的差别：`git diff HEAD`。  
不妨养成一个好习惯：在执行`git commit`命令之前，先执行`git diff HEAD`命令，查看本次提交和上次提交之间有什么差别，等确认完毕之后再进行提交。这里的`HEAD`是指当前分支中最新一次提交的指针。

##### 4.2 分支的操作

在进行多个并行作业时，我们会用到分支。在这类并行开发的过程中，往往同时存在多个最新代码状态。

不同分支中，可以同时进行完全不同的作业。等该分支的作业完成之后再与master分支合并。

通过灵活运用分支，可以让多人同时高效的进行并行开发。

- `git branch`显示分支一览表  
`git branch`命令可以将分支名列表显示，同时可以确认当前所在分支。

- `git checkout -b`创建、切换分支  
实际上`git branch`和`git checkout`命令也能收到同样的效果。  
`git checkout -`切换回上一个分支。

- 特性分支  
当今大部分工作流程中都用到了特性（topic）分支。  
特性分支顾名思义，是集中实现单一特性（主题），除此之外不进行任何作业的分支。在日常开发中，往往会创建数个特性分支，同时在此之外再保留一个随时可以发布软件的稳定分支。稳定分支的角色通常由master分支担当。  
基于特定主题的作业在特性分支中进行，主题完成后再与master分支合并。

-  `git merge`合并分支  
为了在历史记录中明确记录下本次分支合并，我们需要创建合并提交。因此，在合并时加上`--no--ff`参数。

- `git log --graph`以图表形式查看分支

##### 4.3 更改提交的操作

- `git reset`回溯历史版本  
要让仓库的HEAD、暂存区、当前工作树回溯到指定状态，需要用到`git reset --hard hash`命令。只要提供目标时间点的哈希值，就可以完全恢复至该时间点的状态。

- `git log`只能查看以当前状态为终点的历史日志。  
`git reflog`查看当前仓库的操作日志

- `git commit --amend`修改提交信息。

- `git rebase -i`  
在合并特性分支之前，如果发现已提交的内容中有些许拼写错误等，不妨提交一个修改，然后将这个修改包含到前一个提交中，压缩成一个历史记录。

- `git commit -am`可以实现`git add`和`git commit -m`两步操作。

##### 4.4 推送至远程仓库

- `git remote add`添加远程仓库  
`git remote add origin git@github.com:enningxie/test.git`

- `git push`推送至远程仓库  
如果想将当前分支下本地仓库中的内容推送给远仓库，需要用到`git push`命令。  
`git push -u origin master`像这样执行`git push`命令，当前分支的内容就会被推送给远程仓库origin的master分支，`-u`参数可以在推送的同时，将origin仓库的master分支设置为本地仓库当前分支的upstream（上游）。添加了这个参数，将来运行`git pull`命令从远程仓库获取内容时，本地仓库的这个分支就可以直接从origin的master分支获取内容，省去了另外添加参数的麻烦。

##### 4.5 从远程仓库获取

- `git clone`获取远程仓库  
执行`git clone`命令后我们会默认处于master分支下，同时系统会自动将origin设置成该远程仓库的标识符。

- `git branch -a`命令查看当前分支的相关信息。  
`-a`参数可以同时显示本地仓库和远程仓库的分支信息。

- 获取远程的new_branch分支  
`git checkout -b new origin/new_branch`其中`-b`参数是本地仓库中新建分支名称。

- `git pull`获取最新的远程仓库分支  
为了减少冲突情况的发生，建议频繁进行`push`和`pull`操作。
