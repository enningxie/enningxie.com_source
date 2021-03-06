---
title: opencv
tags: opencv
categories: 技术
comments: true
date: 2017-12-13 16:59:51
---


### 安装opencv

<!--more-->

- 安装必要的库

```
sudo apt-get install build-essential

sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

sudo apt-get install --assume-yes libopencv-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip

sudo apt-get install ffmpeg libopencv-dev libgtk-3-dev python-numpy python3-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev qtbase5-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev
```

- 下载opencv的两个包

```
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

- cmake

进第一个文件夹中，`mkdir build`，

```
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/Documents/Codes/opencv/opencv_contrib-3.2.0/modules \
    -D PYTHON_EXECUTABLE=~/anaconda3/envs/tensorflow02/bin/python \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_LAPACK=OFF \
    -D WITH_CUDA=OFF ..
```

- `make -j4`

无穷的等待。

报错请`make clean`、删除并重建build文件夹，google报错信息，fixed，然后重复3。

- `sudo make install`

- `pip install opencv-python`

done :).
