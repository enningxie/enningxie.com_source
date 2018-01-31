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

---

update: 以下是anaconda3下的cmake语句，亲测有效

```
cmake -D CMAKE_BUILD_TYPE=RELEASE     -D CMAKE_INSTALL_PREFIX=/home/enningxie/Documents/Codes/opencv/opencv-3.2.0     -D INSTALL_C_EXAMPLES=OFF     -D INSTALL_PYTHON_EXAMPLES=OFF     -D OPENCV_EXTRA_MODULES_PATH=/home/enningxie/Documents/Codes/opencv/opencv_contrib-3.2.0/modules     -D BUILD_EXAMPLES=OFF     -D BUILD_opencv_python2=OFF     -D WITH_FFMPEG=1     -D WITH_CUDA=0     -D PYTHON3_EXECUTABLE=/home/enningxie/anaconda3/envs/tensorflow02/bin/python     -D PYTHON_INCLUDE_DIR=/home/enningxie/anaconda3/envs/tensorflow02/include/python3.5m     -D PYTHON_INCLUDE_DIR2=/home/enningxie/anaconda3/envs/tensorflow02/include/python3.5m     -D PYTHON_LIBRARY=/home/enningxie/anaconda3/envs/tensorflow02/lib/libpython3.5m.so     -D PYTHON3_PACKAGES_PATH=/home/enningxie/anaconda3/envs/tensorflow02/lib/python3.5     -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/enningxie/anaconda3/envs/tensorflow02/lib/python3.5/site-packages/numpy/core/include ..
```
```
make -j8
make install
```

对于错误：

```
 fatal error: LAPACKE_H_PATH-NOTFOUND/lapacke.h: No such file or directory #include "LAPACKE_H_PATH-NOTFOUND/lapacke.h"
```

解决：

```
sudo apt-get install liblapacke-dev checkinstall
```

Uninstalling OpenCV

```
sudo apt-get purge libopencv*

sudo dpkg -r opencv
```

Then, go to the build directory and

```
sudo make uninstall
```

---

"update"

```
cmake -D CMAKE_BUILD_TYPE=RELEASE     -D CMAKE_INSTALL_PREFIX=/usr/local     -D INSTALL_C_EXAMPLES=OFF     -D INSTALL_PYTHON_EXAMPLES=OFF     -D OPENCV_EXTRA_MODULES_PATH=/home/enningxie/Documents/Codes/opencv/opencv_contrib-master/modules     -D BUILD_EXAMPLES=OFF     -D BUILD_opencv_python2=OFF     -D WITH_FFMPEG=1     -D WITH_CUDA=0     -D PYTHON3_EXECUTABLE=/home/enningxie/anaconda3/envs/tensorflow02/bin/python     -D PYTHON_INCLUDE_DIR=/home/enningxie/anaconda3/envs/tensorflow02/include/python3.5m  -D WITH_GTK=ON   -D PYTHON_INCLUDE_DIR2=/home/enningxie/anaconda3/envs/tensorflow02/include/python3.5m     -D PYTHON_LIBRARY=/home/enningxie/anaconda3/envs/tensorflow02/lib/libpython3.5m.so     -D PYTHON3_PACKAGES_PATH=/home/enningxie/anaconda3/envs/tensorflow02/lib/python3.5     -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/enningxie/anaconda3/envs/tensorflow02/lib/python3.5/site-packages/numpy/core/include ..
```
