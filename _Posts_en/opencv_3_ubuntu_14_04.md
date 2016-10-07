---
layout: post
lang: en
post_name: OpenCv 3 and Python 2 in Ubuntu 14.04
long_name: false
title: Install and run opencv 3 with python 2 in ubuntu 14.04
long_title: true

image_name: python_opencv.jpg

date: 2014-08-02 14:23:23
tags: opencv python ubuntu tutorial installation
---
In this tutorial I will talk about opencv, opencv 3 installation with python 2.7 binding, and we will write/execute simple program.

## Table of Contents
+ [What is OpenCv]({{site.url | append: site.baseurl | append: page.url | append: '#what-is-opencv'}})
+ [Install OpenCv 3 For Python 2.7 In Ubuntu 14.04]({{site.url | append: site.baseurl | append: page.url | append: '#install-opencv-3-for-python-27-in-ubuntu-1404'}})
    + Install Dependency
    + Clone OpenCv
    + Compile From Source
+ [Write Simple Python Program]({{site.url | append: site.baseurl | append: page.url | append: '#write-simple-python-program'}})
<!--more-->

---
<p></p>

## What Is OpenCv
Opencv stand for **`Open Source Computer Vision`** is a **library** for computer vision and image processing. specially for **real-time** processing.  Its written in C++.  
You can use Opencv library in `C++`, `Python`, `Java`, etc with binding libraries.  
OpenCv can run on:

+ Linux
+ Windows
+ OS X
+ Android
+ And ...

---
<p></p>

## Install OpenCv 3 For Python 2.7 In Ubuntu 14.04
+ <big>**Install Dependency**</big>

Run following command to install dependency and some package:
{% highlight bash %}
sudo apt-get install build-essential cmake git libgtk2.0-dev python3-dev \
python-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
libjasper-dev libdc1394-22-dev python-numpy python-opencv
{% endhighlight %}

+ <big>**Clone OpenCv**</big>

{% highlight bash %}
git clone https://github.com/Itseez/opencv.git
cd opencv
mkdir release
cd release
{% endhighlight %}

+ <big>**Compile From Source**</big>

{% highlight bash %}
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_opencv_java=OFF -D PYTHON_LIBRARY=/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so -D PYTHON_INCLUDE_DIR=/usr/include/python2.7 -DPYTHON_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python2.7m -D PYTHON_NUMPY_INCLUDE_DIRS=/usr/lib/python2.7/dist-packages/numpy/core/include/ ..
make -j8
sudo make install
{% endhighlight %}

**note**: if you got an error like this:

{% highlight bash %}
CMake Error at 3rdparty/ippicv/downloader.cmake:73 (file):
  file DOWNLOAD HASH mismatch

    for file: [/home/mlibre/opencv/opencv-3.1.0/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e/ippicv_linux_20151201.tgz]
      expected hash: [808b791a6eac9ed78d32a7666804320e]
        actual hash: [d41d8cd98f00b204e9800998ecf8427e]
             status: [7;"Couldn't connect to server"]
{% endhighlight %}

You need to download **ippicv**. in this case [this file]()  
And copy this in the error mentioned folder. in this case:  
`/home/mlibre/opencv/opencv-3.1.0/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e`

Like this:

{% highlight bash %}
cp ippicv_linux_20151201.tgz /home/mlibre/opencv/opencv-3.1.0/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e/
{% endhighlight %}

Now run **cmake command** again.

---
<p></p>

## Write Simple Python Program

{% highlight python %}
import cv2

def show_image(image):
    cv2.imshow('Hi', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

if( __name__ == '__main__'):
    img = cv2.imread("girl.png")
    show_image(img)
{% endhighlight %}

And now run code from terminal:

{% highlight bash %}
python op.py
{% endhighlight %}

And image showed:

![girl image]({{site.url | append: site.baseurl | append: '/files/post_files' | append: '/girl.png'}})

Download `op.py` from [here]({{site.url | append: site.baseurl | append: '/files/post_files' | append: '/op.py'}}) and image from [here]({{site.url | append: site.baseurl | append: '/files/post_files' | append: '/girl.png'}}).

---
