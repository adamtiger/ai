---
layout: post
lang: en
post_name: Run executable files from bash
long_name: false
title: Run executable files from bash
long_title: false

image_name:

date: 2016-09-26 16:08:53
tags: bash executable tutorial
---
Hi.  
In this document we learn **run/execute** executable **file** in **bash**.

## Table Of Contents
* [Quick Guide to run an executable file in bash]({{site.url | append: site.baseurl | append: page.url | append: '#quick-guide-to-run-an-executable-file-in-bash'}})
* [Deep Understanding]({{site.url | append: site.baseurl | append: page.url | append: '#deep-understanding'}})
	* [Absolute and Relative address/path]({{site.url | append: site.baseurl | append: page.url | append: '#absolute-and-relative-addresspath'}})
    * [PATH environment variable]({{site.url | append: site.baseurl | append: page.url | append: '#path-environment-variable'}})
* [Three Ways to run executable files from bash]({{site.url | append: site.baseurl | append: page.url | append: '#three-way-to-run-executable-files-from-bash'}})

## Quick guide to run an executable file in bash
Consider we have an executable file named "exfi" in /home/user/exfi. follow these steps to run this file:
<!--more-->

&nbsp;&nbsp;&nbsp;&nbsp;1. set **execute permission** for this file:
{% highlight bash %}
chmod a+x /home/user/exfi
{% endhighlight %}

&nbsp;&nbsp;&nbsp;&nbsp;2. run file:
{% highlight bash %}
/home/user/exfi
{% endhighlight %}

## Deep understanding

### Absolute and Relative address/path
There are two methods of addressing.

* **relative addressing**
* **absolute/direct addressing**


**Relative addressing** means that we enter/address to a folder/file, **relative to** where we are. for example consider we are in **/home/** and there is a folder named **user**.  
we can **enter** to this folder(user) in this way:
{% highlight bash %}
cd user
{% endhighlight %}

or back to **parent** directory:
{% highlight bash %}
cd ..
{% endhighlight %}

**Absolute/direct addressing** means that we enter/address to a folder/file, from "/" directory. for example consider we are in **/home/** and there is a folder named **user**.  
we can enter to folder "user" with "cd" command:
{% highlight bash %}
cd /home/user
{% endhighlight %}

### PATH environment variable
bash have an **environment variable** named **PATH**. this variable save number of folders **paths**.  
When we enter a **command** in bash, bash search this command in these **paths**.
## Three way to run executable files from bash
There are Three way we can run an executable file on the bash.  

1. **append** executable file address/path to **PATH** environment variable, and run executable file.  
Consider the "Quick guide" example, we can append /home/exfi address/path to PATH variable like this:
{% highlight bash %}
    export PATH=$PATH:/home/user
{% endhighlight %}

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; now we can type "exfi" in bash. and **exfi** file is run. like this:
{% highlight bash %}
    exfi
{% endhighlight %}

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. enter **absolute/direct** executable file address/path and run executable file, like this:
{% highlight bash %}
    /home/user/exfi
{% endhighlight %}

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. enter **relative** executable file address/path and run executable file, like this:
consider we are in **/home**.
{% highlight bash %}
    ./user/exfi
{% endhighlight %}
