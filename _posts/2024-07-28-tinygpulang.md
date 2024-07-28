---
title: "TinyGPUlang: tutorial on how to implement gpu backend with LLVM"
categories:
  - compiler
tags:
  - tinygpulang
last_modified_at: 2024-07-28T18:23:00-06:00
---

### Motivation

I became really interested in deep learning compilers (e.g. TVM) and I am implementing one to understand them better.
The goal of deep learning compilers is to speed up neural networks for a given hardware. 
AI is known to be resource hungry both for training and inference. 
So any technology, capable of improving the costs and response time of AI system is promising.

Several languages and compilers are built with the help of LLVM, which is a compiler framework.
However, its documentation tends to be superficial.

The goal of my tutorial is to give simple and easy to understand guidance how to implement
a compiler backedn for gpus with LLVM.
For the sake of completness, I also created a simple programming language as frontend.
The tutorial is available on github.

<a href="https://github.com/adamtiger/tinyGPUlang" target="_blank" class="btn btn-success"><i class="fa fa-github fa-lg"></i> More details on GitHub</a>

