---
title: "Transformers In Deep Learning"
date: 2024-02-17T20:19:31Z
draft: false
author: "Pinaki Pani"
description: "Transformers in Sequential Attention"
feature_image: "https://images.unsplash.com/photo-1560700105-7a3450fd2531?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
---

# Transformers In Self and Cross Attention

Initially the transformers began for sequential inputs only where they are later encoded with embeddings. These embeddings are not just encoder embeddings but also include the positional and class token embeddings. These embeddings later allow the transformers to identify each and every input and then associate them with other inputs leading to contextual analysis. The transformer first goes through the input enbedding layer that has all the elements in the input. These elements in the input are associated with positional embeddings that allows to identify the location of the input element within the sequence. This positional embeddings are an additive element to the input embeddings. The original paper can be [referenced here,](https://arxiv.org/pdf/1706.03762.pdf)
