# pytorch NLP入门  复现一些NLP论文/传统机器学习算法

## contributors
- 安洋  ps: 目前就一个人，疯狂暗示

## 项目介绍 update 2021.2.24
- 项目持续更新中
- 本项目主要用于学习nlp知识，中间会用到一些网上的代码，主要力求简单简洁
- 因为本人刚刚开始学习nlp不久，主要是希望和一些小伙伴通过coding快速学习入门
- 因为学生训练资源不足，所以尽量使训练数据简单，本项目很多实例都没有数据集，仅仅有一个字符串，因此会牺牲一些效果达不到自己心中的想法 ps:这方面还在努力想办法
- 并不注重调参，主要在于模型实现和优缺点理解

## 后续改进
- 统一使用dataset和dataloader
- 统一使用tensorboard

## 要求
- 代码风格统一，函数不冗余，尽量使用官方工具函数，在保持原生态实现的情况下尽量保持代码的精简，库版本统一
- 代码注释齐全，重要部分注明论文出处和公式来源
- 测试数据集下载方便，方便运行，不用在环境搭建上花费太多时间

## 实现

## 第一章 Word2Vec's SkipGram NegativeSampling
- [skip-gram](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/nlp/word2vec_skip_gram.py)
- [负样本采样 negative sampling](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/nlp/word2vec_negative_sampling.py) 	
 - [论文](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
## 第二章 lstm
- [lstm](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/nlp/lstm.py)  
    - [论文: LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [lstm + attention](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/nlp/lstm_with_attention.py)
## 第三章 transformer+attention
- [transformer](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/nlp/transformer.py)
    - [论文: Attention is All you need](https://arxiv.org/pdf/1706.03762.pdf)
## 第四章 机器学习相关算法(西瓜书+李航线性 待补充 qwq)
- [交叉熵 nn.CrossEntropyLoss](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/machine_learning/cross_entropyloss.py)
- [svm(smo算法)](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/machine_learning/svm.py)
- [svd奇异值分解](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/machine_learning/svd.py)
- [svd图片压缩](https://github.com/ssw-nlp-study-group/nlp_study/blob/main/machine_learning/svd_img_compress.py)