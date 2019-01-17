语料的构建与清洗====》特征文件的构造==》 聚类工具

configparser.py:
    配置文件解析

generate_corpus.py:
    基于问答对的形式构建用户语料库集

generate_features.py:
    特征构造,目前支持tfidf、词向量加权平均、基于相似度模型的

util.py:
    工具包 包括预处理、读文件等

cluster.py:
    聚类算法，目前支持基于tensorflow的kmeans(距离)以及hdbscan算法(密度)

sim文件夹:
    相似度模型

sensitive data文件夹:
    敏感词文件

config文件夹:
    配置文件，特征构造的配置文件、聚类算法的配置文件

data文件夹:
    语料文件、特征文件、word2vec文件等
