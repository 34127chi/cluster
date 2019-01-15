import sys
import pdb

def aveg_emb_feature(params):
    '''
    句子中字向量的平均
    '''
    from util import read_vectors, batch_iter, read_file
    import numpy as np
    from sklearn import metrics
    import sklearn.datasets as data
    from hdbscan import HDBSCAN
    import pdb
    import pickle
    
    content = read_file(oarams.train_file)

    word_all_data = []
    for line in content:
        word_all_data.extend(list(line))

    from collections import Counter
    word_counter = Counter(word_all_data)#词的频率统计
    word_count_pairs = word_counter.most_common(params.vocab_size - 1)#取vab_size个频率最高的词
    words, _ = list(zip(*word_count_pairs))
    words = ['<PAD>', '<UNK>'] + list(words)
    word2id = dict(zip(words, range(len(words))))
    id2word = dict(zip(range(len(words)), words))
    vocab_size = len(words)

    word_id = []
    for i in range(len(content)):
        tmp_word_id = np.zeros((params.sentence_length,), dtype=int)
        for word_idx, word in enumerate(list(content[i])):
            if word_idx >= params.sentence_length:
                break
            tmp_word_id[word_idx] = word2id.get(word, 1)
        word_id.append(tmp_word_id)

    word_id = np.asarray(word_id).astype(int)

    pickle.dump(word_id, open(params.train_id_file, 'wb'))

    import tensorflow as tf
    word_input_x = tf.placeholder(tf.int32, [None, params.sentence_length], name='word_input_x')
    word_embedding = tf.get_variable('word_embedding', [vocab_size, params.embedding_dim]) 
    word_embedding_inputs = tf.nn.embedding_lookup(word_embedding, word_input_x)
    aveg_emb = tf.reduce_mean(word_embedding_inputs, 1)

    sess = tf.Session()

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    embedding_2dlist[0] = np.zeros(params.embedding_dim)  # assign empty for first word:'PAD'
    word2vec_dict = read_vectors(params.vocab_embedding)
    for i in range(1, vocab_size):  # loop each word
        word = id2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word][:params.embedding_dim]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            embedding_2dlist[i] = embedding
        else:  # no embedding for this word
            embedding_2dlist[i] = np.random.uniform(-bound, bound, params.embedding_dim);
            #embedding_2dlist[i] = np.full(embedding_dim, 1.0)
    embedding_final = np.array(embedding_2dlist)  # covert to 2d array.
    embedding_matrix = tf.constant(embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(word_embedding, embedding_matrix)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)

    aveg_emb_total = []
    for batch in batch_iter(word_id, params.batch_size):
        tmp_aveg_emb = sess.run(aveg_emb, feed_dict = {word_input_x:batch})
        aveg_emb_total.extend(tmp_aveg_emb.tolist())

    pdb.set_trace()

    pickle.dump(aveg_emb_total, open(params.aveg_emb_file, 'wb'))
    pass

def similiar_emb_feature(params):
    '''
    相似句模型产生的向量
    '''
    from sim.classfify import ClassifyModel
    import pdb
    import pickle
    from util import read_file
    content = read_file(params.train_file)
    similiar_emb_total = [] 
    model = ClassifyModel(params, params.model_file, params.vocab_file, params.stopword_file)
    for line in content:
        vec, _ = model.get_sentence_vector(line)
        similiar_emb_total.append(vec.tolist())
    pickle.dump(similiar_emb_total, open(params.similiar_emb_file, 'wb'))
    pass

def tfidf_feature(params):
    '''
    基于字的tfidf产生的向量
    '''
    import numpy as np
    from sklearn import metrics
    import sklearn.datasets as data
    from sklearn.decomposition import TruncatedSVD
    from hdbscan import HDBSCAN
    import pdb
    import pickle
    from util import read_file
    content = read_file(params.train_file)
    seg_list = []
    for line in content:
        seg_list.append(list(line))
    seg_str = []
    for line in seg_list:
        seg_str.append(' '.join(line))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    vectorizer = TfidfVectorizer(analyzer = 'char')
    X = vectorizer.fit_transform(seg_str)
    X = TruncatedSVD(n_components=params.n_components).fit_transform(X)
    pickle.dump(X, open(params.tfidf_feature_file, 'wb'))
    pass

def usage():
    pass

if __name__ == '__main__':
    from configparser import ConfigParser 
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)
        
    if sys.argv[1] == 'similiar':
        #加载配置
        config = ConfigParser('./config/similiar_emb_feature.json') 
        similiar_emb_feature(config.model_parameters)
    elif sys.argv[1] == 'tfidf':
        config = ConfigParser('./config/tfidf_feature.json') 
        pdb.set_trace()
        tfidf_feature(config.model_parameters)
    elif sys.argv[1] == 'aveg':
        config = ConfigParser('./config/aveg_emb_feature.json') 
        aveg_emb_feature(config.model_parameters)
    else:
        pass
