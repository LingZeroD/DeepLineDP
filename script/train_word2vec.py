import os, sys

from gensim.models import Word2Vec

import more_itertools

from DeepLineDP_model import *
from my_util import *

proj_names = list(all_train_releases.keys())
def train_word2vec_model(dataset_name, embedding_dim=50):
    w2v_path = get_w2v_path()

    save_path = w2v_path + '/' + dataset_name + '-' + str(embedding_dim) + 'dim.bin'

    # 如果已经存在了，就不用再训练了
    if os.path.exists(save_path):
        print('word2vec model at {} is already exists'.format(save_path))
        return

    # 如果不存在，就创建文件夹
    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)
    # 获取当前训练集
    train_rel = all_train_releases[dataset_name]

    # 获取当前训练集的数据
    train_df = get_df(train_rel)

    # 获取当前训练集的数据
    train_code_3d, _ = get_code3d_and_label(train_df, True)

    all_texts = list(more_itertools.collapse(train_code_3d[:], levels=1))  # 将3维数组转换为二维数组

    word2vec = Word2Vec(all_texts, size=embedding_dim, min_count=1, sorted_vocab=1)

    word2vec.save(save_path)
    print('save word2vec model at path {} done'.format(save_path))


# p = sys.argv[1]

# train_word2vec_model(p, 50)
for proj in proj_names:
    train_word2vec_model(proj, 50)
    print('finish', proj)
# train_word2vec_model("hbase", 50)
