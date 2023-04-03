import re

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

max_seq_len = 50  # word2vec最大序列长度
# 训练项目目录
all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6',
                      'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0',
                      'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'}
# 预测项目目录
all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'],
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'],
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}

# 所有项目目录
all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
                'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
                'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
                'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/preprocessed_data/'

# word2vec 模型保存路径
word2vec_dir = '../output/Word2Vec_model/'


def get_df(rel, is_baseline=False):
    """
    获取处理后的数据集
    """
    if is_baseline:
        df = pd.read_csv('../' + file_lvl_gt + rel + ".csv")

    else:
        df = pd.read_csv(file_lvl_gt + rel + ".csv")

    df = df.fillna('')

    df = df[df['is_blank'] == False]
    df = df[df['is_test_file'] == False]
    df = df[df['is_comment'] == False]

    return df


def prepare_code2d(code_list, to_lowercase=False):
    '''
        input
            code_list (list): list that contains code each line (in str format)
        output
            code2d (nested list): a list that contains list of tokens with padding by '<pad>'
    '''
    code2d = []
    # 遍历每行
    for c in code_list:
        c = re.sub('\\s+', ' ', c)

        # 小写
        if to_lowercase:
            c = c.lower()

        # 分词
        token_list = c.strip().split()
        # 词数
        total_tokens = len(token_list)

        # 获取最大序列长度
        token_list = token_list[:max_seq_len]

        # 如果词数小于最大序列长度，补齐
        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)
        # 添加到二维列表
        code2d.append(token_list)

    return code2d


def get_code3d_and_label(df, to_lowercase=False):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''

    code3d = []
    all_file_label = []

    # group by filename
    for filename, group_df in df.groupby('filename'):
        # 唯一值
        file_label = bool(group_df['file-label'].unique())

        # 代码行
        code = list(group_df['code_line'])

        # 代码行转换为二维列表
        code2d = prepare_code2d(code, to_lowercase)

        # 添加到三维列表
        code3d.append(code2d)

        # 添加到文件级别标签列表
        all_file_label.append(file_label)

    return code3d, all_file_label


def get_w2v_path():
    """
     word2vec 模型保存路径
    """
    return word2vec_dir


def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    #获取word2vec模型的权重
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.syn0).cuda()

    # 添加一个全0的向量，用于填充
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1, embed_dim).cuda()))


    return word2vec_weights


def pad_code(code_list_3d, max_sent_len, limit_sent_len=True, mode='train'):
    paded = []

    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)

        if mode == 'train':
            if max_sent_len - len(file) > 0:
                for i in range(0, max_sent_len - len(file)):
                    sent_list.append([0] * max_seq_len)

        if limit_sent_len:
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)

    return paded


def get_dataloader(code_vec, label_list, batch_size, max_sent_len):
    # 将标签转换为tensor
    y_tensor = torch.cuda.FloatTensor([label for label in label_list])
    # 填充
    code_vec_pad = pad_code(code_vec, max_sent_len)
    # 转换为tensor
    tensor_dataset = TensorDataset(torch.tensor(code_vec_pad), y_tensor)

    # 获取dataloader
    dl = DataLoader(tensor_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    return dl


def get_x_vec(code_3d, word2vec):
    # 将三维列表转化为三维向量列表
    x_vec = [
        [[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text]
         for text in texts] for texts in code_3d]

    return x_vec
