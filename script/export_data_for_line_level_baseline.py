import os

import pandas as pd
from tqdm import tqdm

from my_util import *

base_data_dir = '../datasets/preprocessed_data/'
base_original_data_dir = '../datasets/original/File-level/'

data_for_ngram_dir = '../datasets/n_gram_data/'
data_for_error_prone_dir = '../datasets/ErrorProne_data/'

# 所有训练集项目名
proj_names = list(all_train_releases.keys())


def export_df_to_files(data_df, code_file_dir, line_file_dir):
    for filename, df in tqdm(data_df.groupby('filename')):
        code_lines = list(df['code_line'])
        code_str = '\n'.join(code_lines)
        code_str = code_str.lower()
        line_num = list(df['line_number'])
        line_num = [str(l) for l in line_num]

        code_filename = filename.replace('/', '_').replace('.java', '') + '.txt'
        line_filename = filename.replace('/', '_').replace('.java', '') + '_line_num.txt'

        with open(code_file_dir + code_filename, 'w', encoding='utf-8') as f:
            f.write(code_str)

        with open(line_file_dir + line_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(line_num))


def export_ngram_data_each_release(release, is_train=False):
    file_dir = data_for_ngram_dir + release + '/'
    # 源代码目录
    file_src_dir = file_dir + 'src/'
    # 行号目录
    file_line_num_dir = file_dir + 'line_num/'

    # 如果目录不存在，创建目录
    if not os.path.exists(file_src_dir):
        os.makedirs(file_src_dir)
    # 如果目录不存在，创建目录
    if not os.path.exists(file_line_num_dir):
        os.makedirs(file_line_num_dir)
    # 读取处理后的数据
    data_df = pd.read_csv(base_data_dir + release + '.csv', encoding='latin')

    # get clean files for training only
    if is_train:  # 训练集
        data_df = data_df[
            (data_df['is_test_file'] == False) & (data_df['is_blank'] == False) & (
                    data_df['file-label'] == False)]  # 去除空行、测试得到无bug的文件

    # get defective files for prediction only
    else:
        data_df = data_df[
            (data_df['is_test_file'] == False) & (data_df['is_blank'] == False) & (
                    data_df['file-label'] == True)]  # 去除空行、测试得到有缺陷的文件

    data_df = data_df.fillna('')  # 填充空值

    export_df_to_files(data_df, file_src_dir, file_line_num_dir)  # 导出数据到文件本地存储


def export_data_all_releases(proj_name):
    # 训练目录
    train_rel = all_train_releases[proj_name]
    # 测试目录
    eval_rels = all_eval_releases[proj_name]
    # 导出训练集
    export_ngram_data_each_release(train_rel, True)
    # 导出测试集
    for rel in eval_rels:
        export_ngram_data_each_release(rel, False)
        # break


# 导出 n-gram 数据
def export_ngram_data_all_projs():
    # 遍历所有项目类型
    for proj in proj_names:
        export_data_all_releases(proj)
        print('finish', proj)


# 导出 易出错（error-prone） 数据
def export_errorprone_data(proj_name):
    cur_eval_rels = all_eval_releases[proj_name][1:]  # 从测试集第二个版本开始

    for rel in cur_eval_rels:  # 遍历测试集

        save_dir = data_for_error_prone_dir + rel + '/'  # 保存目录

        if not os.path.exists(save_dir):  # 如果目录不存在，创建目录
            os.makedirs(save_dir)
        data_df = pd.read_csv(base_original_data_dir + rel + '_ground-truth-files_dataset.csv',
                              encoding='latin')  # 读取原数据

        data_df = data_df[data_df['Bug'] == True]  # 读取有bug的数据

        for filename, df in data_df.groupby('File'):  # 遍历文件

            if 'test' in filename or '.java' not in filename:  # 如果是测试文件，或者不是java文件，
                continue

            filename = filename.replace('/', '_')  # 替换文件名中的/

            code = list(df['SRC'])[0].strip() # 读取代码并且去除每行首尾空格

            with open(save_dir + filename, 'w', encoding='utf-8') as f: # 保存代码到本地 /datasets/ErrorProne_data/+rel(当前测试集名)+/+filename
                f.write(code)

        print('finish release', rel)


def export_error_prone_data_all_projs():
    for proj in proj_names:
        export_errorprone_data(proj)
        print('finish', proj)


export_ngram_data_all_projs()
export_error_prone_data_all_projs()
