import pandas as pd
import os, re
import numpy as np

from my_util import *

data_root_dir = '../datasets/original/'
save_dir = "../datasets/preprocessed_data/"

char_to_remove = ['+', '-', '*', '/', '=', '++', '--', '\\', '<str>', '<char>', '|', '&', '!']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_lvl_dir = data_root_dir + 'File-level/'
line_lvl_dir = data_root_dir + 'Line-level/'


def is_comment_line(code_line, comments_list):
    '''
        input
            code_line (string): source code in a line
            comments_list (list): a list that contains every comments
        output
            boolean value
    '''

    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith('//'):
        return True
    elif code_line in comments_list:
        return True

    return False


def is_empty_line(code_line):
    '''
        input
            code_line (string)
        output
            boolean value
    '''

    if len(code_line.strip()) == 0:
        return True

    return False


def preprocess_code_line(code_line):
    '''
        input
            code_line (string)
    '''

    code_line = re.sub("\'\'", "\'", code_line)# 将''替换为'
    code_line = re.sub("\".*?\"", "<str>", code_line)# 将字符串替换为<str>
    code_line = re.sub("\'.*?\'", "<char>", code_line)# 将字符替换为<char>
    code_line = re.sub('\b\d+\b', '', code_line)# 将数字替换为空
    code_line = re.sub("\\[.*?\\]", '', code_line)# 将数组替换为空
    code_line = re.sub("[\\.|,|:|;|{|}|(|)]", ' ', code_line)# 将标点符号替换为空格

    for char in char_to_remove:
        code_line = code_line.replace(char, ' ')# 将运算符替换为空格

    code_line = code_line.strip()# 去掉行首和行尾的空格

    return code_line


def create_code_df(code_str, filename):
    '''
        input
            code_str (string): a source code
            filename (string): a file name of source code

        output
            code_df (DataFrame): a dataframe of source code that contains the following columns
            - code_line (str): source code in a line
            - line_number (str): line number of source code line
            - is_comment (bool): boolean which indicates if a line is comment
            - is_blank_line(bool): boolean which indicates if a line is blank
    '''
    # 创建一个空的DataFrame
    df = pd.DataFrame()

    # 将源代码按行分割
    code_lines = code_str.splitlines()

    preprocess_code_lines = []

    is_comments = []
    is_blank_line = []

    # 找到源代码中的注释
    comments = re.findall(r'(/\*[\s\S]*?\*/)', code_str, re.DOTALL)

    # 将注释按行分割
    comments_str = '\n'.join(comments)
    comments_list = comments_str.split('\n')

    # 遍历源代码中的每一行
    for l in code_lines:
        # 去掉行首和行尾的空格
        l = l.strip()
        # 判断是否是注释
        is_comment = is_comment_line(l, comments_list)
        # 将判断结果添加到列表中
        is_comments.append(is_comment)
        # preprocess code here then check empty line...
        # 如果不是注释
        if not is_comment:
            # 处理代码
            l = preprocess_code_line(l)
        # 判断是否是空行
        is_blank_line.append(is_empty_line(l))
        # 将处理后的代码添加到列表中
        preprocess_code_lines.append(l)

    # 判断文件名是否包含test
    if 'test' in filename:
        is_test = True
    else:
        is_test = False

    # 文件名*代码行数
    df['filename'] = [filename] * len(code_lines)
    # 是否是测试文件
    df['is_test_file'] = [is_test] * len(code_lines)
    # 处理后的代码行
    df['code_line'] = preprocess_code_lines
    # 行号 = 1~代码行数
    df['line_number'] = np.arange(1, len(code_lines) + 1)
    # 是否是注释
    df['is_comment'] = is_comments
    # 是否是空行
    df['is_blank'] = is_blank_line

    return df


def preprocess_data(proj_name):
    cur_all_rel = all_releases[proj_name]

    for rel in cur_all_rel:
        # 读取文件级别数据集
        file_level_data = pd.read_csv(file_lvl_dir + rel + '_ground-truth-files_dataset.csv', encoding='latin')
        # 读取行级别数据集
        line_level_data = pd.read_csv(line_lvl_dir + rel + '_defective_lines_dataset.csv', encoding='latin')

        # 填充空值
        file_level_data = file_level_data.fillna('')
        # 找到bug文件，找到行级别数据集中file列中的唯一文件名
        buggy_files = list(line_level_data['File'].unique())

        # 创建一个空列表
        preprocessed_df_list = []

        # 遍历文件级别数据集中的每一行
        for idx, row in file_level_data.iterrows():
            # 获取文件名
            filename = row['File']
            # 如果文件名不是以.java结尾的，跳过
            if '.java' not in filename:
                continue
            # 获取源代码
            code = row['SRC']
            # 获取bug标签
            label = row['Bug']
            # 创建一个新定义的数据集
            code_df = create_code_df(code, filename)

            # 将bug标签添加到数据集中
            code_df['file-label'] = [label] * len(code_df)

            code_df['line-label'] = [False] * len(code_df)

            # 如果文件名在bug文件中
            if filename in buggy_files:
                # 获取行级别数据集中文件名为filename的行号
                buggy_lines = list(line_level_data[line_level_data['File'] == filename]['Line_number'])
                # 将行号添加到数据集中
                code_df['line-label'] = code_df['line_number'].isin(buggy_lines)

            if len(code_df) > 0:
                preprocessed_df_list.append(code_df)

        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(save_dir + rel + ".csv", index=False)
        print('finish release {}'.format(rel))


for proj in list(all_releases.keys()):
    print(proj)
    # preprocess_data(proj)
