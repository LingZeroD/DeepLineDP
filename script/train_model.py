import os, re, argparse

import torch.optim as optim

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm

from sklearn.utils import compute_class_weight

from DeepLineDP_model import *
from my_util import *

torch.manual_seed(0)

arg = argparse.ArgumentParser()

arg.add_argument('-dataset', type=str, default='hbase', help='software project name (lowercase)')
arg.add_argument('-batch_size', type=int, default=8)
arg.add_argument('-num_epochs', type=int, default=54)
arg.add_argument('-embed_dim', type=int, default=50, help='word embedding size')
arg.add_argument('-word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')
arg.add_argument('-sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')
arg.add_argument('-word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
arg.add_argument('-sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')
arg.add_argument('-lr', type=float, default=0.001, help='learning rate')
arg.add_argument('-exp_name', type=str, default='')

args = arg.parse_args()

# model setting
batch_size = args.batch_size
num_epochs = args.num_epochs
max_grad_norm = 5
embed_dim = args.embed_dim
word_gru_hidden_dim = args.word_gru_hidden_dim
sent_gru_hidden_dim = args.sent_gru_hidden_dim
word_gru_num_layers = args.word_gru_num_layers
sent_gru_num_layers = args.sent_gru_num_layers
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = args.dropout
lr = args.lr

save_every_epochs = 1
exp_name = args.exp_name

max_train_LOC = 900

prediction_dir = '../output/prediction/DeepLineDP/'
save_model_dir = '../output/model/DeepLineDP/'

file_lvl_gt = '../datasets/preprocessed_data/'

weight_dict = {}

device = torch.device("cuda:0")


def get_loss_weight(labels):
    '''
        input
            labels: a PyTorch tensor that contains labels
        output
            weight_tensor: a PyTorch tensor that contains weight of defect/clean class
    '''
    label_list = labels.cpu().numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])

    weight_tensor = torch.tensor(weight_list).reshape(-1, 1).cuda()
    return weight_tensor


def train_model(dataset_name):
    # 损失函数保存路径
    loss_dir = '../output/loss/DeepLineDP/'
    # 模型保存路径
    actual_save_model_dir = save_model_dir + dataset_name + '/'

    if not exp_name == '':  # 如果有实验名
        actual_save_model_dir = actual_save_model_dir + exp_name + '/'
        loss_dir = loss_dir + exp_name

    if not os.path.exists(actual_save_model_dir):  # 如果不存在该路径
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):  # 如果不存在该路径
        os.makedirs(loss_dir)
    # 训练集
    train_rel = all_train_releases[dataset_name]
    # 校验集
    valid_rel = all_eval_releases[dataset_name][0]

    # 获取训练集的数据
    train_df = get_df(train_rel)
    # 获取校验集的数据
    valid_df = get_df(valid_rel)

    # 获取训练集的代码3d和标签
    train_code3d, train_label = get_code3d_and_label(train_df, True)
    valid_code3d, valid_label = get_code3d_and_label(valid_df, True)

    # 计算样本权重
    sample_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)

    weight_dict['defect'] = np.max(sample_weights)  # 缺陷 = 最大样本权重
    weight_dict['clean'] = np.min(sample_weights)  # 无缺陷 = 最小样本权重

    w2v_dir = get_w2v_path()  # 获取word2vec词向量路径

    # 获取word2vec词向量
    word2vec_file_dir = os.path.join(w2v_dir, dataset_name + '-' + str(embed_dim) + 'dim.bin')

    # 加载word2vec词向量
    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for', dataset_name, 'finished')

    # 获取word2vec权重
    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)

    # 获取词汇表大小
    vocab_size = len(word2vec.wv.vocab) + 1  # for unknown tokens

    # 获取训练集和校验集的向量
    x_train_vec = get_x_vec(train_code3d, word2vec)
    x_valid_vec = get_x_vec(valid_code3d, word2vec)

    # 获取最大句子长度
    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)
    # 加载加载训练集和校验集的数据
    train_dl = get_dataloader(x_train_vec, train_label, batch_size, max_sent_len)
    valid_dl = get_dataloader(x_valid_vec, valid_label, batch_size, max_sent_len)

    abs_path=os.path.abspath(loss_dir + dataset_name + '-loss_record.csv')
    print('train-result:'  + abs_path)
    # 获取模型
    model = HierarchicalAttentionNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=word_gru_hidden_dim,
        sent_gru_hidden_dim=sent_gru_hidden_dim,
        word_gru_num_layers=word_gru_num_layers,
        sent_gru_num_layers=sent_gru_num_layers,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout)

    # 将模型放到GPU上
    model.to(device)
    # 冻结词向量
    model.sent_attention.word_attention.freeze_embeddings(False)

    # 设置优化器
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 使用交叉熵损失函数
    criterion = nn.BCELoss()

    #
    checkpoint_files = os.listdir(actual_save_model_dir)

    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')
    # 获取模型数量
    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:  # 如果没有模型
        model.sent_attention.word_attention.init_embeddings(word2vec_weights)  # 初始化词向量
        current_checkpoint_num = 1  # 从第一轮开始训练

        train_loss_all_epochs = []  # 训练集损失
        val_loss_all_epochs = []  # 校验集损失

    else:
        checkpoint_nums = [int(re.findall('\d+', s)[0]) for s in checkpoint_files]  # 获取模型的轮数
        current_checkpoint_num = max(checkpoint_nums)  # 获取最大轮数

        checkpoint = torch.load(
            actual_save_model_dir + 'checkpoint_' + str(current_checkpoint_num) + 'epochs.pth')  # 加载模型

        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

        loss_df = pd.read_csv(loss_dir + dataset_name + '-loss_record.csv')  # 加载损失记录
        train_loss_all_epochs = list(loss_df['train_loss'])  # 训练集损失
        val_loss_all_epochs = list(loss_df['valid_loss'])  # 校验集损失

        current_checkpoint_num = current_checkpoint_num + 1  # 从下一轮开始训练
        print('continue training model from epoch', current_checkpoint_num)  # 继续训练模型

    for epoch in tqdm(range(current_checkpoint_num, num_epochs + 1)):  # 迭代轮数
        train_losses = []
        val_losses = []

        model.train()  # 训练模式

        for inputs, labels in train_dl:  # 迭代训练集
            inputs_cuda, labels_cuda = inputs.cuda(), labels.cuda()  # 将数据放到GPU上

            output, _, __, ___ = model(inputs_cuda)  # 获取输出

            weight_tensor = get_loss_weight(labels)  # 获取损失权重

            criterion.weight = weight_tensor # 设置损失函数损失权重

            loss = criterion(output, labels_cuda.reshape(batch_size, 1)) # 计算损失

            train_losses.append(loss.item()) # 记录损失

            torch.cuda.empty_cache() # 清空GPU缓存

            loss.backward() # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # 梯度裁剪

            optimizer.step() # 更新参数

            torch.cuda.empty_cache() # 清空GPU缓存

        train_loss_all_epochs.append(np.mean(train_losses)) # 记录平均损失

        with torch.no_grad(): # 不计算梯度

            criterion.weight = None # 设置损失函数损失权重为None
            model.eval() # 验证模式

            for inputs, labels in valid_dl: # 迭代校验集
                inputs, labels = inputs.cuda(), labels.cuda() # 将数据放到GPU上
                output, _, __, ___ = model(inputs) #  获取输出

                val_loss = criterion(output, labels.reshape(batch_size, 1)) # 计算损失

                val_losses.append(val_loss.item()) # 记录损失

            val_loss_all_epochs.append(np.mean(val_losses)) # 记录平均损失

        if epoch % save_every_epochs == 0: # 每save_every_epochs轮保存模型
            print(dataset_name, '- at epoch:', str(epoch)) # 打印轮数

            if exp_name == '': # 如果没有实验名称
                torch.save({ # 保存模型
                    'epoch': epoch, # 轮数
                    'model_state_dict': model.state_dict(), # 模型参数
                    'optimizer_state_dict': optimizer.state_dict() # 优化器参数
                },
                    actual_save_model_dir + 'checkpoint_' + str(epoch) + 'epochs.pth') # 保存模型
            else:
                torch.save({
                    'epoch': epoch, # 轮数
                    'model_state_dict': model.state_dict(), # 模型参数
                    'optimizer_state_dict': optimizer.state_dict() # 优化器参数
                },
                    actual_save_model_dir + 'checkpoint_' + exp_name + '_' + str(epoch) + 'epochs.pth') # 保存模型

        loss_df = pd.DataFrame() # 创建损失记录
        loss_df['epoch'] = np.arange(1, len(train_loss_all_epochs) + 1) # 记录轮数
        loss_df['train_loss'] = train_loss_all_epochs # 记录训练集损失
        loss_df['valid_loss'] = val_loss_all_epochs # 记录校验集损失

        loss_df.to_csv(loss_dir + dataset_name + '-loss_record.csv', index=False) # 保存损失记录


dataset_name = args.dataset
train_model(dataset_name)

