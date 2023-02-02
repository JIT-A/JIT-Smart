#%%

import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

#%%

project_list = ['ant-ivy', 'commons-math', 'opennlp', 'parquet-mr', 'commons-lang',
       'commons-net', 'commons-collections', 'commons-beanutils',
       'commons-codec', 'commons-compress', 'commons-configuration',
       'commons-digester', 'commons-jcs', 'commons-io', 'commons-scxml',
       'commons-validator', 'commons-vfs', 'giraph', 'commons-bcel',
       'commons-dbcp', 'gora']
print('prj nums: ', len(project_list))

#%%

features_train_pkl = pd.read_pickle('./jitfine/features_train.pkl')
print('train data nums：', len(features_train_pkl))

features_valid_pkl = pd.read_pickle('./jitfine/features_valid.pkl')
print('valid data nums：', len(features_valid_pkl))

features_test_pkl = pd.read_pickle('./jitfine/features_test.pkl')
print('test data nums：', len(features_test_pkl))

#%%

df = features_train_pkl
df = df.append(features_valid_pkl, ignore_index=True)
df = df.append(features_test_pkl, ignore_index=True)
df.info()

#%%

df['project'].value_counts()

#%%



#%%

changes_train_pkl = pd.read_pickle('./jitfine/changes_train.pkl')
print('train data nums：', len(changes_train_pkl[0]))
print('info_nums: ', len(changes_train_pkl))
changes_valid_pkl = pd.read_pickle('./jitfine/changes_valid.pkl')
print('valid data nums：', len(changes_valid_pkl[0]))
print('info_nums: ', len(changes_valid_pkl))
changes_test_pkl = pd.read_pickle('./jitfine/changes_test.pkl')
print('test data nums：', len(changes_test_pkl[0]))
print('info_nums: ', len(changes_test_pkl))

# changes_train_pkl[0].index('09fd49b7d8e1a14c6d2741a1bdfa4f2b3089e199')
all_changes_pkl = changes_train_pkl
for i in range(4):
    all_changes_pkl[i].extend(changes_valid_pkl[i])
    all_changes_pkl[i].extend(changes_test_pkl[i])

len(all_changes_pkl), len(all_changes_pkl[0]), len(all_changes_pkl[1]), len(all_changes_pkl[2]), len(all_changes_pkl[3])





#%%
train_buggy_commit_lines_df = pd.read_pickle('./jitsmart/train_buggy_commit_lines_df.pkl')
print('train data nums：', len(train_buggy_commit_lines_df))

valid_buggy_commit_lines_df = pd.read_pickle('./jitsmart/valid_buggy_commit_lines_df.pkl')
print('valid data nums：', len(valid_buggy_commit_lines_df))

test_buggy_commit_lines_df = pd.read_pickle('./jitsmart/test_buggy_commit_lines_df.pkl')
print('test data nums：', len(test_buggy_commit_lines_df))


#%%

buggy_line_df = train_buggy_commit_lines_df
buggy_line_df = buggy_line_df.append(valid_buggy_commit_lines_df, ignore_index=True)
buggy_line_df = buggy_line_df.append(test_buggy_commit_lines_df, ignore_index=True)
buggy_line_df.info()


#%%

for prj in tqdm(project_list):
    print('-'*20)
    output_dir = './cross_prj_data/jitsmart/'+prj+'/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_valid_df = df[df['project']!=prj]
    test_df = df[df['project']==prj]
    train_valid_df.index = list(range(len(train_valid_df)))
    test_df.index = list(range(len(test_df)))

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, valid_idx in ss.split(train_valid_df, train_valid_df['is_buggy_commit']):
        train_idx = train_idx
        valid_idx = valid_idx
        print('finish split => train, valid')

    train_df = train_valid_df.iloc[train_idx]
    valid_df = train_valid_df.iloc[valid_idx]
    train_df.index = list(range(len(train_df)))
    valid_df.index = list(range(len(valid_df)))

    print('train_data_nums: ', len(train_df))
    print('valid_data_nums: ', len(valid_df))
    print('test_data_nums: ', len(test_df))


    train_df.to_pickle(output_dir+'features_train.pkl')
    valid_df.to_pickle(output_dir+'features_valid.pkl')
    test_df.to_pickle(output_dir+'features_test.pkl')

    train_all_info = []
    train_changes_info0 = []
    train_changes_info1 = []
    train_changes_info2 = []
    train_changes_info3 = []
    for cmh in train_df['commit_hash'].to_list():
        idx = all_changes_pkl[0].index(cmh)
        train_changes_info0.append(all_changes_pkl[0][idx])
        train_changes_info1.append(all_changes_pkl[1][idx])
        train_changes_info2.append(all_changes_pkl[2][idx])
        train_changes_info3.append(all_changes_pkl[3][idx])
    train_all_info.append(train_changes_info0)
    train_all_info.append(train_changes_info1)
    train_all_info.append(train_changes_info2)
    train_all_info.append(train_changes_info3)
    print('train_changes_data_nums: ', len(train_all_info[0]))
    # 数据保存
    path = './cross_prj_data/jitsmart/'+prj+'/changes_train.pkl'
    output = open(path, 'wb')
    pickle.dump(train_all_info, output)
    output.close()


    valid_all_info = []
    valid_changes_info0 = []
    valid_changes_info1 = []
    valid_changes_info2 = []
    valid_changes_info3 = []
    for cmh in valid_df['commit_hash'].to_list():
        idx = all_changes_pkl[0].index(cmh)
        valid_changes_info0.append(all_changes_pkl[0][idx])
        valid_changes_info1.append(all_changes_pkl[1][idx])
        valid_changes_info2.append(all_changes_pkl[2][idx])
        valid_changes_info3.append(all_changes_pkl[3][idx])
    valid_all_info.append(valid_changes_info0)
    valid_all_info.append(valid_changes_info1)
    valid_all_info.append(valid_changes_info2)
    valid_all_info.append(valid_changes_info3)
    print('valid_changes_data_nums: ', len(valid_all_info[0]))
    # 数据保存
    path = './cross_prj_data/jitsmart/'+prj+'/changes_valid.pkl'
    output = open(path, 'wb')
    pickle.dump(valid_all_info, output)
    output.close()



    test_all_info = []
    test_changes_info0 = []
    test_changes_info1 = []
    test_changes_info2 = []
    test_changes_info3 = []
    for cmh in test_df['commit_hash'].to_list():
        idx = all_changes_pkl[0].index(cmh)
        test_changes_info0.append(all_changes_pkl[0][idx])
        test_changes_info1.append(all_changes_pkl[1][idx])
        test_changes_info2.append(all_changes_pkl[2][idx])
        test_changes_info3.append(all_changes_pkl[3][idx])
    test_all_info.append(test_changes_info0)
    test_all_info.append(test_changes_info1)
    test_all_info.append(test_changes_info2)
    test_all_info.append(test_changes_info3)
    print('test_changes_data_nums: ', len(test_all_info[0]))
    # 数据保存
    path = './cross_prj_data/jitsmart/'+prj+'/changes_test.pkl'
    output = open(path, 'wb')
    pickle.dump(test_all_info, output)
    output.close()







    train_buggy_line_df = buggy_line_df[buggy_line_df['commit hash'].apply(lambda x: x in train_all_info[0])]
    train_buggy_line_df.index = list(range(len(train_buggy_line_df)))
    train_buggy_line_df.to_pickle(output_dir+'train_buggy_commit_lines_df.pkl')

    valid_buggy_line_df = buggy_line_df[buggy_line_df['commit hash'].apply(lambda x: x in valid_all_info[0])]
    valid_buggy_line_df.index = list(range(len(valid_buggy_line_df)))
    valid_buggy_line_df.to_pickle(output_dir+'valid_buggy_commit_lines_df.pkl')

    test_buggy_line_df = buggy_line_df[buggy_line_df['commit hash'].apply(lambda x: x in test_all_info[0])]
    test_buggy_line_df.index = list(range(len(test_buggy_line_df)))
    test_buggy_line_df.to_pickle(output_dir+'test_buggy_commit_lines_df.pkl')

    print('train_bglc_nums: ', len(set(train_buggy_line_df['commit hash'])))
    print('valid_bglc_nums: ', len(set(valid_buggy_line_df['commit hash'])))
    print('test_bglc_nums: ', len(set(test_buggy_line_df['commit hash'])))

    print('-'*20)