import os
import config

wav_path = '/mfs/data/ASR_dataset/TTS/biaobei'
text_path = 'dataset_text'

dataset_path = '/mfs/data/ASR_dataset/TTS/biaobei'
summary_file = 'biaobei_filelist/000001-010000.txt'
summary_file = os.path.join(dataset_path,summary_file)
with open(summary_file,'r') as f:
    text_data = f.readlines()

# 写训练集文件路径
with open('biaobei_filelist/biaobei_audio_text_train_filelist.txt','w') as f:
    data = []
    for i in range(config.num_train):
        wv_fn = str(i+1).rjust(6,'0') + '.wav'
        wv_fn = os.path.join(wav_path,wv_fn)
        tx = text_data[2*i+1][1:-1]
        data.append(wv_fn + '|' + tx + '\n')
    f.writelines(data)

# 写验证集文件路径
with open('biaobei_filelist/biaobei_audio_text_valid_filelist.txt','w') as f:
    data = []
    for i in range(config.num_train,config.num_train+config.num_valid):
        wv_fn = str(i+1).rjust(6,'0') + '.wav'
        wv_fn = os.path.join(wav_path,wv_fn)
        tx = text_data[2*i+1][1:-1]
        data.append(wv_fn + '|' + tx + '\n')
    f.writelines(data)

# 写测试集文件路径
with open('biaobei_filelist/biaobei_audio_text_test_filelist.txt','w') as f:
    data = []
    for i in range(config.num_train+config.num_valid,10000):
        wv_fn = str(i+1).rjust(6,'0') + '.wav'
        wv_fn = os.path.join(wav_path,wv_fn)
        tx = text_data[2*i+1][1:-1]
        data.append(wv_fn + '|' + tx + '\n')
    f.writelines(data)

