import json
import config
import random
import os

def check(dic):
    '''
    check validation of data
    '''
    s = dic['text']
    for ch in s:
        if ord('a') <= ord(ch) <= ord('z') or ch.isdigit() or ch ==' ':
            pass
        else:
            return False
    if not os.path.exists(dic['audio_filepath']):
        return False
    return True

dataset = 'aixia'
# please give file that contains processed text (pinyin)
file = open('/mfs/zengsunlu/voice_selection/aixia/tts_data_speeed_120_pre.json','r',encoding='utf-8')
data = []
cnt = 0
for line in file.readlines():
    cnt += 1
    dic = json.loads(line)
    if check(dic):
        data.append(dic)
print(f'we have {len(data)}/{cnt} samples.')
random.shuffle(data)
num_train = int(len(data) * 0.9)
num_valid = len(data) - num_train


if not os.path.exists(f'{dataset}_filelist'):
    os.mkdir(f'{dataset}_filelist')

# training set
outfile = open(f'{dataset}_filelist/{dataset}_audio_text_train_filelist.json','w',encoding='utf-8')
for i in range(num_train):
    j = json.dumps(data[i])
    outfile.write(j+'\n')
print(f'complete for {num_train} training files for {dataset}')

# validation set
outfile = open(f'{dataset}_filelist/{dataset}_audio_text_valid_filelist.json','w',encoding='utf-8')
for i in range(num_train,num_train+num_valid):
    j = json.dumps(data[i])
    outfile.write(j+'\n')
print(f'complete for {num_valid} validation file for {dataset}')
