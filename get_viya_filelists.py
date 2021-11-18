import json
import config

filename = 'viya_filelist/viya_pre.json'
file = open(filename,'r',encoding='utf-8')
viya = []
for line in file.readlines():
    dic = json.loads(line)
    viya.append(dic)

num_train = config.num_train
num_valid = config.num_valid
# 训练集
with open('viya_audio_text_train_filelist.json','w') as outfile:
    for i in range(num_train):
        j = json.dumps(viya[i])
        outfile.write(j+'\n')
print('complete for train file')
# 验证集
with open('viya_audio_text_val_filelist.json','w') as outfile:
    for i in range(num_train,num_train+num_valid):
        j = json.dumps(viya[i])
        outfile.write(j+'\n')
print('complete for validation file')
