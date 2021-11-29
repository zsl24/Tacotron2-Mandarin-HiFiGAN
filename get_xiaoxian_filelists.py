import json
import config

filename = 'xiaoxian_filelist/xiaoxian_pre.json'
file = open(filename,'r',encoding='utf-8')
viya = []
for line in file.readlines():
    dic = json.loads(line)
    viya.append(dic)

num_train = 10000
num_valid = 700
# 训练集
with open('xiaoxian_filelist/xiaoxian_audio_text_train_filelist.json','w') as outfile:
    for i in range(num_train):
        j = json.dumps(viya[i])
        outfile.write(j+'\n')
print('complete for train file')
# 验证集
with open('xiaoxian_filelist/xiaoxian_audio_text_valid_filelist.json','w') as outfile:
    for i in range(num_train,num_train+num_valid):
        j = json.dumps(viya[i])
        outfile.write(j+'\n')
print('complete for validation file')
