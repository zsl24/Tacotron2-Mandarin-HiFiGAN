import os
import json
import numpy as np
from env import AttrDict, build_env
from scipy.io.wavfile import read
import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from scipy.io.wavfile import read
from tqdm import tqdm
import matplotlib.pylab as plt
from taco2models import layers
import config
from get_pictures import plot_wv

MAX_WAV_VALUE = 32768.0

if __name__ == '__main__':
    config_file = 'ft_model/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    cnt = 0
    infile0  = open('hifigan_filelist/hifigan_metadata.json','r',encoding='utf-8')
    train_filelist = open('hifigan_filelist/training.json','w',encoding='utf-8')
    val_filelist = open('hifigan_filelist/validation.json','w',encoding='utf-8')
    data = []

    for line in tqdm(infile0.readlines()):
        dic = json.loads(line) 
        audio_filepath = '/home/yanan/zengsunlu/hifigan_ft/aixia/wav/' + dic['audio_filepath']
        mel_filepath = '/home/yanan/zengsunlu/hifigan_ft/aixia/mel_tf/' + dic['audio_filepath'].split('.')[0] + '.npy'
        if not os.path.exists(mel_filepath) or not os.path.exists(audio_filepath):
            continue
        dic['audio_filepath'] = audio_filepath
        if cnt < 1000:
            val_filelist.write(json.dumps(dic,ensure_ascii=False) + '\n')
        else:
            train_filelist.write(json.dumps(dic,ensure_ascii=False) + '\n')
        cnt += 1  
