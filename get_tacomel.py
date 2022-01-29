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
from librosa.util import normalize
import librosa
import soundfile as sf
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm
from utils_1 import text_to_sequence, ensure_folder, plot_data, HParams
from taco2models.models import Tacotron2
from librosa.display import specshow
import matplotlib.pylab as plt
from meldataset import mel_spectrogram
from taco2models import layers
import config
import imageio
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from config import sampling_rate, VOCAB, IVOCAB
from env import AttrDict
from models import Generator

MAX_WAV_VALUE = 32768.0
hparams = config
stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def plot_mel(spec):
    fig, ax = plt.subplots(figsize=(10, 5), sharey=True)
    img = specshow(spec,y_axis='mel',x_axis='time',ax=ax)
    ax.set(title='Mel-Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.2f dB")
    plt.savefig('mel.jpg')


def plot_mels(spec,spec_est):
    fig, ax = plt.subplots(2,1, figsize=(10,8), sharey=True)
    img0 = specshow(spec_est,y_axis='mel',x_axis='time',ax=ax[0])
    img1 = specshow(spec,y_axis='mel',x_axis='time',ax=ax[1])
    ax[0].set(title='estimated mel-spectrogram')
    ax[1].set(title='gt mel-spectrogram')
    fig.colorbar(img1, ax=ax[1], format="%+2.2f dB")
    fig.colorbar(img0, ax=ax[0], format="%+2.2f dB")
    plt.savefig('mels.jpg')

def check(s):
    for ch in s:
        if ch not in VOCAB:
            return False
    return True

if __name__ == '__main__':
    '''
    这个脚本的功能:
    利用已经train好的Tacotron2模型，以及数据集中的文本数据，在teacher forcing模式下，生成mel-spectrogram
    生成的mel-spectrogram用于fine-tune HiFi-GAN
    '''
# load tacotron2
    device = torch.device('cuda')  
    checkpoint_path = 'xiaoxian_checkpoints/12_7BEST_checkpoint.tar'
    ckp_mode = 'dict'
    config = HParams()
    
    if ckp_mode == 'dict':
        print('loading model: {}...'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        tacotron2_model = checkpoint['model']
        print('Validation loss is {:.2f}.'.format(checkpoint['loss']))
    else:
        
        print('loading model: {}...'.format(checkpoint_path))
        tacotron2_model = Tacotron2(config)
        tacotron2_model.load_state_dict(torch.load(checkpoint_path))

    infile  = open('/mfs/zengsunlu/tianchi_data/xiaoxian_pre_v3.json','r',encoding='utf-8')  # 给定数据集的metafile，里面包含音频文件地址，对应的文本
    hifi_metafile = open('/mfs/zengsunlu/tianchi_data/hifigan_v1.json','w',encoding='utf-8') # 为HiFi-GAN生成可解析的metafile，包含音频地址
    wav_dir = '/mfs/zengsunlu/tianchi_data/wav'
    mel_dir = '/mfs/zengsunlu/tianchi_data/mel_data_tf'

    for line in tqdm(infile.readlines()):
        dic = json.loads(line)
        if not os.path.exists(dic['audio_filepath']) or not check(dic['text']):
            # 文字内容不合格，或者音频文件不存在，直接跳过该样本
            continue
        hifi_metafile.write(line)
        # dic['audio_filepath'] = "/mfs/zengsunlu/tianchi_data/wav/6465d5284e9f4ad89b807653abd8e87b.wav"
        mel_filepath = os.path.join(mel_dir, dic['audio_filepath'].split('/')[-1].split('.')[0] + '.npy')
        audio_filepath = os,.path.join(audio_dir, dic['audio_filepath'].split('/')[-1])
        audio, sampling_rate = librosa.core.load(dic['audio_filepath'])
        audio, _ = librosa.effects.trim(audio)
        if sampling_rate != config.sampling_rate:
            print(dic['audio_filepath'])
            audio = librosa.resample(audio,sampling_rate,config.sampling_rate)
        if not os.path.exists(audio_filepath):
            sf.write(audio_filepath,audio,sampling_rate)

        audio = torch.FloatTensor(audio.astype(np.float32)).to(device)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        audio = audio.unsqueeze(0)
        # get input text
        text = dic['text']
        sequence = np.array(text_to_sequence(text))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long().to(device)
        sequences = torch.unsqueeze(sequence,0)

    # get mel-spectrogram with using tacotron2 with teacher forcing
        mel = mel_spectrogram(audio, config.win_length, config.n_mel_channels, config.sampling_rate, config.hop_length, config.win_length, config.mel_fmin, config.mel_fmax).to(device)
        mel_outputs, mel_outputs_postnet, _, alignments = tacotron2_model.inference_tf((sequence,mel))
        plot_mels(mel.detach().cpu().numpy().squeeze(),mel_outputs.detach().cpu().numpy().squeeze())
        if mel_outputs.shape[-1] < mel.shape[-1]:
            mel = mel[:,:,mel_outputs.shape[-1]]
        elif mel_outputs.shape[-1] > mel.shape[-1]:
            mel_outputs = mel_outputs[:,:,mel.shape[-1]]
        np.save(filename,mel_outputs_postnet.detach().cpu().numpy()) # save mel-spectrogram for HiFi-GAN fine-tuning
