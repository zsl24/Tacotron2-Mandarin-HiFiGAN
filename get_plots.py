from librosa.display import specshow,waveplot
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
from librosa import stft
from librosa.feature import melspectrogram
from librosa.core import load
import numpy as np

def plot_wv(wv):
    fig, ax = plt.subplots(figsize=(10, 5), sharey=True)
    img = waveplot(wv,ax=ax)
    ax.set(title='Waveform')
    plt.savefig('waveform.jpg')

def plot_wvs(wv,wv_est):
    fig, ax = plt.subplots(2,1,figsize=(10, 8), sharey=True)
    ax[0].set(title='estimated waveform')
    ax[0].plot(wv_est)
    ax[1].set(title='gt waveform')
    ax[1].plot(wv)
    plt.savefig('waveformes.jpg')    

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
if __name__ == '__main__':
    sr = 22050
    wv, sr = load('/mfs/zengsunlu/tianchi_data/wav/d905674e38194352ae6a69fcfacd1509.wav')
    wv_est, _ = load('output.wav')

    mel = librosa.power_to_db(melspectrogram(y=wv,sr=sr,n_fft=1024,hop_length=256,win_length=1024,n_mels=80,power=2),ref=np.max)
    mel_est = librosa.power_to_db(melspectrogram(y=wv_est,sr=sr,n_fft=1024,hop_length=256,win_length=1024,n_mels=80,power=2),ref=np.max)

    plot_wvs(wv,wv_est)
    plot_mels(mel,mel_est)

