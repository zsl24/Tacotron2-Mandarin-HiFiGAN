import numpy as np
import pinyin
import soundfile as sf
import torch
from time import time
from config import sampling_rate
from taco2models.models import Tacotron2
from utils_1 import text_to_sequence, ensure_folder, plot_data, HParams, process_text
import os
import numpy as np
import json
import torch
from env import AttrDict
from models import Generator
import imageio
import argparse

MAX_WAV_VALUE = 32768.0


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Tacotron2')
    parser.add_argument('--taco_model', default='/mfs/zengsunlu/models/tacotron2/new_dict/aixia.tar', type=str, help='path of tacotron2 model')
    parser.add_argument('--load_type', default='dict', type=str, help='load method of tacotron2 model')
    parser.add_argument('--hifi_model', default='/mfs/zengsunlu/models/hifigan/v1/aixia_generator', type=str, help='load method of tacotron2 model')
    parser.add_argument('--hifi_config', default='/mfs/zengsunlu/models/hifigan/v1/config.json', type=str, help='configuration file of HiFi-GAN')
    parser.add_argument('--text', default='本次拟向您推荐的这款产品已收录在我航个人产品信息查询平台 您可通过我航的官方网站www点bosc点cn中查询到该产品', type=str, help='text to generate voice')
    args = parser.parse_args()
    return args

# tactron2 + HiFi-GAN
if __name__ == '__main__':
    global args
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')    
    device = torch.device('cpu')
# load tacotron2
    config = HParams()
    checkpoint_path = args.taco_model
    ckp_mode = 'dict'
    if ckp_mode == 'dict':
        print('Loading Tacotron2 model: {}...'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        tacotron2_model = checkpoint['model']
        print('Validation Loss of Tacotron2 is {}'.format(checkpoint['loss']))
    else:
        print('Loading Tacotron2 model: {}...'.format(checkpoint_path))
        tacotron2_model = Tacotron2(config)
        tacotron2_model.load_state_dict(torch.load(checkpoint_path))

    tacotron2_model = tacotron2_model.to('cpu')
    tacotron2_model.eval()    
# load HiFi-GAN
    hifigan_path = args.hifi_model
    config_file = args.hifi_config
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    print('Loading HiFi-GAN model ...')
    state_dict_g = load_checkpoint(hifigan_path, device)
    print('Validation loss of HiFi-GAN is: ',state_dict_g['loss'])
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

# get input text
    text = args.text
    text = process_text(text)
    print(text)
    sequence = np.array(text_to_sequence(text))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

# get mel-spectrogram
    taco_start = time()
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2_model.inference(sequence)
    taco_end = time()
    taco_dur = taco_end - taco_start
    print('Inference time for Tacotron2 is :{:.2f} seconds on {}.'.format(taco_dur,device))
    aligments = alignments.detach().numpy().squeeze().T
    imageio.imwrite('alignment_reference.jpg', aligments)

# use HiFi-GAN get waveform
    mel_outputs_postnet = torch.FloatTensor(mel_outputs_postnet).to(device)
    hifi_start = time()
    y_g_hat = generator(mel_outputs_postnet)
    hifi_end = time()
    hifi_dur = hifi_end - hifi_start
    audio = y_g_hat.squeeze()
    print('Inference time for HiFi-GAN is :{:.2f} seconds, generation rate is :{:.2f} kHz on {}.'.format(hifi_dur,audio.shape[0]/hifi_dur/1000, device))
    #audio = audio * MAX_WAV_VALUE
    audio = audio.detach().cpu().numpy().astype('float32')
    
# write waveform file
    sf.write('output.wav',audio,sampling_rate)
