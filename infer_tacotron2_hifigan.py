import matplotlib.pylab as plt
import numpy as np
import pinyin
import soundfile as sf
import torch
from time import time
from config import sampling_rate
from taco2models.models import Tacotron2
from utils_1 import text_to_sequence, ensure_folder, plot_data, HParams
import os
import numpy as np
import json
import torch
from env import AttrDict
from models import Generator
import imageio


MAX_WAV_VALUE = 32768.0


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


# tactron2 + HiFi-GAN
if __name__ == '__main__':
# load tacotron2
    config = HParams()


    checkpoint_path = 'checkpoint.tar'
    ckp_mode = 'dict'

    if ckp_mode == 'dict':
        print('loading model: {}...'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        tacotron2_model = checkpoint['model']
    else:
        print('loading model: {}...'.format(checkpoint_path))
        tacotron2_model = Tacotron2(config)
        tacotron2_model.load_state_dict(torch.load(checkpoint_path))



    
    tacotron2_model = tacotron2_model.to('cpu')
    tacotron2_model.eval()    
# load HiFi-GAN
    hifigan_path = 'LJ_FT_T2_V3/generator_v3'
    config_file = 'LJ_FT_T2_V3/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')    

    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(hifigan_path, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

# get input text
    text = "相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型"
    text = pinyin.get(text, format="numerical", delimiter=" ")
    print(text)
    sequence = np.array(text_to_sequence(text))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

# get mel-spectrogram
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2_model.inference(sequence)
    aligments = alignments.detach().numpy().squeeze()
    imageio.imwrite('alignment_ref.jpg', aligments)

# use HiFi-GAN get waveform
    mel_outputs_postnet = torch.FloatTensor(mel_outputs_postnet).to(device)
    hifi_start = time()
    y_g_hat = generator(mel_outputs_postnet)
    hifi_end = time()
    hifi_dur = hifi_end - hifi_start
    audio = y_g_hat.squeeze()
    print('Inference time for HiFi-GAN is :{:.2f} seconds, generation rate is :{:.2f} kHz.'.format(hifi_dur,(audio.shape[0]/hifi_dur)/1000))
    #audio = audio * MAX_WAV_VALUE
    audio = audio.detach().cpu().numpy().astype('float32')
    
# write waveform file
    sf.write('output.wav',audio,sampling_rate)
