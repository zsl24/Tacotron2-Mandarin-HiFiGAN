import argparse
import logging

import cv2 as cv
import librosa
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pinyin
import torch
import json
import config
import os
import time
from config import sampling_rate, VOCAB, IVOCAB


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, step, steps_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'step': step,
             'steps_since_improvement': steps_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    date = str(time.localtime(time.time()).tm_mon) + '_' + str(time.localtime(time.time()).tm_mday)
    dir = config.dataset+'_checkpoints'
    if not os.path.exists(dir):
        os.mkdir(dir)

    filename = os.path.join(dir, date + 'checkpoint.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        print(f'saving best checkpoint with loss {loss}')
        torch.save(state, os.path.join(dir,date + 'BEST_checkpoint.tar'))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
    

def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

def load_audiopaths_and_text_json(filename):
    '''
    read json file containing audio path, text content, duration of audio
    parse audio file paths and text content, store the results in list, such as:
    [[audiopath0,text0],
     [audiopath1,text1],
     ...,
     [audiopathn,textn]]
    '''
    jsonfile = open(filename,'r',encoding='utf-8')
    filepaths_and_text = []
    for line in jsonfile.readlines():
        dic = json.loads(line)
        filepaths_and_text.append([dic['audio_filepath'],dic['text']])
    return filepaths_and_text


def load_wav_to_torch(full_path):
    # sampling_rate, data = read(full_path)
    y, sr = librosa.core.load(full_path, sampling_rate)
    yt, _ = librosa.effects.trim(y)
    if config.dataset == 'viya' or config.dataset == 'xiaoxian':
        yt = librosa.util.normalize(yt)
    return torch.FloatTensor(yt.astype(np.float32)), sr


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def replace_triple_space(text):
    res = ''
    i = 0
    while i < len(text):
        j = i + 1
        res += text[i]
        if text[i] == ' ':
            while j < len(text) and text[j] == ' ':
                j += 1
            if j - i >= 2:
                res += ' '
        i = j
    return res
            
def process_en(s):
    '''
    由于拼音模块不是很完备，有一些case处理不了
    将文字'嗯'替换成拼音'en3'
    '''
    res = ''
    n = len(s)
    for i in range(n):
        if s[i] == '嗯':
            res += 'en3'
        else:
            res += s[i]
    return res

def process_text(text):
    '''
    This function converts original text into pinyin sequence.
    example:
    text - 'www点bosc点cn 请您抄录风险揭示并签名'
    return - 'w6  w6  w6  dian3 b6  o6  s6  c6  dian3  c6  n6  qing3 nin2 chao1 lu3 feng1 xian3 jie1 shi4 bing4 qian1 ming2'
    '''
    text = pinyin.get(text,format='numerical',delimiter=' ').lower()
    text = process_en(text)
    result = ''
    for i,ch in enumerate(text):
        result += ch
        if i < len(text)-1 and ch.isalpha() and text[i+1] == ' ':
            result += '6 '
        elif i == len(text) - 1 and ord('a') <= ord(ch) <= ord('z'):
            result += '6'
    return result

def text_to_sequence(text):
    # text = chinese_cleaners(text)
    result = [VOCAB[ch] for ch in text]
    return result


def sequence_to_text(seq):
    result = [IVOCAB[str(idx)] for idx in seq]
    return result


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')


def test(model, step_num, loss):
    model.to('cuda')
    model.eval()
    text = "放心吧 我们家羽绒服做的非常好 www点bosc点cn"
    text = process_text(text)
    sequence = np.array(text_to_sequence(text))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    img = alignments.float().data.cpu().numpy()[0].T
    filename = 'images/step{0}_loss{1:.5f}_temp.jpg'.format(step_num,loss)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img * 255.
    cv.imwrite(filename,img)
    
    return img

if __name__ == '__main__':
    from taco2models.models import Tacotron2
    checkpoint = 'ckp/tacotron2-cn.pt'
    checkpoint = torch.load(checkpoint)
    model = Tacotron2(config)
    model.load_state_dict(checkpoint)
    test(model,0,0.23)

class HParams:
    def __init__(self):
        self.n_mel_channels = None
        self.dynamic_loss_scaling = True
        self.fp16_run = False
        self.distributed_run = False

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk = False

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols = 35
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        # Decoder parameters
        self.n_frames_per_step = 1  # currently only 1 is supported
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.batch_size = 64
        self.mask_padding = True  # set model's padded outputs to padded values
