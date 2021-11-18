import numpy as np
import torch
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm

import config
from data_gen import TextMelLoader, TextMelCollate
from taco2models.loss_function import Tacotron2Loss
from taco2models.models import Tacotron2
from taco2models.optimizer import Tacotron2Optimizer
from utils_1 import parse_args, save_checkpoint, AverageMeter, get_logger, test

checkpoint = 'checkpoint.tar'
#checkpoint = 'BEST_checkpoint.tar'
ckp_model = 'dict'

if ckp_model == 'dict':
    model = torch.load(checkpoint)['model']
else:
    checkpoint = torch.load(checkpoint)
    model = Tacotron2(config)
    model.load_state_dict(checkpoint)

model.to(config.device)

def test(test_loader, model, criterion):
    model.eval()

    losses = AverageMeter()

    # Batches
    with torch.no_grad(): 
        for batch in tqdm(test_loader):
            #model.zero_grad()
            x, y = model.parse_batch(batch)

            # Forward prop.
            y_pred = model(x)

            loss = criterion(y_pred, y)

            # Keep track of metrics
            print(loss.item())
            losses.update(loss.item())

        # Print status
        print('\nValidation Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

    return losses.avg

criterion = Tacotron2Loss()

collate_fn = TextMelCollate(config.n_frames_per_step)


test_dataset = TextMelLoader(config.validation_files, config)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                                            pin_memory=True, shuffle=False)
test(test_loader,model,criterion)
