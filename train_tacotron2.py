import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
import config
from data_gen import TextMelLoader, TextMelCollate
from taco2models.loss_function import Tacotron2Loss
from taco2models.models import Tacotron2
from taco2models.optimizer import Tacotron2Optimizer
from utils_1 import save_checkpoint, AverageMeter, get_logger, test


import os

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    steps_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        print('Training from scratch ...')
        # model
        model = Tacotron2(config)
        print(model)
        # model = nn.DataParallel(model)

        # optimizer
        optimizer = Tacotron2Optimizer(
            torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.9, 0.999), eps=1e-6))
    else:
        print('Loading model:{}'.format(checkpoint))
        load_mode = args.load_type
        if load_mode == 'dict':
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            start_epoch = checkpoint['epoch'] + 1
            step = checkpoint['step'] + 1
            steps_since_improvement = checkpoint['steps_since_improvement']
            optimizer = checkpoint['optimizer']
            model = checkpoint['model']
            best_loss = checkpoint['loss']
            if best_loss < 0.4:
                # 为了防止loss由于best_loss太低导致 loss一直无法下降到best_loss而导致best_checkpoint存不下来
                best_loss = 0.4
        else:
            checkpoint = torch.load(checkpoint)
            model = Tacotron2(config)
            model.load_state_dict(checkpoint)
            optimizer = Tacotron2Optimizer(
                torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.9, 0.999), eps=1e-6))
        print(model)
    logger = get_logger()
    print(f'learning rate is',optimizer.lr)

    model = model.to(config.device)

    criterion = Tacotron2Loss()

    collate_fn = TextMelCollate(config.n_frames_per_step)

    # Custom dataloaders
    if args.dataset == 'biaobei':
        training_files = args.dataset + '_filelist/' + args.dataset + '_audio_text_train_filelist.txt'
        validation_files = args.dataset + '_filelist/' + args.dataset + '_audio_text_valid_filelist.txt'
    else:
        training_files = args.dataset + '_filelist/' + args.dataset + '_audio_text_train_filelist.json'
        validation_files = args.dataset + '_filelist/' + args.dataset + '_audio_text_valid_filelist.json'
    
    train_dataset = TextMelLoader(training_files, config, dataset=args.dataset) 
    print('batch size is ', args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                               pin_memory=True, shuffle=True, num_workers=args.num_workers)
    print(f'loaded dataset from {training_files}')
    valid_dataset = TextMelLoader(validation_files, config, dataset=args.dataset)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)
    print(f'loaded dataset from {validation_files}')
    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        losses = AverageMeter()
        for i, batch in enumerate(train_loader):
            model.train()
            model.zero_grad()
            x, y = model.parse_batch(batch)

            # Forward prop.
            y_pred = model(x)

            # loss
            loss = criterion(y_pred, y)

            # Back prop.
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Keep track of metrics
            losses.update(loss.item())
            torch.cuda.empty_cache()
            
            writer.add_scalar('model/train_loss', losses.val, optimizer.step_num)

            # Print status
            if i % args.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))
            # validation
            if i % config.validation_steps == 0 and i != 0:
                valid_losses = AverageMeter()
                model.eval()
                lr = optimizer.lr
                step_num = optimizer.step_num
                print('\nLearning rate: {}'.format(lr))
                writer.add_scalar('model/learning_rate', lr, step_num)
                print('Step num: {}\n'.format(step_num))
                with torch.no_grad():
                    for batch in valid_loader:
                        model.zero_grad()
                        x, y = model.parse_batch(batch)
                        # Forward prop.
                        y_pred = model(x)
                        loss = criterion(y_pred, y)
                        # Keep track of metrics
                        valid_losses.update(loss.item())
                valid_loss = valid_losses.avg
                writer.add_scalar('model/valid_loss', valid_loss, step_num)
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Validation Loss {loss:.4f}'.format(epoch, i, len(train_loader), loss=valid_loss))

                # Check if there was an improvement
                is_best = valid_loss < best_loss
                best_loss = min(valid_loss, best_loss)
                if not is_best:
                    steps_since_improvement += config.validation_steps
                    print("\nSteps since last improvement: %d\n" % (steps_since_improvement,))
                else:
                    steps_since_improvement = 0

                # saving checkpoint and update the best checkpoint                             
                save_checkpoint(epoch, step_num, steps_since_improvement, model, optimizer, best_loss, is_best, dataset=args.dataset,trial_type=args.trial_type)
                # drawing alignment
                img_align = test(model, step_num, valid_loss)
                writer.add_image('model/alignment', img_align, step_num, dataformats='HWC')

def parse_args():
    parser = argparse.ArgumentParser(description='Tacotron2')
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--max_norm', default=1, type=float, help='Gradient norm threshold to clip')
    # trial type
    parser.add_argument('--trial_type', type=str, default='new', help='new vaocal dict or old vaocal dict')
    parser.add_argument('--load_type', type=str, default='dict', help='method to load model')
    # dataset 
    parser.add_argument('--dataset', type=str, default='aixia', help='name of dataset')
    # minibatch
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers to generate minibatch')
    # logging
    parser.add_argument('--print_freq', default=10, type=int, help='Frequency of printing training information')
    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
    parser.add_argument('--l2', default=1e-6, type=float, help='weight decay (L2)')
    parser.add_argument('--checkpoint', type=str, default='biaobei_checkpoints/biaobei.tar', help='checkpoint')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

