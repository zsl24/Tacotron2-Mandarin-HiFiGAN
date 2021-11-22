import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import config
from data_gen import TextMelLoader, TextMelCollate
from taco2models.loss_function import Tacotron2Loss
from taco2models.models import Tacotron2
from taco2models.optimizer import Tacotron2Optimizer
from utils_1 import parse_args, save_checkpoint, AverageMeter, get_logger, test


import os

os.environ['CUDA_VISIBLE_DEVICES'] ='7'

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
        # model
        model = Tacotron2(config)
        print(model)
        # model = nn.DataParallel(model)

        # optimizer
        optimizer = Tacotron2Optimizer(
            torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.9, 0.999), eps=1e-6))

    else:
        print(checkpoint)
        load_mode = 'dict'
        if load_mode == 'dict':
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            step = checkpoint['step']
            steps_since_improvement = checkpoint['steps_since_improvement']
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            optimizer.lr = 1e-4
        else:
            checkpoint = torch.load(checkpoint)
            start_epoch = 0
            model = Tacotron2(config)
            model.load_state_dict(checkpoint)
            optimizer = Tacotron2Optimizer(
                torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.9, 0.999), eps=1e-6))
        print(model)
    logger = get_logger()

    # Move to GPU, if available
    model = model.to(config.device)

    criterion = Tacotron2Loss()

    collate_fn = TextMelCollate(config.n_frames_per_step)

    # Custom dataloaders
    # 创建一个能够访问(text,mel) pair的具体文件的接口，text是以一维数组的形式，mel是以二维数组的形式
    train_dataset = TextMelLoader(config.training_files, config) 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                               pin_memory=True, shuffle=True, num_workers=args.num_workers)
    valid_dataset = TextMelLoader(config.validation_files, config)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)
    
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
            
            # 记录以供tensorboard查看
            writer.add_scalar('model/train_loss', losses.val, optimizer.step_num)

            # Print status
            if i % args.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))
                
            # validation
            if i % config.validation_steps == 0:
                valid_losses = AverageMeter()
                model.eval()
                lr = optimizer.lr
                step_num = optimizer.step_num
                print('\nLearning rate: {}'.format(lr))
                writer.add_scalar('model/learning_rate', lr, step_num)
                
                print('Step num: {}\n'.format(step_num))
        
                with torch.no_grad(): #试试这里，试完记得还原train的代码
                    for batch in valid_loader:
                        #model.zero_grad()
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

                # Save checkpoint                              
                save_checkpoint(epoch, step_num, steps_since_improvement, model, optimizer, best_loss, is_best)
                
                # Show alignments
                img_align = test(model, step_num, valid_loss)
                writer.add_image('model/alignment', img_align, step_num, dataformats='HWC')

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
