import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import math
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
from torch.utils.data import Subset

from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from convert_objax_to_pytorch import load_resnet18_from_objax
from resnet import resnet18
from custom_lr_scheduler import CosLR, FixedLR

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Opacus DP ImageNet Training')
parser.add_argument('--model_dir', metavar='DIR', default=None,
                    help='Model directory.')
parser.add_argument('--keep_ckpts', default=4, type=int,
                    help='Number of checkpoints to keep. (default: 4)')
parser.add_argument('--train_device_batch_size', default=128, type=int,
                    help='Per-device training batch size. (default: 128)')
parser.add_argument('--grad_acc_steps', default=1, type=int,
                    help='Number of steps for gradients accumulation, used to simulate large batches. (default: 1)')
parser.add_argument('--eval_device_batch_size', default=250, type=int,
                    help='Per-device eval batch size. (default: 250)')
parser.add_argument('--eval_every_n_steps', default=1000, type=int,
                    help='How often to run eval. (default: 1000)')
parser.add_argument('--num_train_epochs', default=10, type=int,
                    help='Number of training epochs. (default: 10)')
parser.add_argument('--base_learning_rate', default=2.0, type=float,
                    help='Base learning rate. (default: 2.0)')
parser.add_argument('--lr_warmup_epochs', default=1.0, type=float,
                    help='Number of learning rate warmup epochs. (default: 1.0)')
parser.add_argument('--lr_schedule', default='cos', type=str,
                    help='Learning rate schedule: "cos" or "fixed" (default: cos)')
parser.add_argument('--optimizer', default='momentum', type=str,
                    help='Optimizer to use: "momentum" or "adam" (default: momentum)')
parser.add_argument('--rnd_seed', default=0, type=int,
                    help='Initial random seed (default: 0)')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay (L2 loss) coefficient. (default: 1e-4)')
parser.add_argument('--model', default='resnet18', type=str,
                    help='Model to use. (default: resnet18)')
parser.add_argument('--data_dir', default='imagenet/imagenet-data', type=str,
                    help='Data directory. (default: imagenet/imagenet-data)')
parser.add_argument('--disable_dp', action='store_true',
                    help='If true then train without DP. (default: false)')
parser.add_argument('--dp_sigma', default=0.00001, type=float,
                    help='DP noise multiplier. (default: 0.00001)')
parser.add_argument('--dp_clip_norm', default=1.0, type=float,
                    help='DP gradient clipping norm. (default: 1.0)')
parser.add_argument('--dp_delta', default=1e-6, type=float,
                    help='DP-SGD delta for eps computation. (default: 1e-6)')
parser.add_argument('--finetune_path', default=None, type=str,
                    help='Path to checkpoint which is used as finetuning initialization.')
parser.add_argument('--finetune_cut_last_layer', default=True, type=bool,
                    help='If True then last layer will be cut for finetuning. (default: True)')
parser.add_argument('--num_layers_to_freeze', default=0, type=int,
                    help='Number of layers to freeze for finetuning. (default: 0)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--dist_url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--port', default='12355', type=str,
                    help='port used for distributed training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.rnd_seed is not None:
        random.seed(args.rnd_seed)
        torch.manual_seed(args.rnd_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    args.total_batch_size = args.train_device_batch_size * ngpus_per_node

    if args.multiprocessing_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    args.rank = 0

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.model))
    if args.model == "resnet18":
        model = resnet18()

    # Load pretrained model
    model = load_resnet18_from_objax(model, np.load(args.finetune_path), finetune_cut_last_layer=args.finetune_cut_last_layer)

    # Freeze layer [:args.num_layers_to_freeze]
    layer_id = 0
    prev_major_minor_id = (0, 0)
    in_first_or_last = True
    for name, param in model.named_parameters():
        if "layer" not in name:
            if not in_first_or_last:
                layer_id += 1
                in_first_or_last = True
        else:
            in_first_or_last = False
            major_minor_id = (int(name.split(".")[0][5:]), int(name.split(".")[1]))
            
            if prev_major_minor_id != major_minor_id:
                layer_id += 1
            prev_major_minor_id = major_minor_id
        
        if layer_id < args.num_layers_to_freeze:
            param.requires_grad = False

    # Print model
    if args.rank == 0:
        print(model)
        for name, param in model.named_parameters():
            print(name.ljust(30), " | ", str(param.shape).ljust(30), " | ", "" if param.requires_grad else "Freeze")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs of the current node.
    args.batch_size = int(args.train_device_batch_size)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if not args.disable_dp:
        model = DPDDP(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    args.base_learning_rate = args.base_learning_rate * args.total_batch_size * args.grad_acc_steps / 256

    # no_decay = ["bias", "gn", "GroupNorm"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    #         "weight_decay": 0.0,
    #     },
    # ]
    if args.optimizer == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), args.base_learning_rate,
                                    momentum=args.momentum, nesterov=True)
                                    # weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.base_learning_rate)
                                    #  weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {args.optimizer}')
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_schedule == "cos":
        scheduler = CosLR(optimizer, args.base_learning_rate, args.lr_warmup_epochs, args.num_train_epochs)
    elif args.lr_schedule == "fixed":
        scheduler = FixedLR(optimizer, args.base_learning_rate, args.lr_warmup_epochs)
    else:
        raise ValueError(f'Unsupported LR schedule: {args.lr_schedule}')
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0


    # Data loading code
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed and args.disable_dp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size if args.disable_dp else args.batch_size * args.ngpus_per_node * args.grad_acc_steps, num_workers=args.workers,
        sampler=train_sampler, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_device_batch_size, shuffle=False, num_workers=args.workers,
        sampler=val_sampler, pin_memory=True)

    if not args.disable_dp:
        privacy_engine = PrivacyEngine()

        # PrivacyEngine looks at the model's class and enables
        # distributed processing if it's wrapped with DPDDP
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon = 10.0,
            target_delta = args.dp_delta,
            epochs = args.num_train_epochs,
            max_grad_norm=args.dp_clip_norm,
        )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    os.system(f"taskset -pc 40-79 {os.getpid()}")
    train(train_loader, val_loader, model, criterion, optimizer, scheduler, privacy_engine, args)


def train(train_loader, val_loader, model, criterion, optimizer, scheduler, privacy_engine, args):
    global best_acc1

    with BatchMemoryManager(
        data_loader=train_loader, max_physical_batch_size=args.train_device_batch_size, optimizer=optimizer
    ) as new_train_loader:
        for epoch in range(args.start_epoch, args.num_train_epochs):
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(new_train_loader),
                [batch_time, data_time, losses, top1, top5],
                privacy_engine, args.dp_delta,
                prefix="Epoch: [{}]".format(epoch))

            # switch to train mode
            model.train()

            end = time.time()
            if args.rank == 0:
                pbar = tqdm(range(len(new_train_loader)))

            for i, (images, target) in enumerate(new_train_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                
                # compute weight decay loss
                with torch.enable_grad():
                    wd_loss = args.weight_decay * 0.5 * torch.stack([
                        (p.data ** 2).sum() for n, p in model.named_parameters() if "weight" in n and "gn" not in n and p.requires_grad == True
                        ]).sum()
                loss = criterion(output, target) + wd_loss

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step-based learning scheduler
                scheduler.step(epoch + i / len(new_train_loader))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if args.rank == 0:
                    pbar.update()

                if i % args.print_freq == 0:
                    progress.display(i + 1)


            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % args.ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, is_best)


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()

            if args.rank == 0:
                pbar = tqdm(range(len(loader)))

            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if args.rank == 0:
                    pbar.update()

                # if i % args.print_freq == 0:
                #     progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, privacy_engine=None, delta=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.privacy_engine = privacy_engine
        self.delta = delta

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.privacy_engine:
            entries += ["Epsilon", f"{self.privacy_engine.get_epsilon(self.delta):.4f}", "delta", f"{self.delta:e}"]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()