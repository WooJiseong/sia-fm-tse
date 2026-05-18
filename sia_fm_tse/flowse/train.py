import sys
import time
import yaml
import pprint
import random
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import torchaudio
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.optim import AdamW
from accelerate import Accelerator
from utils.logger import get_logger


from model import DiT,CFM
from model.model_utils import get_tokenizer


import os
from loader.dataloader import make_auto_loader



import warnings

warnings.filterwarnings("ignore")

from torch import autograd

torch.autograd.set_detect_anomaly(True)

from datetime import datetime


def make_dataloader(opt):
    
    train_sampler, train_loader = make_auto_loader(
        **opt["datasets"]["train"],
        **opt["datasets"]["dataloader_setting"],
    )
    val_sampler, val_loader = make_auto_loader(
        **opt["datasets"]["val"],
        **opt["datasets"]["dataloader_setting"],
    )
    return train_sampler, train_loader, val_sampler, val_loader


def save_checkpoint(
    checkpoint_dir,
    nnet,
    optimizer,
    scheduler,
    epoch,
    best_loss,
    step=None,
    save_period=-1,
    best=True,
    logger=None,
):
    """
    Save checkpoint (epoch, model, optimizer, best_loss)
    """
    cpt = {
        "epoch": epoch,
        "model_state_dict": nnet.module.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),  
        "best_loss": best_loss,
    }
    cpt_name = "{0}.pt.tar".format("best" if best else "last")
    torch.save(cpt, checkpoint_dir / cpt_name)
    if not logger is None:
        logger.info(f"save checkpoint {cpt_name}")
    if not step is None:
        torch.save(cpt, checkpoint_dir / f"{epoch}_{step}.pt.tar")
    elif save_period > 0 and epoch % save_period == 0:
        torch.save(cpt, checkpoint_dir / f"{epoch}.pt.tar")
    
    


def make_optimizer(params, opt):
    supported_optimizer = {
        "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
        "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
        "adam": torch.optim.Adam,  # weight_decay, lr
        "adadelta": torch.optim.Adadelta,  # weight_decay, lr
        "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
        "adamax": torch.optim.Adamax,  # lr, weight
    }
    if opt["optim"]["name"] not in supported_optimizer:
        raise ValueError("Now only support optimizer {}".format(opt["optim"]["name"]))
    optimizer = supported_optimizer[opt["optim"]["name"]](
        params, **opt["optim"]["optimizer_kwargs"]
    )
    return optimizer


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return (
            obj.to(device, non_blocking=True) if isinstance(obj, torch.Tensor) else obj
        )

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [str(datetime.now()) + "\t"]
        entries += [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]


def adjust_learning_rate(optimizer, epoch, conf):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epcoh(
    train_loader,
    nnet,
    optimizer,
    scheduler,
    epoch,
    local_rank,
    conf,
    device,
    world_size,
    logger,
):
    if local_rank == 0:
        lr = get_learning_rate(optimizer)
        logger.info("set train mode, lr: {:.3e}".format(lr))
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    sisnr = AverageMeter("sisnr", ":.4f")
    cmse = AverageMeter("cmse", ":.4f")
    cost = AverageMeter("Cost", ":.4f")
    crossentropy = AverageMeter("Crossentropy", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time,
            data_time,
            losses,
            sisnr,
            cmse,
            cost,
            crossentropy,
        ],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )

    nnet.train()

    end = time.time()
    for i, egs in enumerate(train_loader):

        
        egs = load_obj(egs, device)
        data_time.update(time.time() - end)
        noisy = egs["noisy_mel"].transpose(-1,-2)
        label = egs["label_mel"].transpose(-1,-2)
        text = egs["text"]
    
        loss, _, _ = nnet(inp=noisy, clean=label,text=text)
    
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, world_size)
        losses.update(reduced_loss.item(), noisy.size(0))
        
        optimizer.zero_grad()
    
        loss.backward()

        clip_grad_norm_(nnet.parameters(), conf["optim"]["gradient_clip"])
        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % conf["logger"]["print_freq"] == 0 and local_rank == 0:
            progress.display(i)
            
        

    if local_rank == 0:
        progress.display(len(train_loader))


def validate_one_epcoh(
    val_loader,
    nnet,
    local_rank,
    conf,
    device,
    world_size,
    logger,
):
    if local_rank == 0:
        logger.info("set validate mode")
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    sisnr = AverageMeter("sisnr", ":.4f")
    cmse = AverageMeter("cmse", ":.4f")
    cost = AverageMeter("Cost", ":.4f")
    crossentropy = AverageMeter("Crossentropy", ":.4f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, sisnr, cmse, cost, crossentropy],
        prefix="Validation: ",
        logger=logger,
    )

    nnet.eval()

    resampler = torchaudio.transforms.Resample(orig_freq = 16000, new_freq = 24000).to(torch.device("cuda", local_rank))
    with torch.no_grad():
        end = time.time()
        for i, egs in enumerate(val_loader):
            
            
            egs = load_obj(egs, device)

            noisy = egs["noisy_mel"].transpose(-1,-2)
            label = egs["label_mel"].transpose(-1,-2)
            text = egs["text"]
            
            loss, _, _ = nnet.module.forward(inp=noisy, text=text, clean=label)
          
            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, world_size)

            losses.update(reduced_loss.item(), noisy.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % conf["logger"]["print_freq"] == 0 and local_rank == 0:
                progress.display(i)
    if local_rank == 0:
        progress.display(len(val_loader))

    return losses.avg


def main_worker(local_rank, args):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cudnn.benchmark = True
    world_size = dist.get_world_size()
    



    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    random.seed(conf['train']['seed'])
    np.random.seed(conf['train']['seed'])
    torch.cuda.manual_seed_all(conf['train']['seed'])
    checkpoint_dir = Path(conf["train"]["checkpoint"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    logger = get_logger(
        name=(
            (checkpoint_dir / "trainer.log").as_posix()
            if conf["logger"]["path"] is None
            else conf["logger"]["path"]
        ),
        file=True,
    )
    if local_rank == 0:
        logger.info("Arguments in args:\n{}".format(pprint.pformat(vars(args))))
        logger.info("Arguments in yaml:\n{}".format(pprint.pformat(conf)))
        with open(checkpoint_dir / "train.yaml", "w") as f:
            yaml.dump(conf, f)

    model_cls = DiT
    
    ## 
    tokenizer = conf['model']['tokenizer']

    ##
    tokenizer_path = conf['model']['tokenizer_path']
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    
    nnet = CFM(
        transformer=model_cls(**conf['model']['arch'], text_num_embeds=vocab_size,mel_dim=conf['model']['mel_spec']['n_mel_channels']),
        audio_drop_prob=conf['model']['audio_drop_prob'],cond_drop_prob=conf['model']['cond_drop_prob'],
        mel_spec_kwargs=conf['model']['mel_spec'],vocab_char_map=vocab_char_map
    )

    
    if local_rank == 0:
        num_params = sum([param.nelement() for param in nnet.parameters()]) / 10.0**6
        logger.info("model summary:\n{}".format(nnet))
        logger.info(f"#param: {num_params:.2f}M")
  
    start_epoch = 0
    end_epoch = conf["train"]["epoch"]

    if conf["train"]["resume"]:
        if not Path(conf["train"]["resume"]).exists():
            raise FileNotFoundError(
                f"Could not find resume checkpoint: {conf['train']['resume']}")
        

        else:
            cpt = torch.load(conf["train"]["resume"], map_location="cpu")
            if conf["train"]["rm_stft"]:
                for i in list(cpt["model_state_dict"]):
                    if i[: len("stft")] == "stft" or i[: len("istft")] == "istft":
                        del cpt["model_state_dict"][i]
            
            start_epoch = cpt["epoch"] + 1
            
            if local_rank == 0:
                logger.info(
                    f"resume from checkpoint {conf['train']['resume']}: epoch {start_epoch:d}"
                )
            nnet.load_state_dict(
                cpt["model_state_dict"], strict=conf["train"]["strict"]
            )
           
            nnet = nnet.cpu() if device == torch.device("cpu") else nnet.cuda()
    else:
        nnet = nnet.cpu() if device == torch.device("cpu") else nnet.cuda()

    nnet = DistributedDataParallel(
        nnet,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
   
    '''
    accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            gradient_accumulation_steps=conf['optim']['grad_accumulation_steps'],
    )
    '''
    
    optimizer = AdamW(nnet.parameters(), lr=conf['optim']['lr'])
    
    '''
    nnet, optimizer = accelerator.prepare(nnet, optimizer)
    accelerator.even_batches = False
    '''
    
    
    train_sampler, train_loader, val_sampler, val_loader = make_dataloader(conf)
 
    warmup_steps = (
        # conf['optim']['warm_up_step'] * accelerator.num_processes
        conf['optim']['warm_up_step'] * world_size
    )  

    total_steps = len(train_loader) * conf['optim']['max_epoch'] / conf['optim']['grad_accumulation_steps']
    decay_steps = total_steps - warmup_steps
    logger.info(f"warmup_steps:{warmup_steps}")
    logger.info(f"decay_steps:{decay_steps}")
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
    scheduler = SequentialLR(
        optimizer,schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
    )
    
    if conf["train"]["resume"] and not conf["train"]["reset_lr"]:
        if not Path(conf["train"]["resume"]).exists():
            raise FileNotFoundError(
                f"Could not find resume checkpoint: {conf['train']['resume']}"
            )
        cpt = torch.load(conf["train"]["resume"], map_location=device)
        
        optimizer.load_state_dict(cpt["optim_state_dict"])
        scheduler.load_state_dict(cpt["scheduler_state_dict"])  
        
    
    '''
    train_loader, scheduler = accelerator.prepare(
        train_loader, scheduler
    ) 
    '''
    
    best_loss = 10000
    no_impr = 0

    for epoch in range(start_epoch, end_epoch):
        logger.info(f"epoch:{epoch} train")
        train_one_epcoh(
            train_loader,
            nnet,
            optimizer,
            scheduler,
            epoch,
            local_rank,
            conf,
            device,
            world_size,
            logger,
        )
        logger.info(f"epoch:{epoch} val")
        cv_loss = validate_one_epcoh(
            val_loader,
            nnet,
            local_rank,
            conf,
            device,
            world_size,
            logger,
        )
        logger.info(f"epoch:{epoch} val done")
        if cv_loss < best_loss:
            best_loss = cv_loss
            no_impr = 0
            
            if local_rank == 0:
                save_checkpoint(
                    checkpoint_dir,
                    nnet,
                    optimizer,
                    scheduler,
                    epoch,
                    best_loss,
                    save_period=-1,
                    best=True,
                    logger=logger,
                )
            logger.info(f"epoch:{epoch} save best")
        else:
            no_impr += 1
            if local_rank == 0:
                logger.info(f"| no impr, best = {best_loss:.4f}")
                

        logger.info(f"epoch:{epoch} save")
        if local_rank == 0:
            save_checkpoint(
                checkpoint_dir,
                nnet,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                save_period=conf["train"]["save_period"],
                best=False,
                logger=logger,
            )

        if no_impr == conf["train"]["early_stop"]:
            if local_rank == 0:
                logger.info(f"stop training cause no impr for {no_impr:d} epochs")
            break
        logger.info(f"epoch:{epoch} done")


def run(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    main_worker(local_rank, args)


if __name__ == "__main__":
    # os.environ["NCCL_SOCKET_IFNAME"] = "en,eth,em,bond"

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "-conf", type=str, required=True, help="Yaml configuration file for training"
    )
    args = parser.parse_args()
    run(args)
