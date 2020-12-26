import datetime
import os
import re
import time

from os.path import join as ospj

import torch
from torch.utils.tensorboard import SummaryWriter

from options import get_train_parser
from data_loader import get_dataloader

from models.simple_model import Model

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def train(opts):
    print(f"Training on model {opts.model_name} ...")
    exper_dir = ospj(opts.exper_root, opts.model_name)
    os.makedirs(exper_dir, exist_ok=True)
    sample_dir = ospj(exper_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    ckpt_dir = ospj(exper_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = ospj(exper_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_log_dir = ospj(exper_dir, "tb_logs")  # tensorboard logs
    os.makedirs(tb_log_dir, exist_ok=True)
    # Dump options
    with open(ospj(exper_dir, "opts.txt"), "w") as f:
        for key, value in vars(opts).items():
            f.write(str(key) + ": " + str(value) + "\n")

    node_rank = 0

    if opts.multi_nodes:
        torch.distributed.init_process_group(backend="nccl")
        node_rank = torch.distributed.get_rank()

    master_node = opts.multi_nodes is False or node_rank == 0
    train_dataloader, train_sampler = get_dataloader(opts.data_root, opts.dataset_name, opts.image_size, 'train', opts.multi_nodes, opts.batch_size)

    val_dataloader = None
    if master_node:
        # For samples
        val_dataloader = get_dataloader(opts.data_root, opts.dataset_name, opts.image_size, 'val', opts.multi_nodes, opts.batch_size)

    prev_time = time.time()

    logfile, eval_score_logfile = None, None
    writer = None
    if master_node:
        logfile = open(ospj(exper_dir, "loss_log.txt"), 'w')
        writer = SummaryWriter(tb_log_dir)

    model = Model(opts)

    model_file_path = None
    if master_node:
        # Check saved checkpoints, if there are more than 1 pth file:
        if opts.fine_tune and opts.init_epoch > 0:
            # NOTE: model resume is not same as attr model resume
            if opts.azure:
                model_file_path = ospj(opts.model_name, opts.model_resume, str(len(opts.selected_attrs)), f"model_{opts.init_epoch}.pth")
            else:
                model_file_path = ospj(opts.model_resume, str(len(opts.selected_attrs)), f"model_{opts.init_epoch}.pth")
        checkfiles = os.listdir(ckpt_dir)
        n_checkfiles = len(checkfiles)
        if n_checkfiles > 1:
            checkfiles.sort(key=lambda f: int(re.sub('\D', '', f)))  # noqa
            save_latest = int(checkfiles[-1].split('.')[0].split('_')[1])
            if save_latest > opts.init_epoch:
                opts.init_epoch = save_latest
                model_file_path = ospj(ckpt_dir, f"model_{save_latest}.pth")
        if model_file_path:
            model.load_ckpt(model_file_path)

    model.parallel()
    model.summary_model()

    for epoch in range(opts.init_epoch, opts.epochs+1):
        if opts.multi_nodes:
            train_sampler.set_epoch(epoch)

        for batch_idx, data_batch in enumerate(train_dataloader):
            model.train()  # set status
            losses_dict = model.train_model(data_batch)

            batches_done = (epoch - opts.init_epoch) * len(train_dataloader) + batch_idx
            batches_left = (opts.epochs - opts.init_epoch) * len(train_dataloader) - batches_done
            batch_time = time.time() - prev_time
            time_left = datetime.timedelta(seconds=batches_left*batch_time)
            prev_time = time.time()

            if master_node:
                message = f"Epoch: {epoch}/{opts.epochs}, Batch: {batch_idx}/{len(train_dataloader)}, " \
                          f"Time: {batch_time:.2f}, ETA: {time_left}, "
                for k, v in losses_dict.items():
                    message += f"{k}: {v:.6f}, "
                print(message)
                logfile.write(message + '\n')
                for k, v in losses_dict.items():
                    writer.add_scalar(f"Loss/{k}", v, batches_done)
                writer.flush()

            if master_node and batches_done % opts.log_freq == 0:
                save_file = ospj(log_dir, f"train_epoch_{epoch}_batch_{batches_done}.png")
                model.save_training_image_sample(save_file)

            if master_node and batches_done % opts.sample_freq == 0:
                for val_batch_idx, val_data_batch in enumerate(val_dataloader):
                    # if val_batch_idx >= (10 // opts.batch_size):
                    #     break
                    save_file = ospj(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}_val_{val_batch_idx}.png")
                    model.predict(val_data_batch, save_file)

        if master_node and epoch % opts.ckpt_freq == 0:
            model_file_path = ospj(ckpt_dir, f"model_{epoch}.pth")
            model.save_ckpt(model_file_path)

            os.system("nvidia-smi")

        if epoch > opts.decay_epochs:
            model.update_lr()

    if master_node:
        model_file_path = ospj(ckpt_dir, f"model_{opts.epochs}.pth")
        model.save_ckpt(model_file_path)

    if master_node:
        logfile.flush()
        logfile.close()
        eval_score_logfile.flush()
        eval_score_logfile.close()
        writer.flush()
        writer.close()


if __name__ == "__main__":
    print("Enter training...")
    parser = get_train_parser()
    opts = parser.parse_args()
    train(opts)
