import torch
import time
import random
import numpy as np
import os
import logging
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from utils.loss_functions import vgg_loss, l1_loss, triplet_loss, lpips_loss
from torch.nn import TripletMarginWithDistanceLoss
import modules.Unet_common as common
from utils.image_processing import normalize
import config.config as c
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dwt = common.DWT()
iwt = common.IWT()

class Logger(object):
    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, lr, i):
        track_str = '{} | {:5d}/{:<5d}| '.format(self.mode, i, self.length)
        loss_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in loss.items())
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        lr_str = 'current_lr = {}'.format(lr)
        # if i % SAVE_MODEL_INTERVAL == 0:
        logging.info(track_str + loss_str + '| ' + metric_str + '|  ' + lr_str)
        print('\r' + track_str + loss_str + '| ' + metric_str + '   ', end='')
        if i + 1 == self.length:
            logging.info('')
            print('')



class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.

    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=False):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred=(), y=()):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y

def get_random_sample(lst, execlude):
    '''
    Randomly select a sample (from an input list) that is different from execlude
    :param lst: input list
    :param execlude: the value to execlude
    :return: the selected number if there exist, otherwise none
    '''
    for i in range(len(lst)):
        element = random.sample(lst, 1)[0]
        if element != execlude:
            return element
    return None


def get_batch_negative_index(label_list):
    negative_indexes = []
    for i, label in enumerate(label_list):
        other_elements = list(np.delete(label_list, i))
        neg_label = get_random_sample(other_elements, label)
        neg_index = label_list.index(neg_label) if neg_label is not None else i
        negative_indexes.append(neg_index)
    return negative_indexes


def get_batch_triplet_index(batch_labels):
    import itertools
    batch_size = batch_labels.shape[0]
    all_pairs = itertools.permutations(range(batch_size), 2)
    pos_idx = []
    neg_idx = []
    for i, j in all_pairs:
        if not batch_labels.tolist()[i] == batch_labels.tolist()[j]:
            pos_idx.append(i)
            neg_idx.append(j)
    return pos_idx, neg_idx



def save_model(embedder, optimizer, dir_checkpoint, session, epoch, i_batch):
    model_name = f'{session}_ep{epoch}_iter{i_batch}'
    saved_path = f'{dir_checkpoint}/{model_name}.pth'
    # torch.save({'opt': optimizer.state_dict(), 'net': embedder.state_dict()}, saved_path)
    torch.save(embedder.state_dict(), saved_path)
    return saved_path



def gauss_noise(shape):
    # noise = torch.zeros(shape).cuda()
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        # noise[i] = torch.randn(noise[i].shape).cuda()
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


def load_model(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')




# def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
#     batch_size = unsigned_long.shape[0]
#     target_size = passwd_length // 4
#
#     binary = np.empty((batch_size, passwd_length), dtype=np.float32)
#     for idx in range(batch_size):
#         binary[idx, :] = np.array([int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])
#
#     dis_target = np.empty((batch_size, target_size), dtype=np.long)
#     for idx in range(batch_size):
#         tmp = unsigned_long[idx]
#         for byte_idx in range(target_size):
#             dis_target[idx, target_size - 1 - byte_idx] = tmp % 16
#             tmp //= 16
#     return binary, dis_target


def pass_epoch(embedder, recognizer, obfuscator, dataloader, target_set_train, session='',
               dir_image='./images', dir_checkpoint='./checkpoints', optimizer=None, scheduler=None, show_running=True,
               device='cpu', writer=None, epoch=0, max_batch=np.inf, logger_train=None):
    """Train or evaluate over a data epoch.

    Arguments:
        model {torch.nn.Module} -- Pytorch model.
        loss_fn {callable} -- A function to compute (scalar) loss.
        loader {torch.utils.data.DataLoader} -- A pytorch data loader.

    Keyword Arguments:
        optimizer {torch.optim.Optimizer} -- A pytorch optimizer.
        scheduler {torch.optim.lr_scheduler._LRScheduler} -- LR scheduler (default: {None})
        batch_metrics {dict} -- Dictionary of metric functions to call on each batch. The default
            is a simple timer. A progressive average of these metrics, along with the average
            loss, is printed every batch. (default: {{'time': iter_timer()}})
        show_running {bool} -- Whether or not to print losses and metrics for the current batch
            or rolling averages. (default: {False})
        device {str or torch.device} -- Device for pytorch to use. (default: {'cpu'})
        writer {torch.utils.tensorboard.SummaryWriter} -- Tensorboard SummaryWriter. (default: {None})

    Returns:
        tuple(torch.Tensor, dict) -- A tuple of the average loss and a dictionary of average
            metric values across the epoch.
    """

    mode = 'Train' if embedder.training else 'Valid'
    logger = Logger(mode, length=len(dataloader), calculate_mean=show_running)
    loss_image_vgg_total = 0
    loss_triplet_p2p_total = 0
    loss_triplet_p2o_total = 0
    loss_image_l1_total = 0
    loss_rec_total = 0
    loss_rec_wrong_total = 0
    loss_lf = 0
    loss_lf_total = 0
    loss_batch_total = 0
    metrics = {}
    batch_metrics = {
        'fps': BatchTimer(),
    }

    triplet_loss.to(device)
    lpips_loss.to(device)
    l1_loss.to(device)
    percep_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: lpips_loss(x, y), margin=1.0)

    models_saved = []
    i_batch = 0
    obf_name = obfuscator.name
    for i_batch, data_batch in enumerate(dataloader):
        if i_batch > max_batch:
            break

        a, n, p = data_batch
        xa, label_a = a
        xn, label_n = n
        xp, label_p = p

        _bs, _, _w, _h = xa.shape
        # password = torch.randint(0, 2, (_bs, 1, _w, _h))
        # xa = torch.cat((xa, password), dim=1) # Concate image with password
        # xn = torch.cat((xn, password), dim=1)
        # xp = torch.cat((xp, password), dim=1)
        xa = xa.to(device)
        xn = xn.to(device)
        xp = xp.to(device)

        if obf_name in ['blur', 'pixelate', 'medianblur', 'hybrid']:
            ## Perform image processing as target image
            xa_obfs = obfuscator(xa)
            xn_obfs = obfuscator(xn)
            xp_obfs = obfuscator(xp)
        else:
            num_targ_imgs = len(target_set_train)
            targ_img_idx = random.randint(0, num_targ_imgs - 1)
            targ_img, _ = target_set_train[targ_img_idx]
            targ_img_batch = targ_img.repeat(xa.shape[0], 1, 1, 1).to(device)
            xa_obfs = obfuscator.swap(xa, targ_img_batch)
            xn_obfs = obfuscator.swap(xn, targ_img_batch)
            xp_obfs = obfuscator.swap(xp, targ_img_batch)
            targ_img_batch.detach()
            xa_obfs.detach()
            xn_obfs.detach()
            xp_obfs.detach()
            if i_batch % c.SAVE_IMAGE_INTERVAL == 0:
                save_image(obfuscator.targ_img_trans_inv(targ_img),
                           f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_targ.jpg")

################################################################################################################
        ## Perform image protection via HiNet

        # rand_unsigned_long = np.random.randint(0, 2 ** 128, size=(4,), dtype=np.uint64)
        # password = unsigned_long_to_binary_repr(rand_unsigned_long, 128)

        password_img = torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device)
        password = dwt(password_img)
        # password = torch.randint(0, 2, (_bs, 32, 1, 1)).mul(4).sub(2).repeat(1, 1, _w // 2, _h // 2).to(device)
        xa_out_z, xa_proc = embedder(xa, xa_obfs, password)
        xn_out_z, xn_proc = embedder(xn, xn_obfs, password)
        xp_out_z, xp_proc = embedder(xp, xp_obfs, password)

        # random_z_guass = gauss_noise(xa_out_z.shape)
        # xa_rev, xa_obfs_rev = embedder(random_z_guass, xa_proc, rev=True)


        # random_z_guass = gauss_noise(xa_out_z.shape)
        # xa_proc[:, 3:, :, :] = password
        xa_rev, xa_obfs_rev = embedder(password.repeat(1, 4, 1, 1), xa_proc, password, rev=True)

        # xn_rev = embedder(xn_out_z, xn_steg, rev=True)
        # xp_rev = embedder(xp_out_z, xp_steg, rev=True)

        # Compute face embedding
        embed_orig_a = recognizer(recognizer.resize(xa))
        embed_proc_a = recognizer(recognizer.resize(xa_proc))
        embed_proc_n = recognizer(recognizer.resize(xn_proc))
        embed_proc_p = recognizer(recognizer.resize(xp_proc))

        # # For each person, find all other person's images as negative samples
        # pos_idx, neg_idx = get_batch_triplet_index(y)
        # anchor = proc_embedding[pos_idx]
        # positive = orig_embedding[pos_idx]
        # negative = proc_embedding[neg_idx]
        # loss_triplet = triplet_loss(anchor, positive, negative)

        # # For each person, randomly sample another person's image as negative sample
        # negative_indexes = get_batch_negative_index(y.tolist())
        # anchor = proc_embedding
        # positive = orig_embedding
        # negative = proc_embedding[negative_indexes]

        loss_triplet_p2p = triplet_loss(embed_proc_a, embed_proc_p, embed_proc_n)
        loss_triplet_p2o = triplet_loss(embed_proc_a, embed_orig_a, embed_proc_n)

        ## Three kinds of perceptual losses
        # loss_image_vgg = percep_triplet_loss(xa_obfs, xa_proc, xa)
        loss_image_vgg = lpips_loss(xa_obfs, xa_proc)
        loss_image_l1 = l1_loss(xa_obfs, xa_proc)
        loss_image = 5 * loss_image_vgg + loss_image_l1
        # loss_id = 0.5 * loss_triplet_p2p + 0.1 * loss_triplet_p2o
        # loss_id =  2.5 * loss_triplet_p2p + 0.5 * loss_triplet_p2o # AdaFace
        # loss_id =  5 * loss_triplet_p2p + loss_triplet_p2o # AdaFace
        _id_w1, _id_w2 = c.recognizer_weight[recognizer.name][obf_name]
        loss_id = _id_w1 * loss_triplet_p2p + _id_w2 * loss_triplet_p2o

        # loss_rec = reconstruction_loss(xa_rev, xa)
        loss_rec = l1_loss(xa_rev, xa)

        # proc_low = dwt(xa_proc).narrow(1, 0, c.channels_in)
        # obfs_low = dwt(xa_obfs).narrow(1, 0, c.channels_in)
        # loss_lf = low_frequency_loss(proc_low, obfs_low)

        password_img_wrong = torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device)
        password_wrong = dwt(password_img_wrong)
        # password_wrong = torch.randint(0, 2, (_bs, 32, 1, 1)).mul(4).sub(2).repeat(1, 1, _w // 2, _h // 2).to(device)

        xa_rev_wrong, _ = embedder(password_wrong.repeat(1, 4, 1, 1), xa_proc, password_wrong, rev=True)
        # loss_rec_wrong = 1 / l1_loss(xa_rev_wrong, xa_obfs)
        ## Make correctly recovered image closer to original while further to wrong recovered image
        loss_rec_wrong = percep_triplet_loss(xa, xa_rev, xa_rev_wrong)
        # loss_rec = l1_triplet_loss(xa, xa_rev, xa_rev_wrong)
        # loss_rec_wrong = l1_triplet_loss(xa, xa_rev, xa_rev_wrong)

        # rec_weight = c.recognizer_weight[recognizer.name][obf_name]
        loss_batch = loss_image + loss_id + loss_rec + loss_rec_wrong
        # loss_batch = 2 * loss_image_perc + loss_image_l1 + 0.5 * loss_triplet_p2p + 0.1 * loss_triplet_p2o
        # loss_batch = loss_image_vgg + loss_image_l1 + 0.5 * loss_triplet_p2p + 0.1 * loss_triplet_p2o

        # Save images
        if i_batch % c.SAVE_IMAGE_INTERVAL == 0:
            save_image(normalize(xa),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_orig.jpg", nrow=4)
            save_image(normalize(xa_proc),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_proc.jpg", nrow=4)
            save_image(normalize(xa_obfs),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_{obf_name}.jpg", nrow=4)
            save_image(normalize(xa_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev.jpg", nrow=4)
            save_image(normalize(xa_obfs_rev),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_obfs_rev.jpg", nrow=4)
            save_image(normalize(xa_rev_wrong, adaptive=True),
                       f"{dir_image}/{mode}_ep{epoch}_batch{i_batch}_rev_wrong.jpg", nrow=4)

        if embedder.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        # loss_history.append([loss_batch.item(), 0.])
        # image_loss_history.append([loss_image.item(), 0.])
        # rec_loss_history.append([loss_rec.item(), 0.])
        # id_loss_history.append([loss_id.item(), 0.])

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn().detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

        if writer is not None and embedder.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss_image_vgg', {mode: loss_image_vgg.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_triplet_p2p', {mode: loss_triplet_p2p.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_triplet_p2o', {mode: loss_triplet_p2o.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_image_l1', {mode: loss_image_l1.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec', {mode: loss_rec.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_rec_wrong', {mode: loss_rec_wrong.detach().cpu()}, writer.iteration)
                # writer.add_scalars('loss_lf', {mode: loss_lf.detach().cpu()}, writer.iteration)
                writer.add_scalars('loss_batch', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_image_vgg = loss_image_vgg.detach().cpu()
        loss_image_vgg_total += loss_image_vgg
        loss_triplet_p2p = loss_triplet_p2p.detach().cpu()
        loss_triplet_p2p_total += loss_triplet_p2p
        loss_triplet_p2o = loss_triplet_p2o.detach().cpu()
        loss_triplet_p2o_total += loss_triplet_p2o
        loss_image_l1 = loss_image_l1.detach().cpu()
        loss_image_l1_total += loss_image_l1
        loss_rec = loss_rec.detach().cpu()
        loss_rec_total += loss_rec
        loss_rec_wrong = loss_rec_wrong.detach().cpu()
        loss_rec_wrong_total += loss_rec_wrong
        # loss_lf = loss_lf.detach().cpu()
        # loss_lf_total += loss_lf
        loss_batch = loss_batch.detach().cpu()
        loss_batch_total += loss_batch


        if show_running:
            loss_log = {
                'loss_image_vgg': loss_image_vgg_total,
                'loss_image_l1': loss_image_l1_total,
                'loss_triplet_p2p': loss_triplet_p2p_total,
                'loss_triplet_p2o': loss_triplet_p2o_total,
                'loss_rec': loss_rec_total,
                'loss_rec_wrong': loss_rec_wrong_total,
                # 'loss_lf': loss_lf_total
            }
            logger(loss_log, metrics, optimizer.param_groups[0]['lr'], i_batch)
        else:
            loss_log = {
                'loss_image_vgg': loss_image_vgg,
                'loss_image_l1': loss_image_l1,
                'loss_triplet_p2p': loss_triplet_p2p,
                'loss_triplet_p2o': loss_triplet_p2o,
                'loss_rec': loss_rec_total,
                'loss_rec_wrong': loss_rec_wrong_total,
                # 'loss_lf': loss_lf_total
            }
            logger(loss_log, metrics_batch, optimizer.param_groups[0]['lr'], i_batch)

        # Save model every 5000 iteration
        if (i_batch > 0) and (i_batch % c.SAVE_MODEL_INTERVAL == 0) and (mode == 'Train'):
            saved_path = save_model(embedder, optimizer, dir_checkpoint, session, epoch, i_batch)
            models_saved.append(saved_path)

    # saved_path = save_model(embedder, dir_checkpoint, session, epoch, i_batch)
    # models_saved.append(saved_path)

    print('\n')
    if embedder.training and scheduler is not None:
        scheduler.step()

    # epoch_losses = np.mean(np.array(loss_history), axis=0)
    # image_epoch_losses = np.mean(np.array(image_loss_history), axis=0)
    # rec_epoch_losses = np.mean(np.array(rec_loss_history), axis=0)
    # id_epoch_losses = np.mean(np.array(id_loss_history), axis=0)
    # epoch_losses[1] = np.log10(optimizer.param_groups[0]['lr'])

    # logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    # logger_train.info(
    #     f"Train epoch {epoch}:   "
    #     f'Loss: {epoch_losses[0].item():.4f} | '
    #     f'image_Loss: {image_epoch_losses[0].item():.4f} | '
    #     f'rec_Loss: {rec_epoch_losses[0].item():.4f} | '
    #     f'id_Loss: {id_epoch_losses[0].item():.4f} | '
    # )


    loss_image_vgg_total = loss_image_vgg_total / (i_batch + 1)
    loss_triplet_p2p_total = loss_triplet_p2p_total / (i_batch + 1)
    loss_triplet_p2o_total = loss_triplet_p2o_total / (i_batch + 1)
    loss_image_l1_total = loss_image_l1_total / (i_batch + 1)
    loss_batch_total = loss_batch_total / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    if writer is not None and not embedder.training:
        writer.add_scalars('loss_image_vgg', {mode: loss_image_vgg_total.detach()}, writer.iteration)
        writer.add_scalars('loss_triplet_p2p_total', {mode: loss_triplet_p2p_total.detach()}, writer.iteration)
        writer.add_scalars('loss_triplet_p2o_total', {mode: loss_triplet_p2o_total.detach()}, writer.iteration)
        writer.add_scalars('loss_image_l1', {mode: loss_image_l1_total.detach()}, writer.iteration)
        writer.add_scalars('loss_batch', {mode: loss_batch_total.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss_batch_total, metrics, models_saved