# LFW evaluation
import argparse
import json
import torch
import random
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import os
from torchvision.utils import save_image
from utils.utils_eval import read_pairs, get_paths, evaluate
from face_embedder import PrivFaceEmbedder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import logging

dir_home = os.path.expanduser("~")
dir_facenet = os.path.dirname(os.path.realpath(__file__))

from face.face_recognizer import get_recognizer
from utils.loss_functions import triplet_loss, lpips_loss
from torch.nn import TripletMarginWithDistanceLoss
from utils.image_processing import Obfuscator, input_trans, normalize
import config.config as c
import sys
sys.path.append(os.path.join(c.PROJECT_DIR, 'SimSwap'))

from embedder import *

import modules.Unet_common as common

dwt = common.DWT()
iwt = common.IWT()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
lpips_loss.to(device)
perc_triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y : lpips_loss(x, y), margin=0.5)


input_trans = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])



# def normalize(x: torch.Tensor):
#     x_norm = x.add(1.0).mul(0.5)
#     return x_norm



def adaptive_normalize(x: torch.Tensor):
    _min = x.min()
    _max = x.max()
    x_norm = (x - _min) / (_max - _min)
    return x_norm


def proc_for_rec(img_batch, zero_mean=False, resize=0, grayscale=False):
    _res = img_batch
    if zero_mean:
        _res = img_batch.sub(0.5).mul(2.0)
    if resize and resize != img_batch.shape[-1]:
        _res = F.resize(_res, size=[resize, resize])
    if grayscale:
        _res = F.rgb_to_grayscale(_res)
    return _res


def gauss_noise(shape):
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)
    return noise


def run_eval(embedder, recognizer, obfuscator, dataloader, path_list, issame_list, target_set, out_dir, model_name):
    file_paths = []
    classes = []
    embeddings_list_orig = []
    embeddings_list_proc = []
    embeddings_list_obfs = []

    triplet_losses = []
    privacy_scores = []
    lpips_scores   = []

    obf_name = obfuscator.name

    # idx_list = [2, 16, 64, 67, 76, 88, 92, 94, 166, 266, 366, 466, 566, 666, 766, 866, 966]

    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader):
            batch_idx += 1

            # if batch_idx > 200:
            #     break

            xb, (paths, yb) = batch

            _bs, _, _w, _h = xb.shape
            # password = torch.randint(0, 2, (_bs, 1, _w, _h))
            # xb = torch.cat((xb, password), dim=1)
            xb = xb.to(device)

            # password = torch.randint(0, 2, (_bs, 4, _w // 2, _h // 2)).mul(2).sub(1).to(device)
            # password = torch.randint(0, 2, (_bs, 32, 1, 1)).mul(4).sub(2).repeat(1, 1, _w // 2, _h // 2).to(device)
            # password_img = torch.randint(0, 2, xb.shape).mul(2).sub(1).to(device)
            password_img = torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device)
            password = dwt(password_img)

            file_paths.extend(paths)

### 袁霖备注 ####################################
# 参考这一段代码，先通过obfuscator()获得预处理的图像，然后再通过embedder()获得最终图像xb_proc, 其中，embedder()的定义在312-317行
            if obf_name in ['blur', 'pixelate', 'medianblur', 'hybrid']:
                xb_obfs = obfuscator(xb)
                # xb_proc = embedder(xb, xb_obfs)
                xb_out_z, xb_proc = embedder(xb, xb_obfs, password)
            else:
                # Randomly sample a target image and apply face swapping
                # num_targ_imgs = len(target_set)
                # targ_img_idx = random.randint(0, num_targ_imgs - 1)
                targ_img, _ = target_set[batch_idx]
                targ_img_batch = targ_img.repeat(xb.shape[0], 1, 1, 1).to(device)
                xb_obfs = obfuscator.swap(xb, targ_img_batch)
                xb_out_z, xb_proc = embedder(xb, xb_obfs, password)

            xb_rev, xb_obfs_rev = embedder(password.repeat(1, 4, 1, 1), xb_proc, password, rev=True)

            password_img_wrong = torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device)
            password_wrong = dwt(password_img_wrong)
            xb_rev_wrong, xb_obfs_rev_wrong = embedder(password_wrong.repeat(1, 4, 1, 1), xb_proc, password_wrong,
                                                       rev=True)

            # # Change only one bit position of the original password
            # password_img_wrong_part = password_img.clone()
            # password_img_wrong_part[:, :, 0:_w//2, 0:_h//2] *= -1
            # password_wrong_part = dwt(password_img_wrong_part)
            # xa_rev_wrong_part, xa_obfs_rev_wrong_part = embedder(password_wrong_part.repeat(1, 4, 1, 1), xb_proc,
            #                                       password_wrong_part,
            #                                          rev=True)

            xb_proc_clamp = torch.clamp(xb_proc, -1, 1)

            # privacy_cost = perc_triplet_loss(xb_targ, xb_proc_clamp, xb).to('cpu')
            dist_protcted = lpips_loss(xb_obfs, xb_proc_clamp).to('cpu')
            dist_original = lpips_loss(xb_obfs, xb).to('cpu')
            privacy_scores.append(dist_protcted / dist_original)
            lpips_scores.append(dist_original)

            obfs = obfuscator.name
            if batch_idx % 100 == 0:
                # save_image(normalize(targ_img), f"{out_dir}/Eval_{model_name}_batch{batch_idx}_targ.jpg")
                save_image(normalize(xb), f"{out_dir}/{model_name}_{batch_idx}_orig.jpg", nrow=4)
                save_image(normalize(xb_obfs), f"{out_dir}/{model_name}_{batch_idx}_{obfs}.jpg", nrow=4)
                save_image(normalize(xb_proc_clamp), f"{out_dir}/{model_name}_{batch_idx}_proc.jpg", nrow=4)
                save_image(normalize(xb_rev), f"{out_dir}/{model_name}_{batch_idx}_orig_rev.jpg", nrow=4)
                save_image(normalize(xb_rev_wrong, True), f"{out_dir}/{model_name}_{batch_idx}_orig_rev_wrong.jpg",
                           nrow=4)
                save_image(normalize(xb_obfs_rev), f"{out_dir}/{model_name}_{batch_idx}_proc_rev.jpg", nrow=4)
                save_image(normalize(xb_obfs_rev_wrong, True), f"{out_dir}/{model_name}_"
                                                               f"{batch_idx}_proc_rev_wrong.jpg", nrow=4)
                # save_image(normalize(xa_rev_wrong_part), f"{out_dir}/{model_name}_{batch_idx}_rev_orig_wrong_part.jpg",
                #            nrow=4)
                # save_image(normalize(xa_obfs_rev_wrong_part), f"{out_dir}/{model_name}_{batch_idx}_rev_pixelate_wrong_part.jpg",
                #            nrow=4)

                # xb_proc_clamp_dwt = dwt(xb_proc_clamp)
                # xb_proc_clamp_dwt_low = xb_proc_clamp_dwt.narrow(1, 0, c.channels_in)
                # save_image(adaptive_normalize(xb_proc_clamp_dwt_low),
                #            f"{out_dir}/{model_name}_{batch_idx}_proc_dwt_low.jpg",
                #            nrow=4)

            orig_embeddings = recognizer(recognizer.resize(xb))
            proc_embeddings = recognizer(recognizer.resize(xb_proc_clamp))
            obfs_embeddings = recognizer(recognizer.resize(xb_obfs))

            # negative_indexes = get_batch_negative_index(yb.tolist())
            # anchor = proc_embeddings
            # positive = orig_embeddings
            # negative = proc_embeddings[negative_indexes]
            # loss_triplet = triplet_loss(anchor, positive, negative)
            # triplet_losses.append(float(loss_triplet.to('cpu')))

            orig_embeddings = orig_embeddings.to('cpu').numpy()
            proc_embeddings = proc_embeddings.to('cpu').numpy()
            obfs_embeddings = obfs_embeddings.to('cpu').numpy()
            classes.extend(yb.numpy())
            embeddings_list_orig.extend(orig_embeddings)
            embeddings_list_proc.extend(proc_embeddings)
            embeddings_list_obfs.extend(obfs_embeddings)

    embeddings_dict_orig = dict(zip(file_paths, embeddings_list_orig))
    embeddings_dict_proc = dict(zip(file_paths, embeddings_list_proc))
    embeddings_dict_obfs = dict(zip(file_paths, embeddings_list_obfs))

    # %%
    # embeddings_list_o2p = []
    embeddings_list_p2o_ordered = []
    embeddings_list_obfs_ordered = []
    for path_a, path_b in zip(path_list[0::2], path_list[1::2]):
        # embeddings_list_o2p.append(embeddings_dict_orig[path_a])
        # embeddings_list_o2p.append(embeddings_dict_proc[path_b])
        embeddings_list_p2o_ordered.append(embeddings_dict_proc[path_a])
        embeddings_list_p2o_ordered.append(embeddings_dict_orig[path_b])
        embeddings_list_obfs_ordered.append(embeddings_dict_obfs[path_a])
        embeddings_list_obfs_ordered.append(embeddings_dict_orig[path_b])
    # embeddings_list_o2p = np.array(embeddings_list_o2p)
    embeddings_list_p2o_ordered = np.array(embeddings_list_p2o_ordered)
    embeddings_list_orig_ordered = np.array([embeddings_dict_orig[path] for path in path_list])
    embeddings_list_proc_ordered = np.array([embeddings_dict_proc[path] for path in path_list])
    embeddings_list_obfs_ordered = np.array(embeddings_list_obfs_ordered)

    test_cases = [
        ('Original', embeddings_list_orig_ordered, 'r-'),
        ('In-domain', embeddings_list_proc_ordered, 'g--'),
        ('Cross-domain', embeddings_list_p2o_ordered, 'b-.'),
        ('Obfuscated', embeddings_list_obfs_ordered, 'k:'),
    ]

    plt.clf()
    plt.figure(figsize=(4, 4))
    for case, embedding_list, line_style in test_cases:
        tpr, fpr, roc_auc, eer, accuracy, precision, recall, tars, tar_std, fars, bts = \
            evaluate(embedding_list, issame_list, distance_metric=1)

        result_dict = dict(
            tpr=list(tpr),
            fpr=list(fpr),
            roc_auc=roc_auc,
            eer=eer,
            acc=list(accuracy),
            precision=list(precision),
            recall=list(recall),
            tars=list(tars),
            tar_std=list(tar_std),
            fars=list(fars),
            bts=list(bts),
            lpips=str(np.mean(lpips_scores)),
        )
        json_file = f'{out_dir}/json_{model_name}_{case}.json'
        with open(json_file, "w") as f:
            json.dump(result_dict, f)

        acc, thres = np.mean(accuracy), np.mean(bts)
        result_msg = '{}：\n' \
                     '    ACC: {:.4f} | THRES: {:.4f} | AUC: {:.4f} | EER: {:.4f} | TARs: {} | FARs: {} | PS: {:.4f} ' \
                     '| LPIPS: {:.4f}'. \
            format(case, acc, thres, roc_auc, eer,
                   '/'.join([str(round(i, 3)) for i in tars]),
                   '/'.join([str(round(i, 3)) for i in fars]),
                   np.mean(privacy_scores), np.mean(lpips_scores))
        logging.info(result_msg)
        print(result_msg)
        plt.plot(fpr, tpr, line_style, label=case)

    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.title(model_name)
    # plt.legend(fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(f'{out_dir}/roc_{model_name}.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.show()

    # print('AVG. Triplet Loss:', np.mean(triplet_losses))
    print('AVG. Privacy Score:', np.mean(privacy_scores))


def prepare_eval_data(data_dir, data_pairs, transform, dataset_name='lfw', batch_size=8):
    workers = 0 if os.name == 'nt' else 8
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # overwrites class labels in dataset with path so path can be used for saving output
    dataset.samples = [
        (p, (p, idx))
        for p, idx in dataset.samples
    ]
    pairs = read_pairs(data_pairs)
    path_list, issame_list = get_paths(data_dir, pairs, dataset_name)
    test_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    return test_loader, path_list, issame_list


def main(embedder_path, rec_name, data_dir, data_pairs, obfuscator, out_dir, targ_img_path=None, dataset_name='lfw'):
    embedder_basename = os.path.basename(embedder_path)
    # filename, _ = os.path.splitext(embedder_basename)
    filename = rec_name + '_' + dataset_name

    # Load pretrained embedder and recognizer model

    #### Define the models
    # embedder = PrivFaceEmbedder().to(device)
    # if embedder_model_path:
    #     embedder.load_state_dict(torch.load(embedder_model_path))

    # Embedder net
    # embedder = Model()
    embedder = ModelDWT()
    embedder.to(device)
    embedder_state_dict = torch.load(embedder_path)
    # embedder_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    embedder.load_state_dict(embedder_state_dict)
    embedder.eval()

    recognizer = get_recognizer(rec_name)
    recognizer.to(device).eval()

    # Test config:
    test_loader, path_list, issame_list = prepare_eval_data(data_dir, data_pairs, input_trans, dataset_name)

    if obfuscator.name in ['blur', 'pixelate', 'medianblur', 'hybrid']:
        target_set = None
    else:
        target_set = datasets.ImageFolder(targ_img_path, transform=obfuscator.targ_img_trans)

    # dataset_target = datasets.ImageFolder(targ_img_path, transform=input_trans)
    run_eval(embedder, recognizer, obfuscator, test_loader, path_list, issame_list, target_set, out_dir, filename)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embedder_path', type=str, help="Path to trained face embedder.")
    parser.add_argument('-f', '--recognizer_name', type=str, default='MobileFaceNet',
                        help="Name of the face recognizer.")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, filename="eval/pixelate_test.log")
    embedder_dir = f'{dir_facenet}/model/checkpoints_test/'
    embedders = [
        # {'name': 'IResNet100', 'path': '/home/yuanlin/Projects/ProFaceInv/runs/Sep05_14-20-42_YL1_medianblur_17_IResNet100/checkpoints/medianblur_17_IResNet100_ep2_iter2000.pth'},
        # {'name': 'MobileFaceNet', 'path': os.path.join(embedder_dir, 'pixelate_4_10_MobileFaceNet_ep10_BEST.pth')},
        # {'name': 'InceptionResNet', 'path': os.path.join(embedder_dir, 'pixelate_4_10_InceptionResNetV1_ep11_BEST.pth')},
        # {'name': 'IResNet50', 'path': os.path.join(embedder_dir, 'pixelate_4_10_ResNet50_IR_ep8_BEST.pth')},
        # {'name': 'SEResNet50', 'path': os.path.join(embedder_dir, 'pixelate_4_10_SEResNet50_IR_ep8_BEST.pth')},
        # {'name': 'IResNet100', 'path': os.path.join(embedder_dir, 'pixelate_4_10_IResNet100_ep12_BEST.pth')},
        {'name': 'InceptionResNet', #'InceptionResNet', #'IResNet100',#'AdaFaceIR100',
         'path': '/home/yuanlin/Projects/ProFaceInv/runs/Dec27_06-36-21_YL1_hybrid_InceptionResNet/checkpoints/hybrid_InceptionResNet_ep30_iter10000.pth'},
    ]

    test_datasets = [
        {
            'name': 'lfw',
            # 'dir': os.path.join(dir_home, 'Datasets/LFW/LFW_align_crop_224_test_pairs'),
            # 'pairs': os.path.join(dir_home, 'Datasets/LFW/pairs.txt')
            'dir': os.path.join(dir_home, 'Datasets/LFW/LFW_112_test_pairs'),
            'pairs': os.path.join(dir_home, 'Datasets/LFW/pairs.txt')
        },
        # {
        #     'name': 'celeba',
        #     'dir': os.path.join(dir_home, 'Datasets/CelebA/align_crop_224/test_6000pairs'),
        #     'pairs': os.path.join(dir_home, 'Datasets/CelebA/align_crop_224/pairs.txt')
        # },
        # {
        #     'name': 'vggface2',
        #     'dir': os.path.join(dir_home, 'Datasets/VGG-Face2/data/test_6000pairs'),
        #     'pairs': os.path.join(dir_home, 'Datasets/VGG-Face2/data/pairs_2.txt')
        # }
    ]

    # data_dir_targets = os.path.join(dir_home, 'Datasets/CelebA/align_crop_224/test_frontal')

    # obfuscator = Obfuscator('pixelate', 4, 10)
    # obfuscator = Obfuscator('medianblur', 17)
    obfuscator = Obfuscator(c.obfuscator)
    obfuscator.eval()
    output_dir = 'eval' #'eval/pixelate_samples'
    for emd in embedders:
        for ds in test_datasets:
            print("Evaluating {} {}".format(emd['name'], ds['name']))
            logging.info("Evaluating {} {}".format(emd['name'], ds['name']))
            embedder_path = emd['path']
            rec_name = emd['name']
            dataset_name = ds['name']
            dataset_dir = ds['dir']
            dataset_pairs = ds['pairs']
            targ_img_path = c.target_img_dir_test
            main(embedder_path, rec_name, dataset_dir, dataset_pairs, obfuscator, output_dir, targ_img_path,
                 dataset_name)
