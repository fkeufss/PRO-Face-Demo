'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:26
Description:
'''

import cv2
import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import getSimSwapRes
from util.norm import SpecificNorm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_index = 0


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

class SimSwap:
    def __init__(self, pic_a):
        opt = TestOptions().parse()
        self.opt = opt

        crop_size = opt.crop_size
        self.crop_size = crop_size
        opt.Arc_path = 'SimSwap/arcface_model/arcface_checkpoint.tar'
        opt.checkpoints_dir = 'SimSwap/models/checkpoints'
        torch.nn.Module.dump_patches = False
        # if crop_size == 512:
        #     opt.which_epoch = 550000
        #     opt.name = '512'
        #     mode = 'ffhq'
        # else:
        mode = 'None'
        model = create_model(opt)
        model.eval()
        self.model = model
        spNorm = SpecificNorm()
        self.spNorm = spNorm

        app = Face_detect_crop(name='antelope', root='SimSwap/insightface_func/models')
        # app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 480), mode=mode)
        app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)
        self.app = app
        with torch.no_grad():
            img_a_whole = cv2.imread(pic_a)
            img_a_align_crop, _ = app.get(img_a_whole, crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()

            # create latent id
            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = model.netArc(img_id_downsample)
            self.latend_id = F.normalize(latend_id, p=2, dim=1)


    def simswap(self, img_b_whole):
        with torch.no_grad():
            img_b_align_crop_list, b_mat_list = self.app.get(img_b_whole, self.crop_size)
            # detect_results = None
            swap_result_list = []
            b_align_crop_tenor_list = []

            for b_align_crop in img_b_align_crop_list:

                b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                swap_result = self.model(None, b_align_crop_tenor, self.latend_id, None, True)[0]
                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)

            #     use_mask defult false
            # if self.opt.use_mask:
            #     n_classes = 19
            #     net = BiSeNet(n_classes=n_classes)
            #     net.cuda()
            #     save_pth = '/home/lyyang/papercode/PRO_Face/SimSwap/parsing_model/checkpoint/79999_iter.pth'
            #     net.load_state_dict(torch.load(save_pth))
            #     net.eval()
            # else:
            net =None

            simswapres = getSimSwapRes(b_align_crop_tenor_list,swap_result_list, b_mat_list, 224, img_b_whole,
                                       pasring_model =net,use_mask=False, norm = self.spNorm)
            return simswapres

def test_img():
    frame = cv2.imread('/home/lyyang/papercode/PRO_Face/6.JPG')
    frame = simswap.simswap(frame)
    cv2.imwrite('/home/lyyang/papercode/PRO_Face/6_simswap.JPG', frame)

def test_camera():
    import cv2 as cv

    tm = cv.TickMeter()
    cap = cv.VideoCapture(video_index)
    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        tm.start()
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        try:
            frame = simswap.simswap(frame)
        except:
            pass
        tm.stop()
        cv.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (20, 20), 0, 0.5, (255, 0, 0), 1)
        cv.imshow('a',frame)
        tm.reset()
        if cv2.waitKey(1) == 27:  # 按Esc退出
            break
    # print(tm.getFPS())
    cv2.destroyAllWindows()


if __name__ == '__main__':
    simswap_tar = 'SimSwap/Leslie.jpg'
    simswap = SimSwap(simswap_tar)
    test_camera()