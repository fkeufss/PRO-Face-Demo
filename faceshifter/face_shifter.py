import torchvision.transforms as transforms
from faceshifter.face_modules.model import Backbone
from faceshifter.network.AEI_Net import *
from faceshifter.face_modules.mtcnn import *
import cv2
import numpy as np
from PIL import Image

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
detector = MTCNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
G = AEI_Net(c_id=512)
G.eval()
G.load_state_dict(torch.load('faceshifter/saved_models/G_latest.pth',
                             map_location=torch.device('cpu')))
G = G.to(device)
arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(
    torch.load('faceshifter/face_modules/model_ir_se50.pth', map_location=device),
    strict=False)


class FaceShifter:
    def __init__(self, tar):
        Xs_raw = cv2.imread(tar)

        try:
            Xs = detector.align(Image.fromarray(Xs_raw[:, :, ::-1]), crop_size=(256, 256))
        except Exception as e:
            print('the source image is wrong, please change the image')
        Xs = test_transform(Xs)
        Xs = Xs.unsqueeze(0).cuda()

        with torch.no_grad():
            self.embeds = arcface(
                F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))

    def face_shifter(self, Xt_raw):
        try:
            Xt, trans_inv = detector.align(Image.fromarray(Xt_raw[:, :, ::-1]), crop_size=(256, 256),
                                           return_trans_inv=True)
        except Exception as e:
            return Xt_raw
        Xt_raw = Xt_raw.astype(np.float) / 255.0
        Xt = test_transform(Xt)
        Xt = Xt.unsqueeze(0).cuda()

        mask = np.zeros([256, 256], dtype=np.float)
        for i in range(256):
            for j in range(256):
                dist = np.sqrt((i - 128) ** 2 + (j - 128) ** 2) / 128
                dist = np.minimum(dist, 1)
                mask[i, j] = 1 - dist
        mask = cv2.dilate(mask, None, iterations=20)

        with torch.no_grad():
            Yt, _ = G(Xt, self.embeds)
            Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5
            Yt = Yt[:, :, ::-1]
            Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)),
                                          borderValue=(0, 0, 0))
            mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
            mask_ = np.expand_dims(mask_, 2)
            Yt_trans_inv = mask_ * Yt_trans_inv + (1 - mask_) * Xt_raw
            return (Yt_trans_inv * 255).astype(np.uint8)


if __name__ == '__main__':
    faceshifter_tar = '/home/lyyang/papercode/PRO_Face/faceshifter/Leslie.jpg'
    faceshifter = FaceShifter(faceshifter_tar)
    frame = cv2.imread('/home/lyyang/papercode/PRO_Face/1.JPG')
    frame = faceshifter.face_shifter(frame)
    cv2.imwrite('/home/lyyang/papercode/PRO_Face/1_faceshifer.JPG', frame)