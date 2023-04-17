import sys
sys.path.append('proface')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
from torchvision import transforms
import os
from torchvision.utils import save_image

from PIL import Image


dir_home = os.path.expanduser("~")
dir_facenet = os.path.dirname(os.path.realpath(__file__))

# from facerecognize.face.face_recognizer import get_recognizer

from proface.utils.image_processing import Obfuscator, input_trans, normalize

from proface.embedder import *

import proface. modules.Unet_common as common
dwt = common.DWT()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


input_trans = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


obfuscator_medianblur = Obfuscator('medianblur_15')
obfuscator_medianblur.eval()

obfuscator_gaussianblur = Obfuscator('blur_21_6_10')
obfuscator_gaussianblur.eval()

obfuscator_pixelate = Obfuscator('pixelate_9')
obfuscator_pixelate.eval()

embedder_path = 'proface/hybrid_AdaFaceIR100_ep16_iter5000.pth'
embedder = ModelDWT()
embedder.to(device)
embedder_state_dict = torch.load(embedder_path, map_location=device)
embedder.load_state_dict(embedder_state_dict)
embedder.eval()

def get_obs_face(img, obfuscator):
    with torch.no_grad():
        xb = input_trans(img)
        xb = xb.repeat(1, 1, 1, 1)
        _bs, _, _w, _h = xb.shape
        xb = xb.to(device)

        if obfuscator == 'MedianBlur':
            xb_obfs = obfuscator_medianblur(xb)
        elif obfuscator == 'GaussianBlur':
            xb_obfs = obfuscator_gaussianblur(xb)
        elif obfuscator == 'Pixelate':
            xb_obfs = obfuscator_pixelate(xb)

        password_img = torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device)
        password = dwt(password_img)
        xb_out_z, xb_proc = embedder(xb, xb_obfs, password)
        xb_proc_clamp = torch.clamp(xb_proc, -1, 1)
        return xb_proc_clamp
def test():

    with torch.no_grad():
        xb = input_trans(Image.open('lfw112_Aaron_Eckhart_0001.jpg'))
        xb = xb.repeat(1, 1, 1, 1)
        _bs, _, _w, _h = xb.shape
        xb = xb.to(device)

        xb_obfs = obfuscator_medianblur(xb)
        password_img = torch.randint(0, 2, (_bs, 1, _w, _h)).mul(2).sub(1).to(device)
        password = dwt(password_img)
        xb_out_z, xb_proc = embedder(xb, xb_obfs, password)
        xb_proc_clamp = torch.clamp(xb_proc, -1, 1)

        save_image(normalize(xb), f"testimg/orig.jpg", nrow=4)
        save_image(normalize(xb_obfs), f"testimg/xb_obfs.jpg", nrow=4)
        save_image(normalize(xb_proc_clamp), f"testimg/proc.jpg", nrow=4)


if __name__ == '__main__':
    test()
