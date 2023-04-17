import sys
sys.path.append('facerecognize/AdaFace')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
sys.path.append('faceshifter')
import torch
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from proface_wrapper import get_obs_face
from facerecognize.AdaFace.face_alignment import mtcnn
from PIL import Image
mtcnn_model = mtcnn.MTCNN('cuda:0' if torch.cuda.is_available() else 'cpu', crop_size=(112, 112))
from faceshifter.face_modules import mtcnn as mtcnnfaceshifter
mtcnn_model_faceshifter = mtcnnfaceshifter.MTCNN()
from torchvision import transforms
def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None

    return face

def get_aligned_face_bboxes(img):
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
    except Exception as e:
        bboxes, faces = None, None

    return bboxes, faces

def get_aligned_face_trans(img):
    try:
        face, trans_inv, box = mtcnn_model_faceshifter.align(img, return_trans_inv=True, return_boxes=True)
    except Exception as e:
        face, trans_inv, box = None, None, None

    return face, trans_inv, box

if __name__ == '__main__':
    # face = get_aligned_face('/home/lyyang/papercode/PRO_Face/testimg/Leslie.jpg')
    # face.show()
    # face.save('/home/lyyang/papercode/PRO_Face/testimg/Leslie_align.jpg')

    import numpy as np
    import cv2
    img = Image.open('/home/lyyang/papercode/PRO_Face/testimg/Leslie.jpg')
    img_array = np.array(img)
    face, trans_inv, box = mtcnn_model_faceshifter.align(img, return_trans_inv=True, return_boxes=True)
    face_obs = get_obs_face(face, 'pixelate')
    def tensor2numpy(tensor):
        tensor = tensor / 2.0 + 0.5
        tensor = tensor.squeeze(0)
        img = transforms.ToPILImage()(tensor)
        return np.array(img)
    face_obs = tensor2numpy(face_obs)
    face_obs_img_align = Image.fromarray(face_obs)
    # face_obs_img_align.show()
    # face_obs_border_black_array = cv2.warpAffine(face_obs, trans_inv, (np.size(img_array, 1), np.size(img_array, 0)))
    # face_obs_border_black_array = face_obs_border_black_array.astype(np.uint8)
    # face_obs_border_black_img = Image.fromarray(face_obs_border_black_array)
    # face_obs_border_black_img.show()
    # # face_obs_border_black_img.save('/home/lyyang/papercode/PRO_Face/testimg/Leslie_trans.jpg')
    # # result = np.zeros([np.size(img_array, 0), np.size(img_array, 1), 3], dtype=np.uint8)
    # # for i in range(np.size(img_array, 0)):
    # #     for j in range(np.size(img_array, 1)):
    # #         result[i][j] = img_array[i][j] if np.all(face_obs_border_black_array[i][j] == 0) else face_obs_border_black_array[i][j]
    # # result = Image.fromarray(result)
    # x,y,w,h = box[0].astype(np.int32),box[1].astype(np.int32),box[2].astype(np.int32),box[3].astype(np.int32)
    # img_array[y:h, x:w] = face_obs_border_black_array[y:h, x:w]
    # result = Image.fromarray(img_array)
    # result.show()
    # result.save('/home/lyyang/papercode/PRO_Face/testimg/Leslie_result1.jpg')
    img = Image.fromarray(img_array)
    img.show()
    cv2.warpAffine(face_obs, trans_inv, (np.size(img_array, 1), np.size(img_array, 0)), dst=img_array)
    img = Image.fromarray(img_array)
    img.show()
    # face_obs_img_align.show()




