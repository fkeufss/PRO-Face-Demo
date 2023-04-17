import sys
import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# path of simswap and faceshifter target

simswap_tar = 'SimSwap/Leslie.jpg'
faceshifter_tar = 'faceshifter/Leslie.jpg'

import warnings
warnings.filterwarnings("ignore", category=Warning)
import logging

logging.basicConfig(level=logging.WARNING)

logging.disable(logging.WARNING)

from torchvision import transforms
import torchvision.transforms.functional as F
from tkinter import messagebox
from functools import partial
from face_align import get_aligned_face_trans
from proface_wrapper import get_obs_face
from facerecognize.face.face_recognizer import get_recognizer
recognizer = get_recognizer('AdaFaceIR100')

recognizer.to(device).eval()
video_index = 0

input_trans = transforms.Compose([
    transforms.Resize(112, interpolation=F.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
names = []


sys.path.append('SimSwap')
from SimSwap.simswap import SimSwap

simswap = SimSwap(simswap_tar)

from faceshifter.face_shifter import FaceShifter
faceshifter = FaceShifter(faceshifter_tar)


def live_camera_mode():
    global mode
    mode = normal_frame
    label2.configure(text='摄像头实时画面')

def face_recognize_mode():
    global mode
    mode = face_recognize
    label2.configure(text='人脸识别')


def face_recognize(frame):
    try:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        coords = faces_yunet[0][:-1].astype(np.int32)
        frame_face_tensor = input_trans(get_aligned_face_trans(Image.fromarray(frame))[0])
        frame_face_encoding = recognizer(recognizer.resize(frame_face_tensor.repeat(1, 1, 1, 1)).to(device))
        name = get_face_name(frame_face_encoding)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (coords[0] + 6, coords[1] - 10), font, 1.0, (0, 255, 0), 1)
        return frame
    except:
        return frame

def face_registration():
    global mode
    mode = normal_frame
    label2.configure(text='摄像头实时画面')
    frame = frame_register.copy()
    registration_window(frame)


def registration_window(frame):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    face_window = tk.Toplevel(main_window)
    label = tk.Label(face_window, width=frame_width, height=frame_height)
    # faces = face.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=4)

    face = get_aligned_face_trans(Image.fromarray(rgb_frame))[0]
    if not face:
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.pack()
        messagebox.showerror('错误', '未检测到人脸！')
        face_window.destroy()
    else:
        # face_locations = face_recognition.face_locations(rgb_frame)
        # top, right, bottom, left = face_locations[0]
        # rgb_frame = rgb_frame[top:bottom,left:right]
        # face_frame = frame[top:bottom,left:right]
        # img = Image.fromarray(rgb_frame)
        # imgtk = ImageTk.PhotoImage(image=img)
        imgtk = ImageTk.PhotoImage(image=face)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.pack()
        name_entry = tk.Entry(face_window)
        name_entry.pack()
        button_confirm = tk.Button(face_window, text='确认',
                                   command=partial(save_new_person, face, name_entry))
        button_cancel = tk.Button(face_window, text='取消', command=partial(save_cancel, face_window))
        button_confirm.pack()
        button_cancel.pack()


def save_new_person(aligned_face, name_entry):
    try:
        name = name_entry.get()
        face_path = os.path.join(face_database_path, name + '.jpg')
        aligned_face.save(face_path)
        # load_image = face_recognition.load_image_file(os.path.join(path, name + '.jpg'))  # 加载图片
        # image_face_encoding = face_recognition.face_encodings(load_image)[0]  # 获得128维特征值

        image_face_tensor = input_trans(
            get_aligned_face_trans(Image.open(face_path).convert('RGB'))[0])
        image_face_encoding = recognizer(recognizer.resize(image_face_tensor.repeat(1, 1, 1, 1)).to(device))
        # known_names.append(image_name.split(".")[0])
        # known_encodings.append(image_face_encoding)
        messagebox.showinfo('提示', '保存成功')
        known_names.append(name)
        known_encodings.append(image_face_encoding)

        global known_encodings_matrix
        # 将所有已知人脸的特征向量转换为一个矩阵
        known_encodings_matrix = ([ke[0].cpu().detach().numpy() for ke in known_encodings])
    except:
        messagebox.showerror('错误', '注册出错，请检查是否重复注册或保存路径！')


def save_cancel(face_window):
    cancel = messagebox.askyesno('确认', '是否取消保存？')
    if cancel:
        face_window.destroy()

def GaussianBlur_mode():
    global mode
    mode = get_GaussianBlur_frame
    label2.configure(text='GaussianBlur')


def MedianBlur_mode():
    global mode
    mode = get_MedianBlur_frame
    label2.configure(text='MedianBlur')


def Pixelate_mode():
    global mode
    mode = get_pixelate_frame
    label2.configure(text='Pixelate')


def FaceShifter_mode():
    global mode
    mode = get_faceshifter_frame
    label2.configure(text='FaceShifter')


def SimSwap_mode():
    global mode
    mode = get_simswap_frame
    label2.configure(text='SimSwap')

def ProFace_mode():
    global mode
    mode = get_ProFace_frame

def ProFace_GaussianBlur_mode():
    ProFace_mode()
    global anonymous_method
    anonymous_method = 'GaussianBlur'
    label2.configure(text='ProFace GaussianBlur')

def ProFace_MedianBlur_mode():
    ProFace_mode()
    global anonymous_method
    anonymous_method = 'MedianBlur'
    label2.configure(text='ProFace MedianBlur')

def ProFace_Pixelate_mode():
    ProFace_mode()
    global anonymous_method
    anonymous_method = 'Pixelate'
    label2.configure(text='ProFace Pixelate')

def ProFace_recognize_mode():
    global mode
    mode = get_ProFace_recognize_frame
    label2.configure(text='人脸识别')

def ProFace_GaussianBlur_recognize_mode():
    ProFace_recognize_mode()
    global anonymous_method
    anonymous_method = 'GaussianBlur'
    label2.configure(text='ProFace GaussianBlur 识别')

def ProFace_MedianBlur_recognize_mode():
    ProFace_recognize_mode()
    global anonymous_method
    anonymous_method = 'MedianBlur'
    label2.configure(text='ProFace MedianBlur 识别')

def ProFace_Pixelate_recognize_mode():
    ProFace_recognize_mode()
    global anonymous_method
    anonymous_method = 'Pixelate'
    label2.configure(text='ProFace Pixelate 识别')

def exittk():
    """
    退出程序
    """
    cap.release()
    sys.exit(0)


"窗口"
main_window = tk.Tk()
main_window.title('人脸匿名化演示系统')
main_window.wm_resizable(0, 0)
menubar = tk.Menu(main_window)  # 创建菜单栏

# 创建二级菜单
submenu1 = tk.Menu(menubar)
submenu1.add_command(label='实时画面', command=live_camera_mode)
submenu1.add_command(label='人脸识别', command=face_recognize_mode)
submenu1.add_command(label='人脸注册', command=face_registration)
submenu1.add_command(label='退出', command=exittk)

submenu2 = tk.Menu(menubar)
submenu2.add_command(label='GaussianBlur', command=GaussianBlur_mode)
submenu2.add_command(label='MedianBlur', command=MedianBlur_mode)
submenu2.add_command(label='Pixelate', command=Pixelate_mode)
submenu2.add_command(label='FaceShifter', command=FaceShifter_mode)
submenu2.add_command(label='SimSwap', command=SimSwap_mode)

submenu3 = tk.Menu(menubar)

# submenu3.add_command(label='GaussianBlur', command=ProFace_GaussianBlur_mode)
# submenu3.add_command(label='MedianBlur', command=ProFace_MedianBlur_mode)
# submenu3.add_command(label='Pixelate', command=ProFace_Pixelate_mode)
submenu3.add_command(label='GaussianBlur+识别', command=ProFace_GaussianBlur_recognize_mode)
submenu3.add_command(label='MedianBlur+识别', command=ProFace_MedianBlur_recognize_mode)
submenu3.add_command(label='Pixelate+识别', command=ProFace_Pixelate_recognize_mode)

# 创建一级菜单并关联二级菜单
menubar.add_cascade(label='人脸注册', menu=submenu1)
menubar.add_cascade(label='匿名化', menu=submenu2)
menubar.add_cascade(label='可识别的匿名化', menu=submenu3)
# 主窗体加载菜单栏
main_window.config(menu=menubar)

"窗口全局变量"
main_window.num = 0
# main_window.start = False
# main_window.time = time.time()

"窗口布局"
f = tk.Frame(main_window)
f.grid(row=0, column=1, sticky='nw')
frame_width = 640
frame_height = 480
# 视频窗口1
label_left = tk.Label(main_window, width=frame_width, height=frame_height)
label_left.imgtk = None
label_left.grid(row=0, column=0, sticky='nw')

# 视频窗口2
label_right = tk.Label(main_window, width=frame_width, height=frame_height)
label_right.grid(row=0, column=frame_width + 200, sticky='nw')

label1 = tk.Label(main_window, text='摄像头实时画面', font=("黑体", 14))
label1.place(x=10, y=10, anchor='nw')
label2 = tk.Label(main_window, text='摄像头实时画面', font=("黑体", 14))
label2.place(x=frame_width + 10, y=10, anchor='nw')

def normal_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return frame

def get_GaussianBlur_frame(frame):
    for face in faces_yunet:
        coords = face[:-1].astype(np.int32)
        x, y, w, h = max(coords[0], 0), max(coords[1], 0), coords[2], coords[3]
        frame[y:y + h, x:x + w] = cv.GaussianBlur(frame[y:y + h, x:x + w], (111, 111), 11, 11)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # 演示用 添加人脸识别名字
    # font = cv.FONT_HERSHEY_DUPLEX
    # # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # cv.putText(frame, 'liu', (coords[0] + 6, coords[1] - 10), font, 1.0, (0, 255, 0), 1)
    return frame


def get_MedianBlur_frame(frame):
    for face in faces_yunet:
        coords = face[:-1].astype(np.int32)
        x, y, w, h = max(coords[0], 0), max(coords[1], 0), coords[2], coords[3]
        frame[y:y + h, x:x + w] = cv.medianBlur(frame[y:y + h, x:x + w], 35)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # 演示用 添加人脸识别名字
    # font = cv.FONT_HERSHEY_DUPLEX
    # # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # cv.putText(frame, 'liu', (coords[0] + 6, coords[1] - 10), font, 1.0, (0, 255, 0), 1)
    return frame

def get_pixelate_frame(frame):
    for face in faces_yunet:
        coords = face[:-1].astype(np.int32)
        x, y, w, h = max(coords[0], 0), max(coords[1], 0), coords[2], coords[3]

        # 在人脸上加马赛克
        frameBox = frame[y:y + h, x:x + w]
        frameBox = frameBox[::10, ::10]
        frameBox = np.repeat(frameBox, 10, axis=0)
        frameBox = np.repeat(frameBox, 10, axis=1)
        a, b = frame[y:y + h, x:x + w].shape[:2]
        frame[y:y + h, x:x + w] = frameBox[:a, :b]
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # 演示用 添加人脸识别名字
    # font = cv.FONT_HERSHEY_DUPLEX
    # # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # cv.putText(frame, 'liu', (coords[0] + 6, coords[1] - 10), font, 1.0, (0, 255, 0), 1)
    return frame


def get_faceshifter_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return faceshifter.face_shifter(frame)


def get_simswap_frame(frame):
    try:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return simswap.simswap(frame)
    except:
        return frame

def get_ProFace_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    face, trans_inv, box = get_aligned_face_trans(Image.fromarray(frame))
    if not face:
        return frame
    face_obs = get_obs_face(face, anonymous_method)
    face_obs = tensor2numpy(face_obs)
    face_obs_border_black_array = cv.warpAffine(face_obs, trans_inv, (np.size(frame, 1), np.size(frame, 0)))
    x, y, w, h = box[0].astype(np.int32), box[1].astype(np.int32), box[2].astype(np.int32), box[3].astype(np.int32)
    frame[y:h, x:w] = face_obs_border_black_array[y:h, x:w]

    # x, y, w, h = box[0].astype(np.int32), box[1].astype(np.int32), box[2].astype(np.int32), box[3].astype(np.int32)
    # dim = (w - x, h - y)
    # resize_obs = cv.resize(face_obs, dim)
    # frame[y:h, x:w] = resize_obs
    return frame


def get_ProFace_recognize_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    face, trans_inv, box = get_aligned_face_trans(Image.fromarray(frame))
    if not face:
        return frame

    image_face_tensor = input_trans(face)
    # 使用原始人脸识别
    obs_face_encoding = recognizer(recognizer.resize(image_face_tensor.repeat(1, 1, 1, 1)).to(device))
    face_obs = get_obs_face(face, anonymous_method)
    # 使用匿名人脸识别
    # obs_face_encoding = recognizer(face_obs)

    name = get_face_name(obs_face_encoding)
    font = cv.FONT_HERSHEY_DUPLEX
    coords = faces_yunet[0][:-1].astype(np.int32)
    cv.putText(frame, name, (coords[0] + 6, coords[1] - 10), font, 1.0, (0, 255, 0), 1)
    face_obs = tensor2numpy(face_obs)
    face_obs_border_black_array = cv.warpAffine(face_obs, trans_inv, (np.size(frame, 1), np.size(frame, 0)))
    x, y, w, h = box[0].astype(np.int32), box[1].astype(np.int32), box[2].astype(np.int32), box[3].astype(np.int32)
    frame[y:h, x:w] = face_obs_border_black_array[y:h, x:w]
    return frame

def tensor2numpy(tensor):
    tensor = tensor / 2.0 + 0.5
    tensor = tensor.squeeze(0)
    img = transforms.ToPILImage()(tensor)
    return np.array(img)

def get_face_name(face_encoding, threshold = 0.8):
    if len(known_encodings) < 1:
        return 'unknown'
    sims = []
    for known_encoding in known_encodings:
        sim = face_encoding[0].dot(known_encoding[0]).item()
        sims.append(sim)
    max_sim_index = sims.index(max(sims))
    if(sims[max_sim_index] >= threshold):
        return known_names[max_sim_index]
    return 'unknown'

    # sims = np.dot(face_encoding[0], known_encodings_matrix.T)
    #
    # # 找到点积最大的已知人脸特征向量的索引
    # max_sim_index = np.argmax(sims)
    #
    # # 比较点积最大值是否大于等于阈值
    # if sims[max_sim_index] >= threshold:
    #     return known_names[max_sim_index]
    # return 'unknown'

def refresh():
    ret, frame = cap.read()
    tm.start()
    frame = cv.flip(frame, 1)
    frame_left = frame.copy()
    frame_right = frame.copy()
    global frame_register
    frame_register = frame.copy()
    frame_left = normal_frame(frame_left)

    global faces_yunet
    _, faces_yunet = yunet.detect(frame)
    if faces_yunet is not None:
        # frame_right = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB)
        frame_right = mode(frame_right)
        for face in faces_yunet:
            coords = face[:-1].astype(np.int32)
            # Draw face bounding box
            cv.rectangle(frame_left, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 2)
            cv.rectangle(frame_right, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 2)
    else:
        frame_right = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB)

    tm.stop()
    cv.putText(frame_left, 'FPS: {:.2f}'.format(tm.getFPS()), (frame_width - 70, 20), 0, 0.5, (255, 0, 0), 1)
    cv.putText(frame_right, 'FPS: {:.2f}'.format(tm.getFPS()), (frame_width - 70, 20), 0, 0.5, (255, 0, 0), 1)

    img_left = Image.fromarray(frame_left)
    img_right = Image.fromarray(frame_right)
    imgtk_left = ImageTk.PhotoImage(image=img_left)
    imgtk_right = ImageTk.PhotoImage(image=img_right)
    label_left.imgtk = imgtk_left
    label_left.configure(image=imgtk_left)
    label_right.imgtk = imgtk_right
    label_right.configure(image=imgtk_right)
    tm.reset()
    label_left.after(1, refresh)

cap = cv.VideoCapture(video_index)
mode = normal_frame
anonymous_method = ''
tm = cv.TickMeter()
yunet = cv.FaceDetectorYN.create(
    model='facedetect/yunet/face_detection_yunet_2022mar.onnx',
    config='',
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv.dnn.DNN_TARGET_CPU
)
frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
yunet.setInputSize([frame_w, frame_h])

if __name__ == '__main__':
    face_database_path = "facerecognize/face_database"
    known_names = []
    known_encodings = []
    for image_name in os.listdir(face_database_path):
        image_face_tensor = input_trans(get_aligned_face_trans(Image.open(os.path.join(face_database_path, image_name)).convert('RGB'))[0])
        image_face_encoding = recognizer(recognizer.resize(image_face_tensor.repeat(1, 1, 1, 1)).to(device))
        known_names.append(image_name.split(".")[0])
        known_encodings.append(image_face_encoding)
    faces_yunet = []
    frame_register = None

    # 将所有已知人脸的特征向量转换为一个矩阵
    known_encodings_matrix = ([ke[0].cpu().detach().numpy() for ke in known_encodings])
    refresh()
    main_window.mainloop()