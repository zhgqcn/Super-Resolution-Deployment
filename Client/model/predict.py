from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda
from model import LapSrnMS as LapSRN
import numpy as np
import torch.optim as optim
import torchvision
import time
from prepare_images import *


transform = Compose([ToTensor(),
                     Lambda(lambda x: x.repeat(3, 1, 1)),
                     ])


def load_model(model, optimizer, pre_model):
    checkpoint = torch.load(pre_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = checkpoint['epoch']
    return model, optimizer, epochs


def pre_deal(LR_r, LR_g):
    LR_r = transform(LR_r)
    LR_r = LR_r.unsqueeze(0)
    LR_g = transform(LR_g)
    LR_g = LR_g.unsqueeze(0)
    LR_r = LR_r.cuda()
    LR_g = LR_g.cuda()
    return LR_r, LR_g


def testing(model, input):
    HR_2, HR_4 = model(input)
    HR_2 = HR_2.cpu()
    HR_4 = HR_4.cpu()
    return HR_2, HR_4


def post_deal(HR_2, HR_4):
    out_HR_2 = HR_2.data[0].numpy()
    out_HR_2 *= 255.0
    out_HR_2 = out_HR_2.clip(0, 255)
    out_HR_2 = Image.fromarray(np.uint8(out_HR_2[0]), mode='L')

    out_HR_4 = HR_4.data[0].numpy()
    out_HR_4 *= 255.0
    out_HR_4 = out_HR_4.clip(0, 255)
    out_HR_4 = Image.fromarray(np.uint8(out_HR_4[0]), mode='L')
    return out_HR_2, out_HR_4


def predict(model_r, model_g, input_img):
    localtime = time.asctime(time.localtime(time.time()))
    print("Start :", localtime)

    with torch.no_grad():


        file = input_img
        img = Image.open('static/input/{}'.format(file)).convert('RGB')
        LR_r, LR_g, LR_b = img.split()

        LR_r, LR_g = pre_deal(LR_r, LR_g)

        HR_2_r, HR_4_r = testing(model_r, LR_r)
        localtime = time.asctime(time.localtime(time.time()))
        print("R channel have tested and saved , time : ", localtime)

        HR_2_g, HR_4_g = testing(model_g, LR_g)
        localtime = time.asctime(time.localtime(time.time()))
        print("G channel have tested and saved , time : ", localtime)

        out_HR_2_r, out_HR_4_r = post_deal(HR_2_r, HR_4_r)
        out_HR_2_g, out_HR_4_g = post_deal(HR_2_g, HR_4_g)

        # out_HR_2_b = Image.open('static/all_black_1024.tif')
        # out_HR_4_b = Image.open('static/all_black_2048.tif')
        width  = img.size[0]
        height = img.size[1]
        out_HR_2_b = Image.new("L", (width * 2, height * 2))
        out_HR_4_b = Image.new("L", (width * 4, height * 4))

        out_HR_2 = Image.merge('RGB', [out_HR_2_r, out_HR_2_g, out_HR_2_b]).convert('RGB')
        out_HR_4 = Image.merge('RGB', [out_HR_4_r, out_HR_4_g, out_HR_4_b]).convert('RGB')

        # out_HR_2_r.save('static/output/X2/' + input_img)
        # print('output image saved to ', ' /static/output/X2/' + input_img)

        # out_HR_4_r.save('static/output/X4/' + input_img)
        # print('output image saved to ', ' /static/output/X4/' + input_img)

        # localtime = time.asctime(time.localtime(time.time()))
        # print("Finished :", localtime)

        return out_HR_2, out_HR_4



