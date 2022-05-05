import argparse
import os
import time
import numpy as np
import data_test
from importlib import import_module
import shutil
from utils import *
import sys

sys.path.append('../')
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
# from config_training import config as config_training

from layers import acc

def testmain(path, name):
    print('start!')

    torch.manual_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = import_module('dpn3d26')
    config, net, loss, get_pbb = model.get_model()
    start_epoch = 0
    save_dir = '/home/yuanshuai20/giat/wld/results/1'


    checkpoint = torch.load(path)#address
    if start_epoch == 0:
        start_epoch = checkpoint['epoch'] + 1
    if not save_dir:
        save_dir = checkpoint['save_dir']
    else:
        save_dir = os.path.join('results', save_dir)
    net.load_state_dict(checkpoint['state_dict'])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    logfile = os.path.join(save_dir, 'log')
    n_gpu = setgpu('all')
    # args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net).cuda()
    data_dir = '/home/yuanshuai20/giat/wld/savenpy/1/'
    # patientlist = os.listdir(data_dir)
    # patientlist.sort()
    # for patient in patientlist:
    #     if not os.path.exists(os.path.join(save_dir, patient)):
    #         os.mkdir(os.path.join(save_dir, patient))
    #     datadir = os.path.join(data_dir, patient + '/')
    margin = 16
    sidelen = 128
    # print('margin,sidelen',margin,sidelen)
    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
    dataset = data_test.DataBowl3Detector(
        data_dir,
        'testnames.npy',
        config,
        phase='test',
        split_comber=split_comber)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=data_test.collate,
        pin_memory=False)
    print('len(test_loader)', 4 * len(test_loader))  #
    iter1, iter2, iter3, iter4 = next(iter(test_loader))
    with torch.no_grad():
        test(test_loader, net, get_pbb, save_dir, config)
    return


def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    # save_dir = os.path.join(save_dir, 'TestBbox')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        # print('0',data_loader.dataset.filenames)
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = 1
        print('data.size', data.size())
        splitlist = range(0, len(data) + 1, n_per_run)
        # print('splitlist',splitlist)
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = -3
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
        # tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        # print([len(tp),len(fp),len(fn)])
        print('L,P,LEN', lbb.shape, pbb.shape)  # (0,) (289, 5) 0////(1, 4) (640, 5) 0
        print([i_name, name])
        e = time.time()
        np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)
    # np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()
    print('elapsed time is %3.2f seconds' % (end_time - start_time))
