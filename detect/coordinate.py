import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union
def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes



def csv_generate(name):
    nmsthresh = 0.1
    resolution = np.array([1, 1, 1])
    firstline = ['coordX', 'coordY', 'coordZ', 'probability', 'dia_mm']
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    csvfile1 = open('/home/yuanshuai20/giat/wld/csvfile/1/'+name+'.csv', 'w')
    writer1 = csv.writer(csvfile1)
    writer1.writerow(firstline)
    pbb = np.load('/home/yuanshuai20/giat/wld/results/1/'+name+'_pbb.npy', allow_pickle=True)
    extendbox = np.load('/home/yuanshuai20/giat/wld/savenpy/1/'+name+'_extendbox.npy', allow_pickle=True)
    spacing = np.load('/home/yuanshuai20/giat/wld/savenpy/1/'+name+'_spacing.npy', allow_pickle=True)
    origin = np.load('/home/yuanshuai20/giat/wld/savenpy/1/'+name+'_origin.npy', allow_pickle=True)
    pbbold = np.array(pbb[pbb[:, 0] > -1.5])
    pbbold = np.array(pbbold[pbbold[:, -1] > 3])  # add new 9 15
    pbbold = pbbold[np.argsort(-pbbold[:, 0])][:1000]
    pbb = nms(pbbold, nmsthresh)
    pbb_mm = pbb
    pbb = np.array(pbb[:, :-1])
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:, 0], 1).T)
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)
    pos = pbb[:, 1:]
    for nk in range(pos.shape[0]):  # pos[nk, 2], pos[nk, 1], pos[nk, 0]
        if 1 / (1 + np.exp(-pbb[nk, 0])) > 0.5:
            # print  pos[nk, 2], pos[nk, 1], pos[nk, 0], 1 / (1 + np.exp(-pbb[nk, 0])), pbb_mm[nk, -1]
            writer1.writerow([pos[nk, 2], pos[nk, 1], pos[nk, 0], 1 / (1 + np.exp(-pbb[nk, 0])), pbb_mm[nk, -1]])
    csvfile1.close()

def loadcsvdata(path):
    csv_data = open(path, 'r')
    reader = csv.reader(csv_data)
    for nodule in reader:
        if nodule[0] == 'coordX':
            continue
        coordX = float(nodule[0])
        coordY = float(nodule[1])
        coordZ = float(nodule[2])
        probility = float(nodule[3])
        dia_mm = float(nodule[4])
        return True, coordX, coordY, coordZ, probility, dia_mm
    return False, 0, 0, 0, 0, 0