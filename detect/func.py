import os
import csv
from preprocess import savenpy
from imagetest import testmain
from coordinate import csv_generate, loadcsvdata
from plot import plot
from immune import immune


def detect(filename):
    name = filename[0:10]
    name1 = filename.split('.')[0]
    if os.path.exists('/home/yuanshuai20/giat/wld/1115/' + name1+'.png'):
        savenpy(os.path.join('/home/yuanshuai20/giat/wld/1115', name1+'.png'), name,
                '/home/yuanshuai20/giat/wld/savenpy/1')
        testmain('/home/yuanshuai20/giat/wld/300.ckpt', name)
        csv_generate(name)
        flag, coordX, coordY, coordZ, probility, dia_mm=loadcsvdata('/home/yuanshuai20/giat/wld/csvfile/1/'+name+'.csv')
        if flag:
            plot('/home/yuanshuai20/giat/wld/1115/'+name1+'.png', coordX, coordY, dia_mm,'/home/yuanshuai20/giat/wld/plotimage/'+name1)
            immu = immune('/home/yuanshuai20/giat/wld/1115/' + name1+'.png')
            return 1, immu
        else:
            return 0, 0

    elif os.path.exists('/home/yuanshuai20/giat/wld/1115/' + name1+'.jpg'):
        savenpy(os.path.join('/home/yuanshuai20/giat/wld/1115', name1+'.jpg'), name,
                '/home/yuanshuai20/giat/wld/savenpy/1')
        testmain('/home/yuanshuai20/giat/wld/300.ckpt', name)
        csv_generate(name)
        flag, coordX, coordY, coordZ, probility, dia_mm=loadcsvdata('/home/yuanshuai20/giat/wld/csvfile/1/'+name+'.csv')
        if flag:
            plot('/home/yuanshuai20/giat/wld/1115/'+name1+'.png', coordX, coordY, dia_mm,'/home/yuanshuai20/giat/wld/plotimage/'+name1)
            immu = immune('/home/yuanshuai20/giat/wld/1115/' + name1+'.jpg')
            return 1, immu
        else:
            return 0, 0

    elif os.path.exists('/home/yuanshuai20/giat/wld/1115/' + name1+'.jepg'):
        savenpy(os.path.join('/home/yuanshuai20/giat/wld/1115', name1+'.jepg'), name,
                '/home/yuanshuai20/giat/wld/savenpy/1')
        testmain('/home/yuanshuai20/giat/wld/300.ckpt', name)
        csv_generate(name)
        flag, coordX, coordY, coordZ, probility, dia_mm=loadcsvdata('/home/yuanshuai20/giat/wld/csvfile/1/'+name+'.csv')
        if flag:
            plot('/home/yuanshuai20/giat/wld/1115/'+name1+'.png', coordX, coordY, dia_mm,'/home/yuanshuai20/giat/wld/plotimage/'+name1)
            immu = immune('/home/yuanshuai20/giat/wld/1115/' + name1+'.jepg')
            return 1, immu
        else:
            return 0, 0

    else:
        return -1, -1


