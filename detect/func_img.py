import os
import csv
from preprocess_img import savenpy
from imagetest import testmain
from coordinate import csv_generate, loadcsvdata
from plot import plot
from immune import immune


def detect(name, file):
    if os.path.exists('/home/yuanshuai20/giat/wld/1115/' + name+'.png'):
        savenpy(os.path.join('/home/yuanshuai20/giat/wld/1115', name+'.png'), name,
                '/home/yuanshuai20/giat/wld/savenpy/1', file)
        testmain('/home/yuanshuai20/giat/wld/300.ckpt', name)
        csv_generate(name)
        flag, coordX, coordY, coordZ, probility, dia_mm=loadcsvdata('/home/yuanshuai20/giat/wld/csvfile/1/'+name+'.csv')
        if flag:
            plot('/home/yuanshuai20/giat/wld/1115/'+name+'.png', coordX, coordY, dia_mm,'/home/yuanshuai20/giat/wld/plotimage/'+name)
            immu = immune('/home/yuanshuai20/giat/wld/1115/' + name+'.png')
            return 1, immu
        else:
            return 0, 0

    elif os.path.exists('/home/yuanshuai20/giat/wld/1115/' + name+'.jpg'):
        savenpy(os.path.join('/home/yuanshuai20/giat/wld/1115', name+'.jpg'), name,
                '/home/yuanshuai20/giat/wld/savenpy/1', file)
        testmain('/home/yuanshuai20/giat/wld/300.ckpt', name)
        csv_generate(name)
        flag, coordX, coordY, coordZ, probility, dia_mm=loadcsvdata('/home/yuanshuai20/giat/wld/csvfile/1/'+name+'.csv')
        if flag:
            plot('/home/yuanshuai20/giat/wld/1115/'+name+'.png', coordX, coordY, dia_mm,'/home/yuanshuai20/giat/wld/plotimage/'+name)
            immu = immune('/home/yuanshuai20/giat/wld/1115/' + name+'.jpg')
            return 1, immu
        else:
            return 0, 0

    elif os.path.exists('/home/yuanshuai20/giat/wld/1115/' + name+'.jepg'):
        savenpy(os.path.join('/home/yuanshuai20/giat/wld/1115', name+'.jepg'), name,
                '/home/yuanshuai20/giat/wld/savenpy/1', file)
        testmain('/home/yuanshuai20/giat/wld/300.ckpt', name)
        csv_generate(name)
        flag, coordX, coordY, coordZ, probility, dia_mm=loadcsvdata('/home/yuanshuai20/giat/wld/csvfile/1/'+name+'.csv')
        if flag:
            plot('/home/yuanshuai20/giat/wld/1115/'+name+'.png', coordX, coordY, dia_mm,'/home/yuanshuai20/giat/wld/plotimage/'+name)
            immu = immune('/home/yuanshuai20/giat/wld/1115/' + name+'.jepg')
            return 1, immu
        else:
            return 0, 0

    else:
        return -1, -1


