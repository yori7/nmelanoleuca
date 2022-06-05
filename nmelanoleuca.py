#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('files', nargs='+')
    parser.add_argument('--out', '-o', default='pxcount.csv')
    parser.add_argument('--outDir', '-O', default='extImg')
    parser.add_argument('--level', '-l', default='50')
    parser.add_argument('--K', default='2')

    args = parser.parse_args()
    
    return args

def flatFilter(img, level=50):
    bgr = cv2.split(img)
    res = []
    
    for c in bgr:
        dst = cv2.blur(c, (50, 50)) 

        avg_hist = c.mean()
        ffc = (c/dst)*avg_hist
        res.append(ffc)

    resimg = cv2.merge(res)
    
    return resimg

def countfg(img):
    size = img.size
    bgpx = cv2.countNonZero(img)
    fgpx = size - bgpx
    fgratio = fgpx / size

    return fgpx, fgratio

def saveImg(img, fname=None, outDir='.'):
    if fname:
        outDir = os.path.abspath(outDir)
        os.makedirs(outDir, exist_ok=True)
        fname = outDir + '/' + fname
        cv2.imwrite(fname, img)

def detectBG(img, K=5):
    colors = img.reshape(-1, 3).astype(np.float32)
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0 
    compactness, labels, centers = cv2.kmeans(
            colors, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
    uniqs, counts = np.unique(labels, return_counts=True)
    mode = uniqs[counts == np.amax(counts)].max()
    labels = np.where(labels == mode, 255, 0).astype(np.uint8)
    newImg = labels.reshape(img.shape[0:2])

    return newImg

def wrightCSV(pxdict, fname='pxcount.csv'):
    with open(fname, mode='w') as f:
        for k, v in pxdict.items():
            line = k + ',' + v[0] + ',' + v[1] + '\n'
            f.write(line)

if __name__ == '__main__':
    args = parseargs()
    files = args.files
    level = int(args.level)
    K = int(args.K)
    outDir = os.path.abspath(args.outDir)
    out = os.path.abspath(args.out)

    pxdict = {'File Name' : ('Pixel Count', 'Pixel Ratio')}

    for fname in files:
        fname = os.path.abspath(fname)
        basename = os.path.basename(fname)

        img = cv2.imread(fname, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img = flatFilter(img, level)
        saveImg(img, basename, 'flatImg')
        BWimg = detectBG(img, K)
        saveImg(BWimg, basename, outDir)

        fgpx, fgratio = countfg(BWimg)
        pxdict[fname] = (str(fgpx), str(fgratio))

    wrightCSV(pxdict, fname=args.out)


