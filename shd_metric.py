import numpy as np
from medpy.metric.binary import hd


def _extract_implant_surface(data):
    data = np.asarray(data)
    depth, height, width = data.shape
    out = np.zeros_like(data)
    t1 = np.transpose(data, (1, 0, 2))
    for x in range(depth):
        for y in range(height):
            sk = np.where(data[x, y, :] == 1)[0]
            im = np.where(data[x, y, :] == 2)[0]
            if len(im):
                if len(sk):
                    if im[0] < sk[0]:
                        out[x,y,im[0]] = 1
                    if im[-1] > sk[-1]:
                        out[x,y,im[-1]] = 1
                else:
                    out[x,y,im[0]] = 1
                    out[x,y,im[-1]] = 1
            
            sk = np.where(data[x, :, y] == 1)[0]
            im = np.where(data[x, :, y] == 2)[0]
            if len(im):
                if len(sk):
                    if im[0] < sk[0]:
                        out[x,im[0],y] = 1
                    if im[-1] > sk[-1]:
                        out[x,im[-1],y] = 1
                else:
                    out[x,im[0],y] = 1
                    out[x,im[-1],y] = 1
            sk = np.where(t1[x, y, :] == 1)[0]
            im = np.where(t1[x, y, :] == 2)[0]
            if len(im):
                if len(sk):
                    if im[0] < sk[0]:
                        out[y,x,im[0]] = 1
                    if im[-1] > sk[-1]:
                        out[y,x,im[-1]] = 1
                else:
                    out[y,x,im[-1]] = 1
                    out[y,x,im[0]] = 1
            sk = np.where(t1[x, :, y] == 1)[0]
            im = np.where(t1[x, :, y] == 2)[0]
            if len(im):
                if len(sk):
                    if im[-1] > sk[-1]:
                        out[im[-1],x,y] = 1
                else:
                    out[im[-1],x,y] = 1
    return out


def shd(pred, gt, voxelspacing=None):
    surface_pred = _extract_implant_surface(pred.astype(np.uint8))
    surface_gt = _extract_implant_surface(gt.astype(np.uint8))

    return hd(surface_pred, surface_gt, voxelspacing=voxelspacing)