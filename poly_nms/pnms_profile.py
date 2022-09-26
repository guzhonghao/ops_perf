import numpy as np
import torch
import poly_nms_cuda
import sys

def poly_nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    # import pdb
    # print('in nms wrapper')
    # pdb.set_trace()
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = poly_nms_cuda.poly_nms(dets_th, iou_thr)
        else:
            raise NotImplementedError

    if is_numpy:
        raise NotImplementedError
    return dets[inds, :], inds

def pnms_test(repeat):
    nms_thresh=0.5
    dets = np.asarray(np.random.randn(2000,9), dtype=np.float32)
    print("input shape:")
    print(torch.from_numpy(dets).shape)
    for i in range(repeat):
        boxes_gpu=torch.from_numpy(dets).cuda()
        keep = poly_nms(boxes_gpu, nms_thresh)
        print("output:")
        print(keep[1])


if __name__ == '__main__':
    pnms_test(int(sys.argv[1]))
