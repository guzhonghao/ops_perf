import paddle.fluid as fluid
import numpy as np
import paddle
import random
import sys

# import torch
# paddle.enable_static()
print(paddle.__version__)
# print(torch.__version__)

def test_generate_proposals_v2(repeat):
        paddle.disable_static()

        # 1
        N = 1
        A = 15
        H = 54
        W = 40

        scores = paddle.rand((N, A, H, W), dtype=paddle.float32)
        # bbox_deltas = paddle.rand((N, 4*A, H, W), dtype=paddle.float32)

        dn = 0
        mid = 5
        up = 20

        bbox_deltas_data = np.random.rand(N, 4 * A, H, W)
        # for n in range(N):
        #         for a in range(A):
        #                 for h in range(H):
        #                         for w in range(W):
        #                                 bbox_deltas_data[n][0 + 4 * a][h][w] = random.uniform(dn, mid)
        #                                 bbox_deltas_data[n][1 + 4 * a][h][w] = random.uniform(dn, mid)
        #                                 bbox_deltas_data[n][2 + 4 * a][h][w] = random.uniform(mid, up)
        #                                 bbox_deltas_data[n][3 + 4 * a][h][w] = random.uniform(mid, up) 
        bbox_deltas = paddle.to_tensor(bbox_deltas_data, dtype=paddle.float32)

        # img_size = paddle.to_tensor([[0.5, 0.5]])
        img_size_data = np.random.rand(N, 2) * (up - dn)
        img_size = paddle.to_tensor(img_size_data, dtype=paddle.float32)

        # anchors = paddle.rand((H, W, A, 4), dtype=paddle.float32)
        # variances = paddle.rand((H, W, A, 4), dtype=paddle.float32)
        variances_data = np.random.rand(H, W, A, 4) * 5
        anchors_data = np.random.rand(H, W, A, 4)
        # for h in range(H):
        #         for w in range(W):
        #                 for a in range(A):
        #                         anchors_data[h][w][a][0] = random.uniform(dn, mid)
        #                         anchors_data[h][w][a][1] = random.uniform(dn, mid)
        #                         anchors_data[h][w][a][2] = random.uniform(mid, up)
        #                         anchors_data[h][w][a][3] = random.uniform(mid, up)

        anchors = paddle.to_tensor(anchors_data, dtype=paddle.float32)
        variances = paddle.to_tensor(variances_data, dtype=paddle.float32)

        pre_nms_top_n = 10
        post_nms_top_n = 200
        nms_thresh = 1
        min_size = 0.1
        eta = 1.0
        pixel_offset = True
        return_rois_num = True
        name = None

        for i in range(repeat):
                rpn_rois, rpn_roi_probs, rpn_rois_num = paddle.vision.ops.generate_proposals(
                        scores,
                        bbox_deltas,
                        img_size,
                        anchors,
                        variances,
                        pre_nms_top_n,
                        post_nms_top_n,
                        nms_thresh,
                        min_size,
                        eta,
                        pixel_offset,
                        return_rois_num)
                print(rpn_rois, rpn_roi_probs, rpn_rois_num)

test_generate_proposals_v2(int(sys.argv[1]))

# ## -------------------------------------------------------
