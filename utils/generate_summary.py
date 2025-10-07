# -*- coding: utf-8 -*-
import torch
#from knapsack import knapsack_ortools
from utils.knapsack_implementation import knapSack
import math
# from knapsack_implementation import knapSack
import pdb

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """

    n_segs = len(cps)
    n_frames = n_frames[0]
    n_frames_tensor = n_frames.unsqueeze(0)
    
    frame_scores = torch.zeros((n_frames), dtype=torch.float32)
    if positions.dtype != torch.int32:
        positions = positions.to(torch.int32)
    if positions[-1] != n_frames:
        positions = torch.cat([positions, n_frames_tensor])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = cps[seg_idx][0].to(torch.int), (cps[seg_idx][1]+1).to(torch.int)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = torch.floor(n_frames * proportion).to(torch.int)

    # print("limits", limits)
    # print("nfps", nfps)
    # print("seg_score", seg_score)
    # print("len(nfps)", len(nfps))
    picks = knapSack(limits, nfps, seg_score, len(nfps))

    summary = torch.zeros((1), dtype=torch.float32, device='cuda') # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = torch.ones((nf), dtype=torch.float32, device='cuda')
        else:
            tmp = torch.zeros((nf), dtype=torch.float32, device='cuda')
        summary = torch.cat((summary, tmp))

    summary = summary[1:] # delete the first element
    return summary