
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def normalize_clip_scores(scores, ver=1):
    assert ver in [1, 2]
    if ver == 1:
        return [item / np.max(item, axis=0) for item in scores]
    else:
        return [(item - np.min(item, axis=0)) / (np.max(item, axis=0) - np.min(item, axis=0)) for item in scores]


def normalize_one_clip_scores(scores, ver=1):
    assert ver in [1, 2]
    if ver == 1:
        return scores / np.max(scores, axis=0)
    else:
        return (scores - np.min(scores, axis=0)) / (np.max(scores, axis=0) - np.min(scores, axis=0))


def normalize(sequence_n_frame, scores_appe, scores_flow, scores_comb, scores_angle, ver=2, clip_normalize=True):
    if sequence_n_frame is not None:
        if len(sequence_n_frame) > 1:
            accumulated_n_frame = np.cumsum(sequence_n_frame - 1)[:-1]

            scores_appe = np.split(scores_appe, accumulated_n_frame, axis=0)
            scores_flow = np.split(scores_flow, accumulated_n_frame, axis=0)
            scores_comb = np.split(scores_comb, accumulated_n_frame, axis=0)
            scores_angle = np.split(scores_angle, accumulated_n_frame, axis=0)

            if clip_normalize:
                np.seterr(divide='ignore', invalid='ignore')
                scores_appe = normalize_clip_scores(scores_appe, ver=ver)
                scores_flow = normalize_clip_scores(scores_flow, ver=ver)
                scores_comb = normalize_clip_scores(scores_comb, ver=ver)
                scores_angle = normalize_clip_scores(scores_angle, ver=ver)

            scores_appe = np.concatenate(scores_appe, axis=0)
            scores_flow = np.concatenate(scores_flow, axis=0)
            scores_comb = np.concatenate(scores_comb, axis=0)
            scores_angle = np.concatenate(scores_angle, axis=0)

        else:
            if clip_normalize:
                np.seterr(divide='ignore', invalid='ignore')

                scores_appe = np.array(normalize_one_clip_scores(scores_appe, ver=ver))
                scores_flow = np.array(normalize_one_clip_scores(scores_flow, ver=ver))
                scores_comb = np.array(normalize_one_clip_scores(scores_comb, ver=ver))
                scores_angle = np.array(normalize_one_clip_scores(scores_angle, ver=1))

    return scores_appe, scores_flow, scores_angle, scores_comb


def find_max_patch(diff_map_appe, patches=3, size=16, step=4, is_multi=False):
    assert size % step == 0
    # diff_map_appe size: batch * channel * height * width
    b_size = diff_map_appe.shape[0]
    max_mean = np.zeros([b_size, patches])
    std = np.zeros([b_size, patches])
    pos = np.zeros([b_size, patches, 2])

    # sliding window
    for i in range(0, diff_map_appe.shape[-2] - size, step):
        for j in range(0, diff_map_appe.shape[-1] - size, step):
            # mean and std based on patch
            curr_std = np.std(diff_map_appe[..., i:i + size, j:j + size], axis=(1, 2, 3))
            curr_mean = np.mean(diff_map_appe[..., i:i + size, j:j + size], axis=(1, 2, 3))
            for b in range(b_size):
                for n in range(patches):
                    if curr_mean[b] > max_mean[b, n]:
                        max_mean[b, n + 1:] = max_mean[b, n:-1]
                        std[b, n + 1:] = std[b, n:-1]
                        pos[b, n + 1:] = pos[b, n:-1]
                        max_mean[b, n] = curr_mean[b]
                        std[b, n] = curr_std[b]
                        pos[b, n] = [i, j]
                        break

    if is_multi:
        patches_mean = np.sum(max_mean)
        patches_std = np.sum(std)
        return patches_mean, patches_std
    else:
        return max_mean[:, 0], std[:, 0]


