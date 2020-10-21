
import numpy as np


def is_grooves(gradients):
    # is_planes = np.logical_and(np.radians(
    #    30) < gradients, gradients < np.radians(150))
    # return np.logical_not(is_planes)
    # return np.logical_and(np.radians(60) < gradients , gradients < np.radians(120))
    ret = np.logical_and(np.radians(60) < gradients,
                         gradients < np.radians(120))

    print("===============================")
    print(np.count_nonzero(ret))
    print(np.count_nonzero(np.logical_not(ret)))
    print("===============================")
    return ret


def estimate_groove_index(dset):
    gradient_averages = dset[:, -1]
    _is_grooves = is_grooves(gradient_averages)
    groove_candidates = np.where(_is_grooves)
    return groove_candidates[0]


if __name__ == '__main__':

    ary = np.array([np.radians(0), np.radians(30), np.radians(31),
                    np.radians(149), np.radians(150), np.radians(180)])

    candinates = is_grooves(ary)
    print(candinates)

    ary = ary.reshape(1, 6)
    print(ary)
    idx = estimate_groove_index(ary)
    print(idx)
