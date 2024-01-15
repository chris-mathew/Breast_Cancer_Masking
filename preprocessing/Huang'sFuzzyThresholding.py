'''
Part 2 of pre-processing
A. R. Beeravolu, S. Azam, M. Jonkman, B. Shanmugam, K. Kannoorpatti and A. Anwar, "Preprocessing of Breast Cancer Images to Create Datasets for Deep-CNN," in IEEE Access, vol. 9, pp. 33438-33463, 2021, doi: 10.1109/ACCESS.2021.3058773.
'''
import numpy as np
import math

def huang(data):
    # print(data.shape)
    threshold=-1;
    first_bin= 0
    for ih in range(254):
        if np.any(data[ih]) != 0:
            first_bin = ih
            break
    last_bin=254;
    for ih in range(254,-1,-1):
        if np.any(data[ih]) != 0:
            last_bin = ih
            break
    term = 1.0 / (last_bin - first_bin)
    mu_0 = np.empty(shape=(254,))
    num_pix = 0.0
    sum_pix = 0.0
    # print(type(sum_pix), type(num_pix))
    for ih in range(first_bin,254):
        # print(type(ih * data[ih]))
        # print(len(data[ih]))
        sum_pix = sum(sum_pix + (ih * data[ih]))
        num_pix = sum(num_pix + data[ih])
        # print(type(sum_pix), type(num_pix))
        mu_0[ih] = sum_pix / num_pix
    min_ent = float("inf")
    for it in range(254): 
        ent = 0.0
        for ih in range(it):
            mu_x = 1.0 / ( 1.0 + term * math.fabs( ih - mu_0[it]))
            if ( not ((mu_x  < 1e-06 ) or (mu_x > 0.999999))):
                ent = ent + data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x) ) 
        if (np.any(ent) < min_ent):
            min_ent = ent
            threshold = it
    return threshold
