from starter import *
import random
import cvxpy as cp
import numpy as np


def sdp_max_k_cut_solve(k):

    def sdp_helper(G: nx.Graph):
        # S_n-1 is the unit sphere in n-1 dimensions


        
        # Step 1: Solve SDP_k to obtain vectors v_1, v_2, ..., v_n in S_n-1
            # maximize: (k-1)/k sum_{i<j}w_{i,j}(1-v_i*v_j)
            # subject to v_j in S_n-1 for all j
            # v_i*v_j >= -1/(k-1) for all i != j

            # 1 objective function and 2 constraints to implement, then black-box SDP using cvxpy

        # Step 2: Choose k random vectors z_1, z_2, ..., z_k

        # Step 3: Partition V according to which of z_1, z_2, ..., z_k is closest to each v_j

        x = "hello"

    return sdp_helper