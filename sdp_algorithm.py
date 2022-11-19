from starter import *
import random
import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm


def sdp_max_k_cut_solve(k):

    def sdp_helper(G: nx.Graph):
        
        # Step 1: Solve SDP_k to obtain vectors v_1, v_2, ..., v_n in S_n-1
            # maximize: (k-1)/k sum_{i<j}w_{i,j}(1-v_i*v_j)
            # subject to v_j in R^n
            # v_i*v_i = 1for all i
            # v_i*v_j >= -1/(k-1) for all i != j

            # 1 objective function and 2 constraints to implement, then black-box SDP using cvxpy

        n = G.number_of_nodes()


        #v = [cp.Variable(n) for _ in G.nodes]

        Y = cp.Variable((n, n), PSD=True)

        constraints = [Y[i][i] == 1 for i in range(n)]

        constraints += [Y[i][j] >= -1 / (k - 1) for i in range(n) for j in range(n) if i != j]

        prob = cp.Problem(cp.Maximize(sum(d * (1 - Y[i][j]) for i, j, d in G.edges(data='weight') if i < j)), constraints)

        #print("starting to solve the problem")
        prob.solve()
        #print("finished the problem")

        #print(prob.value)

        #print(Y.value)
        Y_solved = np.array(Y.value)
        L = np.linalg.cholesky(Y_solved)

        print(L[89])

        #print(np.dot(L, L.T))



        # Step 2: Choose k random vectors z_1, z_2, ..., z_k
            # each entry of these vectors should be chosen from the normal distribtuion (0, 1)
            # figure out what that means and how to choose it?

        z = [np.random.normal(loc=0.0, scale=1.0, size=(n, 1)) for _ in range(k)]

        # Step 3: Partition V according to which of z_1, z_2, ..., z_k is closest to each v_j
            # find the maximum dot product, and assign it to that partition i

        teams = [1 for _ in range(k)]
        list_of_nodes = list(G.nodes)
        random.shuffle(list_of_nodes)
        for i in list_of_nodes:
            team = max(range(k), key=lambda l: np.dot(L[i], z[l]) / teams[l])
            teams[team] *= 1
            G.nodes[i]['team'] = team + 1


        # teams = [[] for _ in range(k)]
        # for i in range(k):
        #     for j in range(n):
        #         dot_product = np.dot(L[j], z[i])
        #         if not False in [dot_product >= np.dot(L[j], z[l]) for l in range(k) if l != i]:
        #             teams[i].append(j)
        #     #print(teams[i])

        # for i in range(k):
        #     for v in teams[i]:
        #         G.nodes[v]['team'] = i + 1


    return sdp_helper