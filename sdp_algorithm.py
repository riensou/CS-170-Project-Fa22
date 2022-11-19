from starter import *
import random
import cvxpy as cp
import numpy as np

def sdp_solve(k):
    def sdp_solve_helper(G: nx.Graph):
        sdp_max_k_cut_solve(k)(G)
    return sdp_solve_helper

def sdp_max_k_cut_solve(k):

    def sdp_helper(G: nx.Graph):

        n = G.number_of_nodes()

        Y = cp.Variable((n, n), PSD=True)

        constraints = [Y[i][i] == 1 for i in range(n)]

        constraints += [Y[i][j] >= -1 / (k - 1) for i in range(n) for j in range(n) if i != j]

        prob = cp.Problem(cp.Maximize(sum(d * (1 - Y[i][j]) for i, j, d in G.edges(data='weight') if i < j)), constraints)

        prob.solve()

        Y_solved = np.array(Y.value)
        L = np.linalg.cholesky(Y_solved)

        z = [np.random.normal(loc=0.0, scale=1.0, size=(n, 1)) for _ in range(k)]

        teams = [1 for _ in range(k)]
        list_of_nodes = list(G.nodes)
        random.shuffle(list_of_nodes)
        for i in list_of_nodes:
            team = max(range(k), key=lambda l: np.dot(L[i], z[l]) / teams[l])
            teams[team] *= 1.2
            G.nodes[i]['team'] = team + 1

    return sdp_helper