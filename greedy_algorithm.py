from starter import *
import random

def greedy_algorithm(G: nx.Graph):
    V = []
    list_of_vertices = list(G.nodes)
    list_of_groupings = random_groupings(list_of_vertices)

    # Randomly chunk the vertices into groups, R, then run the algorithm (this might be really slow?)

    for R in list_of_groupings:
        best_team_for_R = minimize_score(G, V, R)
        for v in R:
            V.append(v)
        for i in range(len(R)):
            G.nodes[R[i]]['team'] = best_team_for_R[i]

def random_groupings(list_in):
    random.shuffle(list_in)
    n = len(list_in)
    groupings = []
    i = 0
    while list_in:
        next_list_size = random.randint(1, 3)
        end_point = min(i + next_list_size, n)
        groupings.append(list_in[i:end_point])
        list_in = list_in[end_point:]
    return groupings

def partial_score(G: nx.Graph, V):
    """
    Inputs:
    G is a copy of the graph currently being approximated.
    V is the set of vertices that have already been set.

    Outputs:
    Return the partial score of the elements that have been set.
    """
    for v in G.nodes:
        if v not in V:
            G.nodes[v]['team'] = -1

    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
            
    teams, counts = np.unique([i for i in output if i != -1], return_counts=True)

    k = np.max(teams)
    b = np.linalg.norm((counts / len(V)) - 1 / k, 2)
    C_w = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v] and output[u] != -1)

    return C_w + K_COEFFICIENT * math.exp(K_EXP * k) + math.exp(B_EXP * b)

def minimize_score(G: nx.Graph, V, R):
    """
    Inputs:
    G is a copy of the graph currently being approximated.
    V is the set of vertices that have already been set, not including R which is about to be set.
    R is the set of vertices that are about to be added to the set
    
    Outputs:
    Return the optimal partition to place v into.
    """
    minimum_score = float('inf')
    best_team = -1

    teams = np.unique([G.nodes[v]['team'] for v in V if v not in R])
    if teams.any():
        k = np.max(teams)
    else:
        k = 2

    for max_team in range(k, 15):
        current_team = []
        G_copy = G.copy()
        for v in R:
            G_copy_copy = G_copy.copy()
            lowest_score = float('inf')
            v_best_team = -1
            for team in range(max_team):
                G_copy_copy.nodes[v]['team'] = team + 1
                current_score = partial_score(G_copy_copy, V + [v])
                if current_score < lowest_score:
                    lowest_score = current_score
                    v_best_team = team
            #v_team = random.randint(1, max_team)
            v_team = v_best_team
            current_team.append(v_team)
            G_copy.nodes[v]['team'] = v_team
        current_score = partial_score(G_copy, V + R)
        if current_score < minimum_score:
            minimum_score = current_score
            best_team = current_team
    return best_team



