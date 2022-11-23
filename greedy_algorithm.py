from starter import *
import random

def greedy_algorithm(G: nx.Graph):
    V = []
    list_of_vertices = list(G.nodes)
    list_of_groupings = random_groupings(list_of_vertices)

    # Randomly chunk the vertices into groups, R, then run the algorithm (this might be really slow?)

    for R in tqdm(list_of_groupings):
        best_team_for_R = minimize_score(G, V, R)
        for v in R:
            V.append(v)
        for i in range(len(R)):
            G.nodes[R[i]]['team'] = best_team_for_R[i]

def random_groupings(list_in):
    """
    Inputs:
    list_in is the list of all verticies.
    
    Outputs:
    Return groupings, which is a shuffled version of list_in partition into groups of random size.
    """
    random.shuffle(list_in)
    n = len(list_in)
    groupings = []
    i = 0
    while list_in:
        next_list_size = random.randint(2, 15)
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

    percent_done = len(G.nodes) / max(len(V), 1)

    for v in G.nodes:
        if v not in V:
            G.nodes[v]['team'] = -1

    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
            
    teams, counts = np.unique([i for i in output if i != -1], return_counts=True)

    k = np.max(teams)
    b = np.linalg.norm((counts / len(V)) - 1 / k, 2)
    C_w = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v] and output[u] != -1)

    return C_w + (K_COEFFICIENT * math.exp(K_EXP * k)) + math.exp(B_EXP * b)

def set_team_partial_score(G: nx.Graph, V, k):
    """
    Inputs:
    G is a copy of the graph currently being approximated.
    V is the set of vertices that have already been set.
    k is the fixed number of teams that will be used to calculate the score.

    Outputs:
    Return the partial score of the elements that have been set, considering the weights and distribution of sizes
    """

    percent_done = len(G.nodes) / max(len(V), 1)

    for v in G.nodes:
        if v not in V:
            G.nodes[v]['team'] = -1

    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
            
    teams, counts = np.unique([i for i in output if i != -1], return_counts=True)

    b = np.linalg.norm((counts / len(V)) - 1 / k, 2)
    C_w = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v] and output[u] != -1)

    return C_w + math.exp(B_EXP * b)



def minimize_score(G: nx.Graph, V, R):
    """
    Inputs:
    G is a copy of the graph currently being approximated.
    V is the set of vertices that have already been set, not including R which is about to be set.
    R is the set of vertices that are about to be added to the set
    
    Outputs:
    Return the optimal partition to place each v from R into.
    """
    minimum_score = float('inf')
    best_team = -1

    teams = np.unique([G.nodes[v]['team'] for v in V if v not in R])
    if teams.any():
        k = max(2, np.max(teams))
    else:
        k = 2

    # test if k is updating correctly ... (this should be speeding it up maybe, because less stuff needs to be searched as k increases)

    for max_team in range(k, min(int(k * 2.5), 16)):
        current_team = []
        G_copy = G.copy()
        V_copy = V.copy()
        for v in R:
            lowest_score = float('inf')
            v_best_team = -1
            for team in range(max_team):
                G_copy_copy = G_copy.copy()
                V_copy_copy = V_copy.copy()
                G_copy_copy.nodes[v]['team'] = team + 1
                V_copy_copy.append(v)
                current_score = set_team_partial_score(G_copy_copy, V_copy_copy, max_team)
                #print("v:", v,"team:", team + 1, "current_score:", current_score)
                if current_score < lowest_score:
                    lowest_score = current_score
                    v_best_team = team + 1
                #print(v_best_team)
            v_team = v_best_team
            current_team.append(v_team)
            G_copy.nodes[v]['team'] = v_team
            V_copy.append(v)
        current_score = partial_score(G_copy, V_copy)
        if current_score < minimum_score:
            minimum_score = current_score
            best_team = current_team
    return best_team



