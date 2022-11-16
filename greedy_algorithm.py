from starter import *

def optimize_k(G: nx.Graph):
    """Searches for the optimal k for a given graph G."""

    # TODO: Save time by finding k by with binary search, instead of incrementing it up by 1 each time

    G_copy = G.copy()

    k = 2
    max_k_cut_solve(k)(G_copy)
    validate_output(G_copy)
    minimum_score = score(G_copy)
    best_k = 2
    #print("Score:", minimum_score, "k:", k)

    increase_k = True

    while increase_k:
        G_copy = G.copy()
        k += 1
        max_k_cut_solve(k)(G_copy)
        validate_output(G_copy)
        current_score = score(G_copy)
        #print("Score:", current_score, "k:", k)
        if current_score < minimum_score:
            best_k = k
            minimum_score = current_score
        else:
            increase_k = False
    
    return best_k

# Below are 2 approached to the greedy algorithm:
# max_k_cut_solve maximizes the weights of the edges between cuts at each step
# min_k_partition_solve minimizes the weights within each partition at each step

# Based on a couple of trials, it seems as if the two approaches are identical, however, it needs more testing

def max_k_cut_solve(k):
    """Returns a function that approximates max-k cut."""
    teams = [[] for _ in range(k)]
    not_in_teams = [[] for _ in range(k)]
    def max_k_cut_helper(G: nx.Graph):
        for v in G.nodes:
            i = potential_change_cut(G, teams, not_in_teams, v)
            teams[i].append(v)
            for j in range(len(not_in_teams)):
                if j != i:
                    not_in_teams[j].append(v)
            G.nodes[v]['team'] = i + 1
    return max_k_cut_helper

def potential_change_cut(G, teams, not_in_teams, v):
    """Returns the index of the team that would maximize the cut if v is placed into that team."""
    v_into_teams = [0 for _ in range(len(teams))]
    for t in range(len(teams)):
        v_into_teams[t] += sum(d for i, j, d in G.edges(data='weight') if (i == v and j in not_in_teams[t]) or (j == v and i in not_in_teams[t]))
    return max([l for l in range(len(teams))], key=lambda l: v_into_teams[l])

def min_k_partition_solve(k):
    """Returns a function that approximates min-k partition."""
    teams = [[] for _ in range(k)]
    not_in_teams = [[] for _ in range(k)]
    def min_k_partition_helper(G: nx.Graph):
        for v in G.nodes:
            i = potential_change_partition(G, teams, not_in_teams, v)
            teams[i].append(v)
            for j in range(len(not_in_teams)):
                if j != i:
                    not_in_teams[j].append(v)
            G.nodes[v]['team'] = i + 1
    return min_k_partition_helper

def potential_change_partition(G, teams, not_in_teams, v):
    """Returns the index of the team that would minimize the edges within if v is placed into that team."""
    v_into_teams = [0 for _ in range(len(teams))]
    for t in range(len(teams)):
        v_into_teams[t] += sum(d for i, j, d in G.edges(data='weight') if (i == v and j in teams[t]) or (j == v and i in teams[t]))
    return min([l for l in range(len(teams))], key=lambda l: v_into_teams[l])