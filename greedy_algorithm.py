from starter import *
import random

def optimize_k(G: nx.Graph):
    """Searches for the optimal k for a given graph G."""
    
    if G.number_of_nodes() > 100:
        quick_search = 2
    else:
        quick_search = 0

    # TODO: Save time by finding k by with binary search, instead of incrementing it up by 1 each time
    # TODO: Clean up this code

    G_copy = G.copy()

    k = 2
    max_k_cut_solve(k)(G_copy)
    validate_output(G_copy)
    minimum_score = score(G_copy)
    second_minimum_score = minimum_score
    best_k = 2
    second_best_k = best_k
    best_G = G_copy
    #print("Score:", minimum_score, "k:", k)

    if quick_search:
        increase_k = True
        while increase_k:
            G_copy = G.copy()
            k += quick_search
            max_k_cut_solve(k)(G_copy)
            current_score = score(G_copy)
            #print("Score:", current_score, "k:", k)
            if current_score < minimum_score:
                second_best_k = best_k
                best_k = k
                best_G = G_copy
                minimum_score = current_score
            else:
                increase_k = False

        increase_k = True
        k = second_best_k
        search_space = 2 * quick_search - 1

        if (best_k == second_best_k):
            search_space = 1

        for i in range(1, search_space + 1):
            G_copy = G.copy()
            if k + i == best_k:
                continue
            max_k_cut_solve(k + i)(G_copy)
            current_score = score(G_copy)
            #print("Score:", current_score, "k:", k + i)
            if current_score < minimum_score:
                best_k = k + i
                best_G = G_copy
                minimum_score = current_score
                break

        # while increase_k:
        #     G_copy = G.copy()
        #     k += 1
        #     if k == best_k:
        #         continue
        #     max_k_cut_solve(k)(G_copy)
        #     current_score = score(G_copy)
        #     print("Score:", current_score, "k:", k)
        #     if current_score < minimum_score:
        #         best_k = k
        #         best_G = G_copy
        #         minimum_score = current_score

    else:
        increase_k = True
        while increase_k:
            G_copy = G.copy()
            k += 1
            max_k_cut_solve(k)(G_copy)
            current_score = score(G_copy)
            #print("Score:", current_score, "k:", k)
            if current_score < minimum_score:
                best_k = k
                best_G = G_copy
                minimum_score = current_score
            else:
                increase_k = False


    # weight = sum(d for u, v, d in G.edges(data='weight'))
    # print("number of nodes:", G.number_of_nodes(), "\nnumber of edges:", G.number_of_edges(), "\nweight of edges:", weight, "\nk:", best_k)

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
        list_of_nodes = list(G.nodes)
        random.shuffle(list_of_nodes)
        for v in list_of_nodes:
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