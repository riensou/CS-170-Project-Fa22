from starter import *

def optimize_k(G: nx.Graph, binary_search=False):
    """Searches for the optimal k for a given graph G."""

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

    if binary_search:

        found_optimal_k = False
        binary_search_phase = True
        final_phase = False
        search_space = -1

        while not found_optimal_k:
            if binary_search_phase:
                G_copy = G.copy()
                k *= 2
                max_k_cut_solve(k)(G_copy)
                current_score = score(G_copy)
                print("Score:", current_score, "k:", k)
                if current_score < minimum_score:
                    second_best_k = best_k
                    second_minimum_score = minimum_score
                    best_k = k
                    best_G = G_copy
                    minimum_score = current_score
                else:
                    binary_search_phase = False
                    k = second_best_k
                    #search_space = k - best_k
                    #second_best_k = best_k
                    #k = best_k
                    #final_phase_k = k
            else:
                G_copy = G.copy()
                k += 1
                if k == best_k:
                    continue
                max_k_cut_solve(k)(G_copy)
                current_score = score(G_copy)
                print("Score:", current_score, "k:", k)
                if current_score < minimum_score:
                    best_k = k
                    best_G = G_copy
                    second_minimum_score = current_score
                    minimum_score = current_score
                elif current_score < second_minimum_score:
                    second_minimum_score = current_score
                else:
                    found_optimal_k = True


                # G_copy = G.copy()
                # k += int(search_space / 2)
                # if k == final_phase_k:
                #     break
                # elif k == best_k:
                #     search_space = best_k - second_best_k
                #     k = second_best_k
                #     k += int(search_space / 2)
                #     final_phase = True
                # print(search_space)
                # max_k_cut_solve(k)(G_copy)
                # validate_output(G_copy)
                # current_score = score(G_copy)
                # print("Score:", current_score, "k:", k)
                # if current_score < minimum_score:
                #     second_best_k = best_k
                #     best_k = k
                #     best_G = G_copy
                #     minimum_score = current_score
                #     search_space = int(search_space/2)
                # elif search_space > 0:
                #     search_space = k - best_k
                #     k = best_k


        return best_k, best_G

        # found_optimal_k = False
        # continue_binary_search = True
        # find_direction = False
        # search_space = -1
        # direction = 1

        # while not found_optimal_k:
        #     if continue_binary_search:
        #         k *= 2
        #     else:
        #         if find_direction and search_space >= 8:
        #             G1_copy = G.copy()
        #             G2_copy = G.copy()
        #             max_k_cut_solve(k + 1 + int(search_space / 2))(G1_copy)
        #             max_k_cut_solve(k - 1 + int(search_space / 2))(G2_copy)
        #             up_score = score(G1_copy)
        #             down_score = score(G2_copy)
        #             print("searching direction:", up_score, down_score)
        #             if up_score > down_score:
        #                 direction = 1
        #                 current_score = up_score
        #             else:
        #                 direction = -1
        #                 current_score = down_score
        #             find_direction = False
        #         k += int(search_space / 2) * direction
        
        #     G_copy = G.copy()

        #     if search_space == 0:
        #         break
        #     max_k_cut_solve(k)(G_copy)
        #     current_score = score(G_copy)
        #     print("Score:", current_score, "k:", k)
        #     if current_score < minimum_score:
        #         best_k = k
        #         best_G = G_copy
        #         minimum_score = current_score
        #         search_space = search_space / 2
        #     else:
        #         if continue_binary_search:
        #             search_space = k - best_k
        #             k = best_k
        #             continue_binary_search = False
        #             find_direction = True
        #         elif search_space > 0:
        #             search_space = k - best_k
        #             k = best_k
        #             find_direction = True
        #         else:
        #             found_optimal_k = True
            
        # return best_k, best_G
    else:
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
                best_G = G_copy
                minimum_score = current_score
            else:
                increase_k = False
        
        print(best_k)

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