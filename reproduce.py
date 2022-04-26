# Based on supplementary material
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

N, num_agents = 100, 2

size = N*N
m = np.random.randint(num_agents + 1, size=size)

def get_neighbor_idxs(idx):
    all_neighbors = [idx - N, idx + N]

    start_row = (idx//N)*N
    end_row = start_row + N - 1
    if idx != end_row: all_neighbors.append(idx + 1)
    if idx != start_row: all_neighbors.append(idx - 1)

    return np.array([n for n in all_neighbors if n >= 0 and n < size])

def U(idx, neighbors, J=0.5, R=0.5):
    s = 0
    for n_idx in neighbors:
        s += J*(m[n_idx] == m[idx]) - R*(1 - m[n_idx] == m[idx])*(1 - m[n_idx] == 0)

    return s

def do_move(idx, new_idx):
    m[new_idx] = m[idx]
    m[idx] = 0

def faster_random_choice(l): 
    if len(l) == 1: return l[0]
    return l[np.random.randint(0, len(l))]

def agent_move(idx, T=0.5):
    if m[idx] == 0: return

    neighbor_idxs = get_neighbor_idxs(idx)
    empty_neighbors = neighbor_idxs[m[neighbor_idxs] == 0]
    if not len(empty_neighbors): return

    # Pick new site
    new_idx = faster_random_choice(empty_neighbors)

    # More neighbors of the same type
    neighbors = m[neighbor_idxs]

    new_neighbor_idxs = get_neighbor_idxs(new_idx)
    new_neighbors = m[new_neighbor_idxs]

    my_type = m[idx]
    def num_of_type(neighbors):
        return len([n for n in neighbors if n == my_type])

    if num_of_type(new_neighbors) > num_of_type(neighbors):
        do_move(idx, new_idx)
        return

    # Probability func
    p = np.exp((U(idx, neighbor_idxs) - U(new_idx, new_neighbor_idxs))/T)
    if np.random.rand() >= p:
        do_move(idx, new_idx)
        return

def simulate():
    n_steps = 50*size
    for i in range(n_steps):
        if i % int(n_steps/100) == 0:
            print('At {:.0f}%'.format(100*i/n_steps), end='\r')

        idx = np.random.randint(0, size)
        agent_move(idx)

    # Plot
    im = m.reshape(N, N)
    plt.imshow(im)
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    simulate()

