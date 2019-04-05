from collections import defaultdict
import random

FOLLOWER_COUNT = 174
follower_list = [i+1 for i in range(FOLLOWER_COUNT)]


class Graph():
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

g = Graph()
edges = []
print("Creating graph...")
d = set()
for follower in follower_list:
    # add edge
    #g.add_edge(0, follower)
    if (0, follower) not in d and (follower, 0) not in d:
        edges.append([0, follower])
        d.add((0, follower))
        d.add((follower, 0))
    # do edge exist between this follower and other followers?
    # with 30% probability add edge
    for other_followers in follower_list:
        if follower == other_followers:
            continue
        x = random.random()
        if x < .10:
            #g.add_edge(follower, other_followers)
            if (follower, other_followers) not in d and (other_followers, follower) not in d:
                edges.append((follower, other_followers))
                d.add((follower, other_followers))
                d.add((other_followers, follower))

print(len(edges))
print("Writing graph...")
with open('analysis/output.csv', 'w') as f:
    f.write('Source, Target\n')
    for edge in edges:
        f.write(str(edge[0]) + ',' + str(edge[1])+'\n')
