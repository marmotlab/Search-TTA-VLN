#######################################################################
# Name: env.py
#
# - Adapted from https://gist.github.com/betandr/541a1f6466b6855471de5ca30b74cb31
# - Simple graph class to perform distance calculations (E.g. A-Star, Djikstra)
#######################################################################


class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]
        from_node_edges[to_node] = edge

    def clear_edge(self, from_node):
        if from_node in self.edges:
            self.edges[from_node] = dict()

def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    Implemented to keep the main algorithm clean.
    """
    min_node = None
    for node in q:
        if min_node == None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node

    return min_node


INFINITY = float('Infinity')


def dijkstra(graph, source):
    q = set()
    dist = {}
    prev = {}

    for v in graph.nodes:       
        dist[v] = INFINITY      # unknown distance from source to v
        prev[v] = INFINITY      # previous node in optimal path from source
        q.add(v)                # all nodes initially in q (unvisited nodes)

    # distance from source to source
    dist[source] = 0

    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)

        q.remove(u)

        try:
            if u in graph.edges:
                for _, v in graph.edges[u].items():
                    alt = dist[u] + v.length
                    if alt < dist[v.to_node]:
                        # a shorter path to v has been found
                        dist[v.to_node] = alt
                        prev[v.to_node] = u
        except:
            pass

    return dist, prev


def to_array(prev, from_node):
    """Creates an ordered list of labels as a route."""
    previous_node = prev[from_node]
    route = [from_node]

    while previous_node != INFINITY:
        route.append(previous_node)
        temp = previous_node
        previous_node = prev[temp]

    route.reverse()
    return route


def h(index, destination, node_coords):
    current = node_coords[index]
    end = node_coords[destination]
    h = abs(end[0] - current[0]) + abs(end[1] - current[1])
    return h


def a_star(start, destination, node_coords, graph):
    if start == destination:
        return [], 0
    if str(destination) in graph.edges[str(start)].keys():
        cost = graph.edges[str(start)][str(destination)].length
        return [start, destination], cost
    open_list = {start}
    closed_list = set([])

    g = {start: 0}
    parents = {start: start}

    while len(open_list) > 0:
        n = None
        h_n = 1e5
        for v in open_list:
            h_v = h(v, destination, node_coords)
            if n is not None:
                h_n = h(n, destination, node_coords)
            if n is None or g[v] + h_v < g[n] + h_n:
                n = v

        if n is None:
            print('Path does not exist!')
            return None, 1e5

        if n == destination:
            reconst_path = []
            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
            reconst_path.append(start)
            reconst_path.reverse()
            return reconst_path, g[destination]

        for edge in graph.edges[str(n)].values():
            m = int(edge.to_node)
            cost = edge.length
            if m not in open_list and m not in closed_list:
                open_list.add(m)
                parents[m] = n
                g[m] = g[n] + cost

            else:
                if g[m] > g[n] + cost:
                    g[m] = g[n] + cost
                    parents[m] = n

                    if m in closed_list:
                        closed_list.remove(m)
                        open_list.add(m)

        open_list.remove(n)
        closed_list.add(n)

    print('Path does not exist!')
    return None, 1e5



