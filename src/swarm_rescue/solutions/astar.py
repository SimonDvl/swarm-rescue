import heapq
import numpy as np

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # coût du départ au nœud actuel
        self.h = 0  # estimation du coût du nœud actuel à l'objectif
        self.f = 0  # coût total : g + h

    def __lt__(self, other):
        return self.f < other.f
    
def astar(start, end, grid):
    """
    Algorithme A*
    """
    # Initialisation des nœuds de départ et d'arrivée
    start_node = Node(start)
    end_node = Node(end)

    # Initialisation des listes ouvertes et fermées
    open_set = []
    closed_set = set()

    # Ajout du nœud de départ à la liste ouverte
    heapq.heappush(open_set, (start_node.f, start_node))

    while open_set:
        # Récupération du nœud avec le coût total le plus bas depuis la liste ouverte
        current_node = heapq.heappop(open_set)[1]

        # Vérification si le nœud actuel est le nœud d'arrivée
        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Retourner le chemin inversé

        # Ajout du nœud actuel à la liste fermée
        closed_set.add(current_node.position)

        # Génération des voisins du nœud actuel
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Déplacements possibles : haut, bas, droite, gauche

        for neighbor in neighbors:
            neighbor_position = (current_node.position[0] + neighbor[0], current_node.position[1] + neighbor[1])

            # Vérification des limites du graphe
            if (0 <= neighbor_position[0] < grid.shape[0]) and (0 <= neighbor_position[1] < grid.shape[1]):
                # Vérification si la case est accessible
                if grid[neighbor_position[0]][neighbor_position[1]] == 0:
                    # Création du nœud voisin
                    neighbor_node = Node(neighbor_position, parent=current_node)
                    # Calcul des coûts
                    neighbor_node.g = current_node.g + 1
                    neighbor_node.h = abs(neighbor_position[0] - end_node.position[0]) + abs(
                        neighbor_position[1] - end_node.position[1])
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    # Vérification si le nœud voisin est déjà dans la liste fermée
                    if neighbor_position not in closed_set:
                        # Vérification si le nœud voisin est déjà dans la liste ouverte
                        for item in open_set:
                            if neighbor_node.position == item[1].position and neighbor_node.g > item[1].g:
                                break
                        else:
                            # Ajout du nœud voisin à la liste ouverte
                            heapq.heappush(open_set, (neighbor_node.f, neighbor_node))

    return None  # Aucun chemin trouvé