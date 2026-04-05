import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

# ----- Parameters and Setup -----
num_agents = 10  # For demonstration, we use 10 agents (IDs: 0 to 9)
max_depth = 10  # Length of the interaction script
interaction_types = ["email", "document", "conversation"]
# Assign a weight to each interaction type (can reflect importance, complexity, etc.)
interaction_weights = {"email": 1, "document": 2, "conversation": 3}
random.seed(42)


# Precompute all possible actions: (sender, receiver, interaction_type)
def get_possible_actions():
    actions = []
    for sender in range(num_agents):
        for receiver in range(num_agents):
            if sender != receiver:
                for itype in interaction_types:
                    actions.append((sender, receiver, itype))
    return actions


all_actions = get_possible_actions()


# ----- Define the Simulation Functions -----
def reward(state):
    """Reward is the total weight of the interactions in the script."""
    return sum(interaction_weights[act[2]] for act in state)


def is_terminal(state):
    """Terminal state: script length reaches max_depth."""
    return len(state) >= max_depth


def random_simulation(state):
    """Run a random simulation from current state until terminal state."""
    current_state = state.copy()
    while not is_terminal(current_state):
        action = random.choice(all_actions)
        current_state.append(action)
    return reward(current_state)


# ----- MCTS Node Class -----
class Node:
    def __init__(self, state, parent=None):
        self.state = state  # List of actions (the script so far)
        self.parent = parent
        self.children = []
        self.wins = 0  # Cumulative reward from simulations
        self.visits = 0  # Number of times node was visited
        # For simplicity, we assume that at each state all actions are available.
        self.untried_actions = all_actions.copy()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        # UCB1 formula: exploitation + exploration term.
        choices_weights = [
            (child.wins / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        # Choose an untried action at random and create a new child node.
        action = self.untried_actions.pop(random.randrange(len(self.untried_actions)))
        new_state = self.state.copy()
        new_state.append(action)
        child_node = Node(new_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result


# ----- MCTS Functions -----
def tree_policy(node):
    """Selection and expansion: traverse the tree until reaching a non-fully expanded node or terminal state."""
    while not is_terminal(node.state):
        if not node.is_fully_expanded():
            return node.expand()
        else:
            node = node.best_child()
    return node


def default_policy(node):
    """Rollout: from the given node, run a simulation until reaching a terminal state."""
    return random_simulation(node.state)


def backup(node, result):
    """Backpropagation: update all ancestors with the simulation result."""
    while node is not None:
        node.visits += 1
        node.wins += result
        node = node.parent


def mcts(root, iterations=1000):
    """Run MCTS starting from the root node for a given number of iterations."""
    for _ in range(iterations):
        leaf = tree_policy(root)
        simulation_result = default_policy(leaf)
        backup(leaf, simulation_result)
    # Return the child of root with the highest average reward
    return root.best_child(c_param=0)


# ----- Run MCTS to Generate an Interaction Script -----
root = Node(state=[])  # Start with an empty script
best_final_node = mcts(root, iterations=10) #2000
best_script = best_final_node.state

print("Best script (sequence of interactions):")
for i, act in enumerate(best_script):
    print(f"Step {i}: Sender: {act[0]}, Receiver: {act[1]}, Type: {act[2]}")

# ----- Aggregate and Plot the Interactions as a Network -----
G = nx.DiGraph()
for act in best_script:
    sender, receiver, itype = act
    if G.has_edge(sender, receiver):
        G[sender][receiver]['weight'] += 1
    else:
        G.add_edge(sender, receiver, weight=1)

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=[w for w in edge_weights], arrowstyle='->', arrowsize=10)
nx.draw_networkx_labels(G, pos)
plt.title("Interaction Network from MCTS-generated Script")
plt.axis("off")
plt.show()
