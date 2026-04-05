import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import math


@dataclass
class Agent:
    id: int
    name: str
    role: str
    state: Dict[str, float] = field(default_factory=lambda: {
        'cooperation': random.random(),
        'conflict_potential': random.random(),
        'knowledge': random.random()
    })


class MCTSInteractionNode:
    def __init__(self,
                 agents: List[Agent],
                 parent=None,
                 action: Optional[Tuple[Agent, Agent]] = None):
        """
        Initialize MCTS node for agent interactions

        :param agents: List of agents in the simulation
        :param parent: Parent node in the search tree
        :param action: Interaction action (sender, receiver)
        """
        self.agents = agents
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0
        self.action = action

        # Unique identifier for tracking
        self.id = id(self)

    def is_fully_expanded(self):
        """
        Check if all possible interactions have been explored
        """
        return len(self.children) == len(self.agents) * (len(self.agents) - 1)

    def select_child(self):
        """
        Select child node using Upper Confidence Bound (UCB)
        """
        log_total_visits = math.log(self.visits) if self.visits > 0 else 0

        def ucb_score(child):
            # Exploration-exploitation balance
            if child.visits == 0:
                return float('inf')

            exploitation = child.total_value / child.visits
            exploration = math.sqrt(2 * log_total_visits / child.visits)

            return exploitation + exploration

        return max(self.children, key=ucb_score)

    def expand(self):
        """
        Expand the search tree by creating new interaction possibilities
        """
        # Find unexplored interactions
        explored_interactions = set((child.action for child in self.children))

        # Generate possible interactions
        for sender in self.agents:
            for receiver in self.agents:
                if sender != receiver:
                    interaction = (sender, receiver)
                    if interaction not in explored_interactions:
                        # Create new child node with this interaction
                        child = MCTSInteractionNode(
                            agents=self.agents,
                            parent=self,
                            action=interaction
                        )
                        self.children.append(child)
                        return child

        return None

    def simulate_interaction(self, interaction: Tuple[Agent, Agent]) -> float:
        """
        Simulate the outcome of an interaction between two agents

        :param interaction: Tuple of (sender, receiver)
        :return: Interaction value/impact
        """
        sender, receiver = interaction

        # Interaction probability based on agent states
        interaction_prob = (
                sender.state['cooperation'] *
                (1 - receiver.state['conflict_potential'])
        )

        # Interaction impact
        impact = interaction_prob * (
                sender.state['knowledge'] +
                receiver.state['knowledge']
        )

        # Update agent states
        sender.state['knowledge'] += 0.1 * impact
        receiver.state['knowledge'] += 0.1 * impact

        return impact

    def backpropagate(self, value: float):
        """
        Backpropagate the interaction value up the tree

        :param value: Interaction value to propagate
        """
        current = self
        while current:
            current.visits += 1
            current.total_value += value
            current = current.parent


class MCTSInteractionSimulation:
    def __init__(self, num_agents: int = 100, time_steps: int = 100):
        """
        Initialize MCTS interaction simulation

        :param num_agents: Number of agents
        :param time_steps: Number of simulation time steps
        """
        # Generate agents
        self.agents = [
            Agent(
                id=i,
                name=f"Agent_{i}",
                role=random.choice(['Plaintiff', 'Defendant', 'Lawyer', 'Witness'])
            ) for i in range(num_agents)
        ]

        self.time_steps = time_steps
        self.interaction_log = []

    def run_mcts(self):
        """
        Run Monte Carlo Tree Search for agent interactions
        """
        for _ in range(self.time_steps):
            # Initialize root node
            root = MCTSInteractionNode(agents=self.agents)

            # Perform MCTS iterations
            for _ in range(len(self.agents) * 10):  # Adjust iteration count as needed
                # Selection
                node = root
                while not node.is_fully_expanded():
                    if not node.children:
                        node = node.expand()
                        break
                    node = node.select_child()

                # Expansion
                if not node.is_fully_expanded():
                    node = node.expand()

                # Simulation
                if node.action:
                    interaction_value = node.simulate_interaction(node.action)

                    # Backpropagation
                    node.backpropagate(interaction_value)

                    # Log interaction
                    self.interaction_log.append({
                        'sender': node.action[0].id,
                        'receiver': node.action[1].id,
                        'value': interaction_value
                    })

    def visualize_interactions(self):
        """
        Create network visualization of interactions
        """
        plt.figure(figsize=(20, 20))

        # Create graph
        G = nx.DiGraph()

        # Add edges from interaction log
        for interaction in self.interaction_log:
            G.add_edge(
                interaction['sender'],
                interaction['receiver'],
                weight=interaction['value']
            )

        # Use spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=50)

        # Draw edges with varying thickness based on interaction value
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for (u, v) in edges]

        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=weights,
            alpha=0.5,
            arrows=True
        )

        plt.title('Agent Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('agent_interaction_network.png', dpi=300)
        plt.close()

    def generate_interaction_summary(self):
        """
        Generate summary of interactions
        """
        # Aggregate interaction statistics
        interaction_counts = {}
        interaction_values = {}

        for interaction in self.interaction_log:
            # Count interactions
            key = (interaction['sender'], interaction['receiver'])
            interaction_counts[key] = interaction_counts.get(key, 0) + 1

            # Aggregate interaction values
            interaction_values[key] = interaction_values.get(key, []) + [interaction['value']]

        # Compute summary
        summary = {
            'total_interactions': len(self.interaction_log),
            'unique_interactions': len(interaction_counts),
            'most_frequent_interaction': max(interaction_counts, key=interaction_counts.get),
            'highest_value_interaction': max(
                interaction_values,
                key=lambda k: sum(interaction_values[k]) / len(interaction_values[k])
            )
        }

        return summary


def main():
    # Run simulation
    simulation = MCTSInteractionSimulation(num_agents=100, time_steps=100)

    # Perform MCTS
    simulation.run_mcts()

    # Visualize interactions
    simulation.visualize_interactions()

    # Generate and print summary
    summary = simulation.generate_interaction_summary()
    print("Interaction Simulation Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()