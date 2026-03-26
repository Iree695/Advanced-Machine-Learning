import numpy as np
import matplotlib.pyplot as plt


# --- 1. THE ENVIRONMENT ---
class SlotMachine:
    """
    Represents a single Bandit (Arm).
    It has a hidden true probability of paying out (reward = 1).
    """

    def __init__(self, true_probability):
        self.true_probability = true_probability

    def pull(self):
        """
        Simulates pulling the arm.
        Returns:
            1 (Reward) if a random number is < true_probability
            0 (No Reward) otherwise
        """
        # TD: Implement the logic to return 1 or 0 based on probability
        return int(np.random.random() < self.true_probability)

# --- 2. THE AGENTS ---


class NaiveAgent:
    """
    Strategy: Explore each machine for k rounds, then purely exploit the best one.
    (Matches Slide 15 description)
    """

    def __init__(self, n_arms, exploration_rounds_per_arm):
        self.n_arms = n_arms
        self.explore_rounds = exploration_rounds_per_arm
        self.arm_counts = np.zeros(n_arms)  # How many times each arm was pulled
        self.arm_rewards = np.zeros(n_arms)  # Total rewards for each arm
        self.is_exploring = True
        self.best_arm = None

    def select_arm(self):
        """
        Decides which arm to pull.
        """
        # TD: 1. If still in exploration phase: select the next arm to test.
        if self.is_exploring:
            total_exploration_pulls = self.n_arms * self.explore_rounds
            pulls = int(self.arm_counts.sum())
            if pulls < total_exploration_pulls:
                pulls % self.n_arms
                return
        # 2. If exploration is done: calculate the best arm (highest avg reward) and select it forever.         
            self.is_exploring = False
            avg_rewards = np.divide(
                self.arm_rewards,
                self.arm_counts,
                out = np.zeros_like(self.arm_rewards),
                where = self.arm_counts !=0,
            )

            self.best_arm = int(np.argmax(avg_rewards))

        return self.best_arm
    def update(self, arm, reward):
        """
        Updates the agent's knowledge after pulling an arm.
        """
        # TD: Update arm_counts and arm_rewards
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward


class EpsilonGreedyAgent:
    """
    Strategy: With probability epsilon, explore (random arm).
    Otherwise, exploit (current best arm).
    (Matches Slide 16 description)
    """

    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)  # Estimated value (mean reward) of each arm
        self.arm_counts = np.zeros(n_arms)  # How many times each arm was pulled

    def select_arm(self):
        """
        Decides which arm to pull based on Epsilon-Greedy logic.
        """
        # TODO:
        # Generate a random number 'p'.
        # If p < epsilon: Return a random arm index (Exploration).
        # Else: Return the arm index with the highest q_value (Exploitation).
        p = .
        if p < self.epsilon():
            ...

    def update(self, arm, reward):
        """
        Updates the Q-value for the selected arm using the incremental mean formula.
        NewEstimate = OldEstimate + (Reward - OldEstimate) / N
        """
        # TD: Update the count for this arm.
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        # TODO: Update the Q-value for this arm.
        old_value = ...
        new_value = ...
        self.q_values


# --- 3. SIMULATION LOOP ---

def run_experiment(agent, machines, n_steps):
    rewards_history = []

    for i in range(n_steps):
        # 1. Agent chooses an arm
        arm_index = agent.select_arm()

        # 2. Environment provides reward
        reward = machines[arm_index].pull()

        # 3. Agent learns
        agent.update(arm_index, reward)

        rewards_history.append(reward)

    return rewards_history


# --- 4. EXECUTION ---

# Configuration
TRUE_PROBABILITIES = [0.2, 0.5, 0.75, 0.4]  # The "hidden" truths
N_STEPS = 1000

# Initialize Environment
machines = [SlotMachine(p) for p in TRUE_PROBABILITIES]


# TD: Initialize Agents
naive_agent = NaiveAgent(n_arms=len(TRUE_PROBABILITIES), exploration_rounds_per_arm=10)
e_greedy_agent = EpsilonGreedyAgent(n_arms=len(TRUE_PROBABILITIES), epsilon=0.1)

# TD: Run Experiments
naive_rewards = run_experiment(naive_agent, machines, N_STEPS)
e_greedy_rewards = run_experiment(e_greedy_agent, machines, N_STEPS)

# --- 5. VISUALIZATION (Matches Slide 18) ---
def plot_results(agent, true_probs):
    estimated_probs = []

    # Extract estimated probabilities depending on agent type
    if hasattr(agent, 'q_values'):
        estimated_probs = agent.q_values
    elif hasattr(agent, 'arm_rewards'):
        # Calculate mean for naive agent (avoid div by zero)
        estimated_probs = np.divide(agent.arm_rewards, agent.arm_counts,
                                    out=np.zeros_like(agent.arm_rewards),
                                    where=agent.arm_counts != 0)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(true_probs))

    # Plot True Probabilities
    plt.bar(x - 0.2, true_probs, 0.4, label='True Probability (Real)', color='orange')

    # Plot Estimated Probabilities
    plt.bar(x + 0.2, estimated_probs, 0.4, label='Estimated Probability (Predicción)', color='gray')

    plt.xlabel('Slot Machine (Arm)')
    plt.ylabel('Probability of Reward')
    plt.title(f'Agent Estimates vs Real World ({type(agent).__name__})')
    plt.legend()
    plt.show()

plot_results(e_greedy_agent, TRUE_PROBABILITIES)