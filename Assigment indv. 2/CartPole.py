import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

class Actor_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(4, 64)
        self.actor_head = nn.Linear(64, 2)
        self.critic_head = nn.Linear(64, 1)
    
    def forward(self, state):
        x = torch.relu(self.hidden_layer(state))
        actions = self.actor_head(x)
        state_value = self.critic_head(x)
        return actions, state_value
    
# Person
def User(global_model, optimizer, gamma, env_name, user_id):
    environment = gym.make(env_name)
    localmodel = Actor_Critic()
    localmodel.load_state_dict(global_model.state_dict())

    while True:
        state, _ = environment.reset()
        done = False
        probs_list = []
        values_list = []
        rewards_list = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            actions, state_value = localmodel(state_tensor)
            action_probs = torch.softmax(actions, dim=0)
            action = torch.distributions.Categorical(action_probs).sample()

            probs_list.append(torch.log(action_probs[action]))
            values_list.append(state_value)

            next_state, reward, terminated, _, _ = environment.step(action.item())
            rewards_list.append(reward)

            state = next_state
            done = terminated

        # Compute returns
        returns = []
        discounted_sum = 0

        for reward in reversed(rewards_list):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values_list).squeeze()
        log_probs = torch.stack(probs_list)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        total_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()


        # Copy local gradients to global model
        for global_param, local_param in zip(global_model.parameters(), localmodel.parameters()):
            global_param._grad = local_param.grad

        optimizer.step()

        # Sync local model with updated global model
        localmodel.load_state_dict(global_model.state_dict())
        print(f"Worker {user_id} | Episode reward: {sum(rewards_list)}")

# Main
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    env_name = "CartPole-v1"
    gamma = 0.99
    num_workers = 4

    global_model = Actor_Critic()
    global_model.share_memory()

    optimizer = optim.Adam(global_model.parameters(), lr=1e-3)

    processes = []
    for user_id in range(num_workers):
        p = mp.Process(
            target=User,
            args=(global_model, optimizer, gamma, env_name, user_id)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


