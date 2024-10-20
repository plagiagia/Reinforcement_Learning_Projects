import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import os
from utils import save_model, load_model

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

# Create the Taxi environment
env = gym.make('Taxi-v3', render_mode='rgb_array')
env = RecordVideo(env, video_folder="EpisodeRecordings", name_prefix="taxi-agent-training",
                  episode_trigger=lambda x: x % 1000 == 0)
env = RecordEpisodeStatistics(env)

# Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])


# Function to train the agent
def train_q_learning(env, episodes=10000):
    global q_table
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        while not done:
            # Explore or exploit
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)

            # Q-learning formula
            q_table[state, action] = q_table[state, action] + alpha * (
                        reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            state = next_state

        # Logging after every 1000 episodes
        if episode % 1000 == 0:
            print(f"Episode {episode} completed")

    # Save the trained Q-table
    save_model(q_table, "models/q_table.npy")
    print("Training completed and model saved.")


# Main execution
if __name__ == "__main__":
    # Check if a pre-trained model exists
    if os.path.exists("models/q_table.npy"):
        q_table = load_model("models/q_table.npy")
        print("Loaded pre-trained Q-table.")
    else:
        print("No pre-trained model found, starting training.")
        train_q_learning(env)
    env.close()
