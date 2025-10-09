# Reinforcement-Learning-Basic-Algorithms
The code was wrote when I first started learning RL

### 🧠 GridWorld Environment (for notebooks **2–8**)

The algorithms in notebooks **2–8** are implemented in a custom **7×7 GridWorld** environment.

**Environment details:**
- The grid includes **one start state**, **one goal state**, and **several forbidden states**.  
- The agent aims to find an **optimal policy** that maximizes the total return (sum of discounted rewards).  
- **Rewards and penalties:**
  - Each step incurs a **−1** penalty.  
  - Reaching the goal gives an additional **+50** reward.  
  - Hitting a wall or entering a forbidden cell gives an additional **−10** penalty.  
- **Action stochasticity:**
  - Each chosen action is executed correctly with **80%** probability,  
    deviates **10% left**, and **10% right**.  
  - Action **4 (stay)** means the agent remains in the same state.

This setup allows testing algorithms such as Monte Carlo, SARSA, Q-Learning, and DQN within a discrete and interpretable environment.

---

### 🎯 CartPole Environment (for notebook **9**)

The algorithm in notebook **9** is implemented in the **CartPole-v0** environment from **OpenAI Gym**.

**Environment description:**
- The agent controls a **cart** that moves left or right to balance a **pole** upright on it.  
- Each time step provides a **+1 reward** as long as the pole remains balanced.  
- An episode ends when:
  - The pole falls beyond **±12 degrees**, or  
  - The cart moves more than **2.4 units** away from the center, or  
  - The maximum step limit (200 steps) is reached.  
- The objective is to learn a control policy that keeps the pole balanced for as long as possible.

This environment provides a continuous observation space and a discrete action space, making it a standard benchmark for testing **Deep Q-Network (DQN)** algorithms.

---

### ⚙️ Summary

| Notebook ID | Algorithm Environment | Description |
|--------------|-----------------------|--------------|
| 2–8 | **GridWorld** | Tabular RL algorithms (MC, SARSA, Q-Learning, etc.) |
| 9 | **CartPole-v0** | Deep Q-Network (DQN) implementation |