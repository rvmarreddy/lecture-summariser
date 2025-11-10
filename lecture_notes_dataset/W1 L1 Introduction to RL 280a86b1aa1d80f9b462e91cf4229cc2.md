# W1 L1: Introduction to RL

2 October 2025 

[Lecture 1 - What is Reinforcement Learning.pdf](W1%20L1%20Introduction%20to%20RL/Lecture_1_-_What_is_Reinforcement_Learning.pdf)

## What is Reinforcement Learning?

<aside>
💡

**Reinforcement Learning** is a computational approach to **goal-directed learning from interaction.**

</aside>

![Screenshot 2025-10-02 at 12.51.38.png](W1%20L1%20Introduction%20to%20RL/Screenshot_2025-10-02_at_12.51.38.png)

<aside>
💡

The **environment** is the world the agent interacts with.

The **state** is the current situation or snapshot of that environment. 

The **agent** is the decision-maker that chooses actions based on its current understanding (its policy). 

</aside>

- The agents goal is to earn as much reward as possible

A more intuitive definition:

> **RL** is **learning how to act** through **trial-and-error** interaction with the world.
> 
- Want to be able to decide what action to take in each state in order to maximise this reward signal that I earn in the long run.

> A **policy** is a **strategy or rule** that tells an agent **what action to take in each situation (state)**.
> 
- e.g. Greedy policy: the agent picks the action with the highest expected reward in the current state

**Summary**

- Have an agent interacting with some environment.
- We consider the environment at a particular time, a state, and an action is made to casue the state to change in some way.
- The agent observes this change in the form of the transitioned state.
- We represent an agent’s goals through the addition of aa reward signal.
- The agent’s objective is to choose actions which maximise the expected reward that it is able to earn in the long run.

---

## What can Reinforcement learning do?

- We aim to solve:
    
    > A **sequential decision problem** is a situation where the agent must take a sequence of actions over time where each action affects not only immediate rewards but also the overal reward.
    > 
    - e.g. AlphaGo Zero
- If a given problem requires an agent to make a sequence of decisions in order to reach some goal, we can make use of RL.
    - e.g. **AlphaFold for protein synthesis** or **self driving car technology**

---

## Key features of reinforcement learning

- **rewards can be delayed**
    - tetris example: one block actions affect subsequent blocks
    - short-time sacrifices may lead to long-term gains
- Trade off between **exploration and exploitation**
    - may find a policy which works but may not be the best policy
    - exploring too much means you are exhausting resources where you could have exploited the method you know will work

---

## RL as a branch of machine learning

<aside>
💡

An **RL algorithm** uses an agents experience to learn a policy that maximises the total reward in the long run.

</aside>

- Using the RL algorithm, we learn a policy that describes the action we take given the state we are in.
- The policy describes a **mapping from state inputs to action outputs**.
    - similar to supervised learning however RL agents are not told which action to take in each state
    - Instead learn through trial-and-error interactions with the world
- Similarly, RL **doesn’t rely on examples of correct behaviour**.
    - Not unsupervised learning since agents do not try to find structure in unlabelled data.
        - In unsupervised learning there is no reward or feedback, the algorithm just models the data distribution to find regularities or patterns.
    - They learn a policy that maximises a reward signal
- RL is a branch in its own right