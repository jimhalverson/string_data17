# Breakout Group on Reinforcement Learning

## What we will learn
|<img src="https://github.com/jimhalverson/string_data17/blob/master/reinforcement/Gridworld_exploration_untrained_worker.gif" width="400" alt="untrained" />|<img src="https://github.com/jimhalverson/string_data17/blob/master/reinforcement/Gridworld_exploration_trained_worker.gif" width="400" alt="trained" />|
|---|---|
|Untrained worker. Performs essentially a random walk. 718 steps to solution|Same worker trained for roughly a minute (30,000 games). 10 steps to solution.

## Run the code
To run the code simply execute `python example.py`
I don't use any fancy packages, but if you find the code is not running try `pip install --user xxx` where `xxx` is whatever python was complaining about being missing.

## Theory
Reinforcement learning is a machine learning technique based on behavioural psychology. The idea is to have a machine interact with its environment (e.g. traverse the string landscape). The machine receives rewards for interactions that lead to good results (e.g. approaching a physically interesting string vacuum) and/or is punished for steps that lead to undesirable results (e.g. mathematical consistency of the model is violated). The machine explores its environment with the goal to maximize its long-term reward, thus hopefully finding interesting states in its environment (like the Standard Model of Particle Physics).

## Code
In order to exemplify the idea behind reinforcement learning, we will set up a very simple environment called **Gridworld** to mimic the exploration of the string landscape. In our implementation, this environment is essentially a maze with *walls*, *pitfalls*, and an *exit*. In the analogy, the *walls* would be *boundaries of the string landscape* (such as a negative number of branes), the *pitfalls* would be *undesirable physical properties* (such as mathematically inconsistent states), and the *exit* would be a *Standard Model* state. We then expose an agent to this environment and let it learn to navigate the maze and find the exit. I assume next to no knowledge of python and have included a lot of comments into the code.

## Discussion
Possible questions we can discuss are:
 - Is gridworld a good analogy to the actual string landscape? E.g. there are (believed to be) more than one Standard Model-like solution in a given string construction?
 - What could gridworld look like for F-Theory model building, intersecting brane models in IIA/B, Heterotic CYs with vector bundles, Heterotic SCFTs (free fermionic, orbifolds), ...?
 - How does it compare to other machine learning techniques?
 - What are the best methods to use in RL (SARSA, A3C, Wolpertinger, Rainbow, ...)?
