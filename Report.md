## Problem
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

We chose to use the second version of the environment which contains contains 20 identical agents, each with its own copy of the environment.

## Benchmark: take a random action
As part of project, we tried to solve the environemnt just by taking a random action (randomly distributed). Although naive this is de-facto an initial benchmarking exercise. 
See the results ![Random action](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/blob/master/Random action.png)
It is clear that we need to do a little more in order to solve the problem.

## "Know thyself": which algorithm to use
Before delving into the core of exercise, there is one main outstanding question here: is the action space continous or discrete? The action space is now continuous, which allows each agent to execute more accurate movements, contrary to the navigation project which was in discrete space with only four discrete actions available: left, right, forward, backward.

Realising which algorithm to use before dealing with the numberscan save a lot of time. This is why theory is so important. Fot this of problems, it is better to use a **policy-based method** rather than a **value-based method** (as we did in our earlier project).

Policy-based methods are **well suited for continuos spaces**, hence they will be very useful in this context. Furthermore, differently from the value-based methods, they can learn also **stochasic policies** rather than just deterministic. Finally they can directly learn the optimal policy ![pi star](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/pi_star.png) without having to maintain a separate value function estimate. Intuititevly we can see how this can be a main advantage of the method both from a theorethical standpoint as well computational. Within the value-based methods, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function, from which an optimal policy is derived. The computational cost for maintaining this estimate of the optimal action-value function can soon become expensive.



## Methodology
After some research (see the suggesting readings: [_Benchmarking Deep Reinforcement Learning for Continuous Control_][benchmarking-paper], [_Continuous Control with Deep Reinforcement Learning_][ddpg-paper]) we decided in favor of using a DDPG algorithm (Deep Deterministic Policy Gradients) on the 20 agents of the Reacher environment.

Our code base has been built upon the udacity repository originally with only slight modifications, please find it here: [Udacity DRL `ddpg-bipedal` notebook][ddpg-repo],however after fixing an initial bug we needed to do further changes. Our final setup is the result of the concepts covered in our course and our own research.

At its core, DDPG is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn. Policy gradient algorithms utilize a form of policy iteration: they evaluate the policy, and then follow the policy gradient to maximize performance. Since DDPG is off-policy and uses a deterministic target policy, this allows for the use of the Deterministic Policy Gradient theorem For more information read [here][ddpg-blog]. 


[benchmarking-paper]: https://arxiv.org/pdf/1604.06778.pdf
[ddpg-paper]: https://arxiv.org/pdf/1509.02971.pdf
[ddpg-repo]: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb
[ddpg-blog]: https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

The algorithm we lives under the umbrella of the so called "actor-critic methods" which, in a nutshell, they implement a "generalised policy iteration" alternating between a policy evaluation and a policy improvement step.
There are two closely related processess:
- actor improvement which aims at improving the current policy: the main task of the agent (actor) is to learn how to act by directly estimating the optimal policy and maximizing rewards;
- critic evaluation which evaluates the current policy: via a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. 
The power behind "Actor-critic methods" is that they combine these two approaches in order to accelerate the learning process. Actor-critic agents are generally also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

The image belowe shows a general graphical representation of actor-critic methods:
 ![Actor-critic architecture](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/Actor-critic architecture.png)

## Actor-Critic models

You can find actor-critic logic implemented here as part the `Agent()` class in `ddpg_agent.py` of the source code ![Actor-critic network](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/actor_critic network.png). 
Please find the source code [here](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/ddpg_agent.py#L44).

Note: As we did with Double Q-Learning in the last project, we're again leveraging local and target networks to improve stability. This is where one set of parameters w is used to select the best action, and another set of parameters w' is used to evaluate that action. In this project, local and target networks are implemented separately for both the actor and the critic.
 

## Network architecture 


You can find both the `Actor()` and the `Critic()` class in `model.py`. Please find the source code [here](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/model.py#L1). 
Our architecture, which is quite standard, have 2 networks with the following structures and hyperameters (layers and number of units per layers):

- Actor: 258 -> 258
- Critic: 258 -> 258 -> 126

Although we tested smaller and bigger networks we realised that just augmenting the networks size might not have been enough, so we kept the structure unchanged from our base code in the Udacity repository.

## Exploration vs Explotation
A major challenge of learning in continuous action spaces is exploration. An advantage of off-policies algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm. As suggested from the Deep Mind paper ([_Continuous Control with Deep Reinforcement Learning_][ddpg-paper]) and from Udacity lessons, a suitable random process to use is the Ornstein-Uhlenbeck process which adds a certain amount of noise to the action values at each timestep. This noise is correlated to previous noise, and therefore tends to stay in the same direction for longer durations without canceling itself out. This allows the arm to maintain velocity and explore the action space with more continuity. Therefore an exploration policy µ is constructed by adding noise sampled from a noise process N to our actor policy

µ(st) = µ(st|θµt) + N

where N can be chosen to suit the environment.

You can find the Ornstein-Uhlenbeck process implemented `OUNoise()` class in `ddpg_agent()` of the source code. Please find the source code [here](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/ddpg_agent.py#L143).


The Ornstein-Uhlenbeck process itself has three hyperparameters that determine the noise characteristics and magnitude:

mu: the long-running mean --> 0
theta: the speed of mean reversion --> 0.15
sigma: the volatility parameter --> 0.2

We haven't experimented with the parameters. 

After many experiments we opted for adding couple of extra parameters:
epsilon: the long-running mean --> 1
epsilon_decay: the speed of mean reversion --> 1e-6

This decay mechanism ensures that more noise is introduced earlier in the training process (i.e., higher exploration), and the noise decreases over time as the agent gains more experience (i.e., higher exploitation).

For more information please read here.
[ddpg-blog]: http://reinforce.io/blog/introduction-to-tensorforce/

Again even in this case we didn't have too much to experiment on those parameters so we just used the ones above. However we found that adding the espilon and epsilon decay hyperparameters massively improved the performance of our agents.

Please find their respective implementations: [epsilon](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/ddpg_agent.py#L79) and [epsilon decay](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/ddpg_agent.py#L127)  



##Learning Interval
We implemented an interval in which the learning step is performed which is equalto 20 timesteps in this instance. As part of each learning step, the algorithm samples experiences from the buffer and runs the method 10 times.

LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes


##Learning stability: gradient clipping and batch normalisation

We implemented gradient clipping set it at 1, therefore placing an upper limit on the size of the parameter updates, and preventing them from growing exponentially. Gradient clipping has been explained during the coursework and documented in the papers quoted as well. It was also present in the base code we used. We didn't try to change this hyperparameter. You can find its implementation [here](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/ddpg_agent.py#L110). 

Along with this, we implemented batch normalization achieving higher model stability after a certain number of episodes and rapidity. We added it both for the [actor](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/model.py#L31) and for the [critic](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/model.py#L67).

In principle we could have applied to every other layer beyond the first one but it would have slowed the learning time. Both those features are essential for solving this challenging environment.

##Experience Replay

Learning from past experiences is an essential part reinforcemtn learning. As with DQN in the previous project, DDPG also utilizes a replay buffer to gather experiences from each agent. Each experience is stored in a replay buffer as the agent interacts with the environment. In this project, there is one central replay buffer utilized by all 20 agents, therefore allowing agents to learn from each others' experiences.

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). Each agent samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

Please find its implemantation [here](https://github.com/MatteoJohnston/deepRL-Continous_control-p2/ddpg_agent.py#L172).

 

## Results 

We were able to solve task in 1 episodes with an average score of 30.03.
![Final results](/home/matteojohnston/deep-reinforcement-learning/p2_continuous-control/Final results.png)



## Further Enhancements

- As discussed in the benchmarking paper other model such as PPO, D3PG or D4PG and many others can probably produce better results. Hence it could be a worthwhile avenue of further research


