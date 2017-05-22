[![dqnavg](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnweC1uYkI2eWtyUnM  "Link to my blog")](https://adamtiger.github.io/ai/)

# Looking deeper, understanding the world better

## A3C

In this project the aim is to implement the A3C algorithm. A3C stands for Asynchronous Advantage Actor Critic. The implementation basically follows the details provided by:

[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

The project is in-progress.

## BasicExamples

### MountainCar

In this subproject the classical control problem Mountain Car was examined and solved in C++. There is no third party library in the project. The applied reinforcement learning algorithm was: Sarsa with coarse coding.

### NArmedBandit

The classical N armed bandit was solved with a TD like algorithm. The solution was tested in both stationary and non-stationary environments.

### Q-learning

In this subproject the Q-learning algorithm was tested in the classical Windy Gridworld.

## CV4RLframework

This project strongly connects to the [cv4sensorhub](http://bmeaut.github.io/cv4sensorhub/) framework. Some of the UI tools use reinforcement learning to enhance its performance. cv4sensorhub is a C# based software which can be easily integrated into business intelligence systems. Unfortunately, the training is difficult under C#. This framework provides an environment for this purpose. For details see the [documentation](/CV4RLframework/docs/overview.md).

## DoubleDQN

In this project the break-through result on using machine learning to play Atari2600's games is reproduced. The following two article is taken account:

[Human-level control through deep reinforcment learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

In this case three games are examined: Breakout, Freeway and Riverraid. The environment is given by OpenAi gym, the neural network was built with Keras. The code was tested on GPU and the currently achieved maximal reward in a game was 181. The parameter tuning is still in-progress.

## WindyGridWorld

In this project the several algorithms (Sarsa, Q, Q_lambda, Sarsa_lambda) were tested in the classical Windy Gridworld.
