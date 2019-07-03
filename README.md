[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project: MADDPG for Collaboration
Different from single agent model, agents in this project should be capable to collaborate to max both scores as much as possible.

# 1. Unity Tennis
a 2D tennis environment.

### 1.1 Reward
![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play. And this is a collaboration model. 

### 1.2 State Space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Number of stacked vector observation is 3.

a single observation s=[x_ball, y_ball, vx_ball, vy_ball, x_racket, y_racket, vx_racket, vy_racket]. need to confirm the order and axis setup.

### 1.3 Action Space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

a single action a=[move_horizon, move_verticle]. all actions are between -1 and 1.

### 1.4 Task
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### 1.5 Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
2. Place the file in the root folder of GitHub repository, and unzip (or decompress) the file. 
3. The environment was modified by Udacity, and it is **not identical** to original unity tennis environment.

# 2. Unity Crawler (Optional)
![Trained Agent][image2]

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

2. Place the file in the root folder of GitHub repository, and unzip (or decompress) the file. 
3. The environment was modified by Udacity, and it is **not identical** to original unity crawler environment.