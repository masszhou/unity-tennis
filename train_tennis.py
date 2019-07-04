from unityagents import UnityEnvironment
import numpy as np

from MADDPG import MADDPG


if __name__ == "__main__":
    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agents = MADDPG(state_size=24,
                    action_size=2,
                    n_agents=2)

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    n_episodes = 2500
    max_t = 1000

    for i in range(n_episodes):  # play game for 5 episodes
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)

        step_counter = 0
        while True:
            # actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
            # actions = np.clip(actions, -1, 1)  # all actions between -1 and 1

            actions = agents.act(states, noise=noise)
            noise *= noise_reduction

            env_info = env.step(actions)[brain_name]  # send all actions to tne environment

            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished

            agents.step(states, actions, rewards, next_states, dones)

            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step

            step_counter += 1
            if np.any(dones) or step_counter>max_t:  # exit loop if episode finished
                break
        # update target network per episode
        agents.update_targets()

        print('episode: {}, total steps: {}, max score: {}'.format(i, step_counter, np.round(np.max(scores), 4)))

    env.close()