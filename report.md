# 1. MADDPG
### 1.1 Motivation
* Due to the interactions between multiple agents, the state of game could be not stationary, and hard(impossible) to learn.
* The solution is using Actor-Critic, more specifically "centralized critic" + "decentralized actors"
* Benefits of centralized critic
  * training with global estimation, which let considering collaborate and competitive possible
* Benefits of decentralized actors
  * independent actors during working
  * communication between actors

### 1.2 Notation and Model
* There are $N$ agents
* with a set of state $S$ in total
* each agent has a set of possible action $\mathcal{A}_1,\dots,\mathcal{A}_N$
* each agent has a set of observation $\mathcal{O}_1,\dots,\mathcal{O}_N$
* each agent has own a stochastic policy $\pi_{\theta_i}:\mathcal{O}_i\times\mathcal{A}_i\mapsto[0,1]$
  * or a deterministic policy $\mu_{\theta_i}:\mathcal{O}_i\mapsto\mathcal{A}_i$
* let $\vec{o}=o_1,o_2,\dots,o_N$ as observation of all agents
* let $\vec{\mu}=\mu_1,\mu_2,\dots,\mu_N$, which are parameterized by $\vec{\theta}=\theta_1,\theta_2,\dots,\theta_N$, where $\mu_{\theta_i}$ abbreviated as $\mu_i$.
* the critic in MADDPG learns **a centralized action-value function** $Q_i^{\vec{\mu}}(\vec{x},a_1,a_2,\dots,a_N)$ for the i-th agent, where $a_1\in\mathcal{A}_1\dots a_N \in\mathcal{A}_N$
  * For the simplest case, we concatenate states of all agents together as $\vec{x}=\vec{o}=(o_1,o_2,\dots,o_N)$
  * or we can add more reward structure for competitive setting or others.
* each $Q_i^{\vec{\mu}}$ is learned **separately**.
  * separately means each agent has own critic network
  * the critic of each agent learns from training data of all agents
  * a better illustration can be found in Fig.1 
![maddpg](./imgs/maddpg.gif)
* **critic updates** with
$$
\begin{eqnarray}
\mathcal{L}(\theta_i) &=& \mathbb{E}[(Q_i^{\vec{\mu}}(\vec{o},a_1,a_2,\dots,a_N) -y)^2] \\
\text{where } y &=& r_i + \gamma Q_i^{\vec{\mu}'}(\vec{o}',a'_1,a'_2,\dots,a'_N) \vert_{a_j'=\vec{\mu}'_j(o_j)}
\end{eqnarray}
$$
  * prime notion means target network, e.g. $Q_i^{\vec{\mu}'}$ is target critic network, delayed updating. $\vec{\mu}'$ is **target actor network**.
  * $\vert_{a_j'=\vec{\mu}'_j(o_j)}$ means using action $a_j'$ of the j-th agent from j-th **target actor network**, which suppose to max Q

* **actor update** with
$$
\nabla_{\theta_j}J(\mu_j)=\mathbb{E}_{\vec{o},a\in \mathcal{D}}[\nabla_{\theta_j}\mu_i(a_i\vert o_i)\nabla_{a_i}Q_i^{\vec{\mu}}(\vec{o},a_1,\dots,a_N)\vert_{a_i=\mu_i(o_i)}]
$$
  * $\vert_{a_i=\mu_i(o_i)}$ means using action from **local actor network**. 
  * $\mathcal{D}$ contains the SARS tuples $(o,a,r,o')$ for **all agents**. 
  * $Q_i^{\vec{\mu}}$ is centralized action-value function.
  * $Q_i^{\vec{\mu}}(\vec{o},a_1,\dots,a_N)$ is the TD view of $R(\tau)=r_1+r_2+\dots+r_H+r_{H+1}$, recall
$$
\nabla_{\theta}J(\theta) \approx \hat{g} := \frac{1}{m} \sum_i^m \sum_{t=0}^H \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)})
$$

### 1.3 Inferring Policies of Other Agents (Communication)

### 1.4 Agents with Policy Ensembles
* each agent has K different sub-policies
  * equivalents to K actor networks
* At each episode, we randomly select one particular sub-policy for each agent to execute. 
### Reference
* a good [blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#maddpg) 
* the original paper ([Lowe et al., 2017](https://arxiv.org/pdf/1706.02275.pdf)). 
* The original paper and this explanation blog have very clean and beautiful math notation.

* a good [blog](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/) from openai
