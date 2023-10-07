### This repo contain some of my re-implement of lunar lander with reinforcement learning algorithm
#### Lunar lander v1
* Solve lunar lander problem from openai Gymnasium use Q-learning and experience replay memory [2].
* The implementation base on [fakemonk1 [1]](https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander) and references from [juliankappler [3]]( https://github.com/juliankappler/lunar-lander)
* Friendly and simple implementation with pytorch
* Run `python lunar_lander_v1.py`
##### Training result
![alt text](https://github.com/kytaithon/lunar-lander/blob/main/Figure%201:%20Reward%20for%20each%20training%20episode.png)

#### Lunar lander v2
* Solve lunar lander problem from openai Gymnasium [2] use Q-learning.
* Periodly update q_target network parameter [4]
* Use softmax policy instead of epsilon greedy policy
* Multiple training step from replay memory

### Reference
* [1] https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander
* [2] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
* [3] https://github.com/juliankappler/lunar-lander
