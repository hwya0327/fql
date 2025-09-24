# Frictional Q-Learning (FQL)

Frictional Q-Learning (FQL) is a batch deep reinforcement learning algorithm that learn off-policy with frictional constraints.

[runs.zip](https://drive.google.com/file/d/1iRxhVomISLPwDjOazPrOp1I3veaw72xX/view?usp=sharing)



```python
wandb login # only required for the first time
uv run python cleanrl/ppo.py

    --seed 1

    --env-id CartPole-v0

    --total-timesteps 50000

    --track

    --wandb-project-name cleanrltest
```
