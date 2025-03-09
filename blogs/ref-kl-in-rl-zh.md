# GRPO 公式有错？RL 中 KL over reference 的迷思

> 童雨轩 tongyuxuan361@gmail.com

最近看到许多朋友在讨论 RL 中的 KL over reference，例如：

- @Hurry Z 的 [GRPO 中的 KL Loss 实现细节问题](https://zhuanlan.zhihu.com/p/28440962040)
- @lym 的 [k2 loss 就是比 k3 loss 好！以及 GRPO_off-policy](https://zhuanlan.zhihu.com/p/28735759256)

恰好前段时间我也对这一问题有所思考，所以写就这篇博客，希望能对大家有所帮助，也希望朋友们能够批评指正。

## 引言：“TRPO/PPO 与 GRPO 中的 KL 为什么不一样？”

我对这一问题的思考开始于 X 上 Fanyi Pu 提出的 [这样一个问题](https://x.com/pufanyi/status/1888845956684370202)：

> A small question about GRPO: I noticed that the KL divergence in GRPO is written as KL(new || old), while TRPO and PPO use KL(old || new) as the constraint/penalty. Is there a difference between the two? Would modifying this part have any impact?
>
> TRPO

```math
\begin{aligned}
& \underset{\theta}{\text{maximize}} L_{\theta_{\text {old }}}(\theta) \\
& \text { subject to } \bar{D}_{\mathrm{KL}}^{\rho_{\theta_{\text {old }}}}\left(\theta_{\text {old }}, \theta\right) \leq \delta
\end{aligned}
```

> PPO

```math
L^{K L P E N}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(\mathbf{y}_t \mid \mathbf{x}_t\right)}{\pi_{\theta_{\text {old }}}\left(\mathbf{y}_t \mid \mathbf{x}_t\right)} \hat{A}_t-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid \mathbf{x}_t\right), \pi_\theta\left(\cdot \mid \mathbf{x}_t\right)\right]\right]
```

> GRPO

```math
\begin{aligned}
& \mathcal{J}_{G R P O}(\theta)=\mathbb{E}\left[q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{o l d}}(O \mid q)\right] \\
& \frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\left\{\min \left[\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,\lt t}\right)}{\pi_{\theta_{o l d}}\left(o_{i, t} \mid q, o_{i,\lt t}\right)} \hat{A}_{i, t}, \text{clip}\left(\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,\lt t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,\lt t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]\right\}
\end{aligned}
```

这个问题本身的答案是非常简单的。这个问题混淆了两种不同的 KL：

1. $`\text{KL}[\pi_{\theta_{old}},\pi_{\theta}]`$，其作用是约束最新策略 $`\pi_{\theta}`$ 不要离采样策略 $`\pi_{\theta_{o l d}}`$ 太远，避免过大的更新导致策略崩溃，从而构成信任域（Trust Region, TR），这也就是 TRPO 中的 TR，而 PPO 作为 TRPO 的近似实现，继承了这一点。
2. $`\text{KL}[\pi_{\theta},\pi_{\theta_{ref}}]`$，其作用是约束最新策略 $`\pi_{\theta}`$ 不要离参考策略 $`\pi_{\theta_{ref}}`$ 太远，从而更充分地利用参考策略中的先验。

另外，这个问题忽略了 TRPO/PPO 公式中的 KL 损失项与 GRPO 公式中的 clip 函数实际上是出于同一目的，即约束 $`\text{KL}[\pi_{\theta_{old}},\pi_{\theta}]`$。如 [PPO 论文](https://arxiv.org/abs/1707.06347) 第 3-4 节所说，两者可以相互替代或结合使用：

> Let $`r_t(\theta)`$ denote the probability ratio $`r_t(\theta)=\frac{\pi_\theta\left(a_t \mid s_t\right)}{\left.\pi_{\theta_{\text {old }}}\left|a_t\right| s_t\right)}`$, so $`r\left(\theta_{\text {old }}\right)=1`$. TRPO maximizes a "surrogate" objective

```math
L^{C P I}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t\right]=\hat{\mathbb{E}}_t\left[r_t(\theta) \hat{A}_t\right] .
```

> ...
>
> The main objective we propose is the following:

```math
L^{C L I P}(\theta)=\hat{\mathbb{E}}_t\left[\min \left(r_t(\theta) \hat{A}_t, \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right)\right]
```

> where epsilon is a hyperparameter, say, $`\epsilon=0.2`$. The motivation for this objective is as follows. The first term inside the $`\min`$ is $`L^{C P I}`$. The second term, $`\text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t`$, modifies the surrogate objective by clipping the probability ratio, which removes the incentive for moving $`r_t`$ outside of the interval $`[1-\epsilon, 1+\epsilon]`$.
>
> ...
>
> Another approach, which can be used as an alternative to the clipped surrogate objective, or in addition to it, is to use a penalty on KL divergence, and to adapt the penalty coefficient so that we achieve some target value of the KL divergence $`d_{\text {targ }}`$ each policy update. In our experiments, we found that the KL penalty performed worse than the clipped surrogate objective, however, we've included it here because it's an important baseline.
>
> In the simplest instantiation of this algorithm, we perform the following steps in each policy update:
>
> - Using several epochs of minibatch SGD, optimize the KL-penalized objective

```math
L^{K L P E N}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right]
```

还可以从以下角度理解两者的共通之处：clip 函数约束的 $`r_t(\theta)=\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)}`$ 就是 $`K L\left[\pi_{\theta_{d d}}, \pi_\theta\right]=\mathbb{E}_{a_t \sim \pi_{\theta_{d t}}\left(\cdot \mid s_t\right)}\left[\log \frac{\pi_{\theta_{d t}}\left(a_t \mid s_t\right)}{\pi_\theta\left(a_t \mid s_t\right)}\right]`$ 中对单个样本 $`(s_t, a_t)`$ 的值 $`\log`$ 上方的真数。

## 新的问题：GRPO 公式中的 KL 项有错？

然而，在思考上述问题的过程中，我注意到了另一个问题：GRPO 公式中的 KL 项似乎存在错误。具体来说，[GRPO 论文](https://arxiv.org/abs/2402.03300)中给出的公式如下：

```math
\begin{aligned}
\mathcal{J}_{G R P O}(\theta) & =\mathbb{E}\left[q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{o l d}}(O \mid q)\right] \\
& \frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\left\{\min \left[\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,\lt t}\right)}{\pi_{\theta_{o l d}}\left(o_{i, t} \mid q, o_{i,\lt t}\right)} \hat{A}_{i, t}, \text{clip}\left(\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,\lt t}\right)}{\pi_{\theta_{o l d}}\left(o_{i, t} \mid q, o_{i,\lt t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]\right\}
\end{aligned}
```

其中 $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$ 按定义展开为（此处使用更常用的符号系统，实际上 $`q`$ 对应 $`s_1`$, $`o_{i,t}`$ 对应 $`\left(\mathbf{s}_t, \mathbf{a}_t\right)`$）：

```math
\begin{aligned}
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & =\mathbb{E}_{\mathbf{\tau} \sim p_{\theta}}\left[\log \frac{p_{\theta}\left(\mathbf{\tau}\right)}{p_{r e f}\left(\mathbf{\tau}\right)}\right] \\
& = \mathbb{E}_{\left(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T\right) \sim p_{\theta}}\left[\log \frac{p_{\theta}\left(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T\right)}{p_{r e f}\left(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T\right)}\right] \\
& = \mathbb{E}_{\left(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T\right) \sim p_{\theta}}\left[\log \frac{p(\mathbf{s}_1) \prod_{t=1}^{T} \pi_{\theta}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t) \prod_{t=1}^{T-1} p(\mathbf{s}_{t+1} \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t, \mathbf{a}_t)}{p(\mathbf{s}_1) \prod_{t=1}^{T} \pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t) \prod_{t=1}^{T-1} p(\mathbf{s}_{t+1} \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t, \mathbf{a}_t)}\right] \\
& = \mathbb{E}_{\left(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T\right) \sim p_{\theta}}\left[\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t)}\right]
\end{aligned}
```

此处将联合概率展开为了：

```math
p(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) = p(s_1) \prod_{t=1}^{T} \pi_{\theta}(\mathbf{a}_t \mid \mathbf{s_1}, \mathbf{a}_1, \cdots, \mathbf{s}_t) \prod_{t=1}^{T-1} p(\mathbf{s}_{t+1} \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t, \mathbf{a}_t)
```

如果决策过程满足 Markov 性质，转移概率只依赖于当前状态和动作，即

```math
p(\mathbf{s}_{t+1} \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t, \mathbf{a}_t) = p(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)
```

则可以进一步简化为：

```math
p(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) = p(s_1) \prod_{t=1}^{T} \pi_{\theta}(\mathbf{a}_t \mid \mathbf{s}_t) \prod_{t=1}^{T-1} p(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)
```

同时 KL over reference 也可以简化为：

```math
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta}}\left[\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_t \mid \mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_t)}\right]
```

不过，目前的语言模型（Language Model, LM）通常使用自回归的方式建模，即当前 token 的生成依赖于所有之前的 token。

如果令 $`s_t`$ 表示第 $`t`$ 个 token，则自回归模型不满足 Markov 性质；而如果令 $`s_t`$ 表示前 $`t`$ 个 token 组成的序列，则自回归模型满足 Markov 性质。

我们先不依赖于 Markov 性质进行推导，以获得更通用的结论。在必要时，我们会再引入 Markov 性质。

而要计算统计量 $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$，需要基于使用当前策略 $`\pi_{\theta}`$ 采样得到的样本 $`\tau \sim p_{\theta}`$，然而这在 off-policy 场景下通常是做不到的，因为我们只有使用采样策略 $`\pi_{\theta_{o l d}}`$ 采样得到的样本 $`\tau \sim p_{\theta_{o l d}}`$，在多轮更新时，我们不会使用当前策略 $`\pi_{\theta}`$ 去重新采样。

而如果说 GRPO 只考虑 on-policy 场景，也是不恰当的，因为 GRPO 公式中的剩余部分，出现了 $`\pi_{\theta_{o l d}}`$，这来自于 off-policy policy gradient (PG) 的推导。

因此，我们可以得出结论，即 GRPO 公式中的 KL 项存在错误，其与 PG 项暗示的 off-policy 场景存在矛盾。

## 另一种可能：直接计算 KL over reference？

但似乎还有另一种可能：我们可以直接准确地计算 $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$ 的值，因为

```math
\mathbb{E}_{\mathbf{a}_t \sim \pi_{\theta}(\cdot \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}\left[\log \frac{\pi_{\theta}(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}\right] = \sum_{a_t \in \mathcal{A}} \pi_{\theta}(a_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t) \log \frac{\pi_{\theta}(a_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t)}{\pi_{r e f}(a_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t)}
```

此处 $`\mathcal{A}`$ 为动作空间，对于语言模型来说即为整个词表，这是我们可以遍历的。

然而，这实际上是不可行的，因为我们优化的 KL 散度与上述期望不同：

```math
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta}}\left[\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}\right]
\neq \mathbb{E}_{\mathbf{a}_t \sim \pi_{\theta}(\cdot \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t)}\left[\log \frac{\pi_{\theta}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_t)}\right]
```

而要直接计算 $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$，需要遍历整个轨迹空间 $`\mathcal{T} = \set{(s_1, a_1, \cdots, s_T, a_T) \mid s_t \in \mathcal{S}, a_t \in \mathcal{A}}`$，其关于轨迹长度 $`T`$ 指数增大，显然是不可行的。

[John Schulman 这篇著名的博客](http://joschu.net/blog/kl-approx.html) 也提到直接计算 KL 散度的开销非常大：

> Our options for computing KL depend on what kind of access we have to $`p`$ and $`q`$. Here, we'll be assuming that we can compute the probabilities (or probability densities) $`p(x)`$ and $`q(x)`$ for any $`x`$, but we can't calculate the sum over $`x`$ analytically. Why wouldn't we be able to calculate it analytically?
>
> 1. Computing it exactly requires too much computation or memory.
> 2. There's no closed form expression.
> 3. We can simplify code by just storing the log-prob, not the whole distribution. This is a reasonable choice if KL is just being used as a diagnostic, as is often the case in reinforcement learning.

## 插曲：KL over reference 放在 reward 中还是 loss 中？

主流 LLM RL 框架中，KL over reference 通常被放在 reward 中，而非 loss 中。以 TRL 为例，对应代码如下：

```python
# https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/ppo_trainer.py#L500-506
# 4. compute rewards
kl = logprobs - ref_logprobs
non_score_reward = -args.kl_coef * kl
rewards = non_score_reward.clone()
actual_start = torch.arange(rewards.size(0), device=rewards.device)
actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
rewards[[actual_start, actual_end]] += scores
```

### InstructGPT

这一做法主要参考的应该是 [InstructGPT 论文](https://arxiv.org/abs/2203.02155)：

> Reinforcement learning (RL). Once again following Stiennon et al. (2020), we fine-tuned the SFT model on our environment using PPO (Schulman et al., 2017). The environment is a bandit environment which presents a random customer prompt and expects a response to the prompt. Given the prompt and response, it produces a reward determined by the reward model and ends the episode. In addition, we add a per-token KL penalty from the SFT model at each token to mitigate overoptimization of the reward model. The value function is initialized from the RM. We call these models "PPO."
>
> We also experiment with mixing the pretraining gradients into the PPO gradients, in order to fix the performance regressions on public NLP datasets. We call these models "PPO-ptx." We maximize the following combined objective function in RL training:

```math
\begin{aligned}
\text { objective }(\phi)= & E_{(x, y) \sim D_\pi^{\mathrm{RL}}}\left[r_\theta(x, y)-\beta \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right]+ \\
& \gamma E_{x \sim D_{\text {remin }}}\left[\log \left(\pi_\phi^{\mathrm{RL}}(x)\right)\right]
\end{aligned}
```

> where $`\pi_\phi^{\mathrm{RL}}`$ is the learned RL policy, $`\pi^{\mathrm{SFT}}`$ is the supervised trained model, and $`D_{\text {pretrain }}`$ is the pretraining distribution. The KL reward coefficient, $`\beta`$, and the pretraining loss coefficient, $`\gamma`$, control the strength of the KL penalty and pretraining gradients respectively. For "PPO" models, $`\gamma`$ is set to 0 . Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models.

其中并没有提到为什么讲 KL 项放在 reward 中，而非 loss 中。

### OpenAI 论文中 KL reward 的出处

然而，在 [OpenAI 早期的一篇论文 "Learning to summarize from human feedback"](https://arxiv.org/abs/2009.01325) 中，他们就已经采用了 KL reward，并提及了出处：

> Human feedback policies. We want to use the reward model trained above to train a policy that generates higher-quality outputs as judged by humans. We primarily do this using reinforcement learning, by treating the output of the reward model as a reward for the entire summary that we maximize with the PPO algorithm [58], where each time step is a BPE token. $`{ }^8`$ We initialize our policy to be the model fine-tuned on Reddit TL;DR. Importantly, we include a term in the reward that penalizes the KL divergence between the learned RL policy $`\pi_\phi^{\mathrm{RL}}`$ with parameters $`\phi`$ and this original supervised model $`\pi^{\mathrm{SFT}}`$, as previously done in [25]. The full reward $`R`$ can be written as:

```math
R(x, y)=r_\theta(x, y)-\beta \log \left[\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right]
```

> This KL term serves two purposes. First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode. Second, it ensures the policy doesn't learn to produce outputs that are too different from those that the reward model has seen during training.
>
> For the PPO value function, we use a Transformer with completely separate parameters from the policy. This prevents updates to the value function from partially destroying the pretrained policy early in training (see ablation in Appendix G.1). We initialize the value function to the parameters of the reward model. In our experiments, the reward model, policy, and value function are the same size.

### KL reward 最早的出处

前面 OpenAI 论文中引用的 KL reward 出处 [25] 是 ["Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog"](https://arxiv.org/abs/1907.00456)。

实际上，其中提出引入 KL 最初的形式是 loss 项，而非 reward 项，只是指出了两者的等价性：

> Rather than simply sample from the prior, we would like the $`Q`$-learning algorithm to directly incorporate the prior into the policy. Thus, we use KL-control to penalize divergence between the prior $`p(y \mid x)`$, and the $`Q`$-network policy $`\pi_\theta`$, while still maximizing reward. Given a trajectory of actions, $`\tau=\left\{a_1, a_2, \ldots a_{t-1}\right\}`$, let $`q(\tau)=\prod_{t=1}^T \pi_\theta\left(a_t, s_t\right)`$ be the policy of our $`Q`$-learning algorithm at the trajectory level. Similarly, let $`p(\tau)=\prod_{t=1}^T p\left(a_t \mid s_t\right)`$ be the prior distribution over the trajectory, and $`r(\tau)`$ be the rewards. We seek to maximize the following KL-regularized objective:

```math
L(q)=\mathbb{E}_{q(\tau)}[r(\tau)] / c-D_{K L}[q(\tau) \| p(\tau)]
```

> Since $`D_{K L}[q \| p]=\sum_x q(x)(\log q(x)-\log p(x))`$, we can see that this is equivalent to maximizing the following expected value function of the policy $`\pi_\theta`$ at the action level:

```math
Q^\pi\left(s_t, a_t\right)=\mathbb{E}_\pi\left[\sum^T r\left(s_{t^{\prime}}, a_{t^{\prime}}\right) / c+\log p\left(a_{t^{\prime}} \mid s_{t^{\prime}}\right)-\log \pi\left(a_{t^{\prime}} \mid s_{t^{\prime}}\right)\right]
```

然而，从前面的分析可以看到，这一等价性要求 on-policy 或直接计算，如果在 off-policy 场景下使用样本估计，则不再等价。

## 主流 LLM RL 框架中的实现

我们可以再梳理一番主流 LLM RL 框架中对于 KL over reference 的实现。

### TRL

TRL 使用样本值估计 KL over reference，并将其放在 reward 中。对应代码如下：

```python
# https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/ppo_trainer.py#L500-506
# 4. compute rewards
kl = logprobs - ref_logprobs
non_score_reward = -args.kl_coef * kl
rewards = non_score_reward.clone()
actual_start = torch.arange(rewards.size(0), device=rewards.device)
actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
rewards[[actual_start, actual_end]] += scores
```

此处 `logprobs` 与对应的样本均来自采样策略 $`\pi_{\theta_{old}}`$，而非当前策略 $`\pi_{\theta}`$。对应代码如下：

```python
# https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/ppo_trainer.py#L406-L432
queries = data["input_ids"].to(device)
# ...

with unwrap_model_for_generation(
    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
) as unwrapped_model:
    query_responses, logitss = batch_generation(
        unwrapped_model.policy,
        queries,
# ...
    )


for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
# ...
    logits = logitss[i : i + args.local_rollout_forward_batch_size]
    logprob = selective_log_softmax(logits, response)
```

注意，这里的 KL over reference 作为 $`\mathbb{D}_{K L}\left[\pi_{\theta_{o l d}} \| \pi_{r e f}\right]`$ 的估计值是正确的，但我们希望使用的是 $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$。

随后进行多轮 PPO 更新时，并没有基于当前策略 $`\pi_{\theta}`$ 重新计算 KL over reference $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$。对应代码如下：

```python
# https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/ppo_trainer.py#L528-L577
# Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
for ppo_epoch_idx in range(args.num_ppo_epochs):
    b_inds = np.random.permutation(args.local_batch_size)
    minibatch_idx = 0
    for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
        mini_batch_end = mini_batch_start + args.local_mini_batch_size
        mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
        gradient_accumulation_idx = 0
        for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
            with accelerator.accumulate(model):
                micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                mb_advantage = advantages[micro_batch_inds]
                mb_responses = responses[micro_batch_inds]
                mb_query_responses = query_responses[micro_batch_inds]
                mb_logprobs = logprobs[micro_batch_inds]
                mb_return = returns[micro_batch_inds]
                mb_values = values[micro_batch_inds]


                output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.temperature + 1e-7
                new_logprobs = selective_log_softmax(logits, mb_responses)
                new_logprobs = torch.masked_fill(
                    new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                )
                vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                vpredclipped = torch.clamp(
                    vpred,
                    mb_values - args.cliprange_value,
                    mb_values + args.cliprange_value,
                )
                vf_losses1 = torch.square(vpred - mb_return)
                vf_losses2 = torch.square(vpredclipped - mb_return)
                vf_loss_max = torch.max(vf_losses1, vf_losses2)
                vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                vf_clipfrac = masked_mean(
                    (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                )
                logprobs_diff = new_logprobs - mb_logprobs
                ratio = torch.exp(logprobs_diff)
                pg_losses = -mb_advantage * ratio
                pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                pg_loss_max = torch.max(pg_losses, pg_losses2)
                pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                loss = pg_loss + args.vf_coef * vf_loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
```

### OpenRLHF

#### 采样策略的 KL over reference 作为 reward 项

与 TRL 类似，OpenRLHF 计算了 $`\mathbb{D}_{K L}\left[\pi_{\theta_{o l d}} \| \pi_{r e f}\right]`$，并将其放在 reward 中。对应代码如下：

```python
# https://github.com/OpenRLHF/OpenRLHF/blob/cdcabf3548ed67f7454eed4fb70905ac8faa8694/openrlhf/models/utils.py#L7-L88
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/abs/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        kl_reward = -kl_coef * kl
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward
    else:
        # TODO: write a more efficient version
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)

    return reward
```

同样，此处的 `log_probs=action_logprobs` 在 `make_experience` 时被计算，和对应的样本 `sequences` 都来自采样策略 $`\pi_{\theta_{old}}`$，而非当前策略 $`\pi_{\theta}`$。对应代码如下：

```python
# https://github.com/OpenRLHF/OpenRLHF/blob/cdcabf3548ed67f7454eed4fb70905ac8faa8694/openrlhf/trainer/ppo_utils/experience_maker.py#L592-L595
def make_experience(self, samples: Samples) -> Experience:
    """
    Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
    """
# ...
# https://github.com/OpenRLHF/OpenRLHF/blob/cdcabf3548ed67f7454eed4fb70905ac8faa8694/openrlhf/trainer/ppo_utils/experience_maker.py#L673-L680
    action_log_probs = self.actor(
        sequences,
        num_actions,
        # ...
    )
# ...
# https://github.com/OpenRLHF/OpenRLHF/blob/cdcabf3548ed67f7454eed4fb70905ac8faa8694/openrlhf/trainer/ppo_utils/experience_maker.py#L704-L709
    kl = compute_approx_kl(
        action_log_probs,
        base_action_log_probs,
        # ...
    )
```

#### “当前策略的 KL over reference” 作为 loss 项

此外，OpenRLHF 还支持估计“当前策略的 KL over reference”作为 loss 项。对应代码如下：

```python
# https://github.com/OpenRLHF/OpenRLHF/blob/cdcabf3548ed67f7454eed4fb70905ac8faa8694/openrlhf/trainer/ppo_trainer.py#L337-L470
    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            # pad seq makes the sequence a multiple of ring_attention_size.
            if self.strategy.ring_attn_group is not None:
                pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                    sequences, attention_mask, num_actions, packed_seq_lens, self.strategy.ring_attn_group
                )
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )
# ...
        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )

        if self.args.use_kl_loss:
            if self.initial_model is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    experience.action_mask,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

            if not self.args.packing_samples:
                kl_mean = masked_mean(kl, experience.action_mask, dim=-1)
            else:
                # ...

            kl_loss = kl_mean.mean()
            experience.info["kl"] = kl_loss.item()
        else:
            kl_loss = 0
# ...
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
# ...
        return status
```

注意，这里同名的 `action_log_probs` 来自更新过程中的当前策略 $`\pi_{\theta}`$，而非采样策略 $`\pi_{\theta_{old}}`$。

然而，计算 `action_log_probs` 使用的样本却来自采样策略 $`\pi_{\theta_{old}}`$。

所以，这里对“当前策略的 KL over reference”的估计实际上是错误的。我们会在后文讨论修正的方法。

### verl

#### 采样策略的 KL over reference 作为 reward 项

verl 同样计算了 $`\mathbb{D}_{K L}\left[\pi_{\theta_{o l d}} \| \pi_{r e f}\right]`$，并将其放在 reward 中。对应代码如下：

```python
# https://github.com/volcengine/verl/blob/f8acd9017b4db4eead1f34beb39fce9c39143194/verl/trainer/ppo/ray_trainer.py#L131-L160
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
# ...
    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld
# ...
```

#### “当前策略的 KL over reference” 作为 loss 项

verl 也支持估计“当前策略的 KL over reference”作为 loss 项。对应代码如下：

```python
# https://github.com/volcengine/verl/blob/f8acd9017b4db4eead1f34beb39fce9c39143194/verl/workers/actor/dp_actor.py#L226-L327
def update_policy(self, data: DataProto):
    # make sure we are in training mode
    self.actor_module.train()
# ...
    for epoch in range(self.config.ppo_epochs):
        for batch_idx, data in enumerate(dataloader):
# ...
            self.actor_optimizer.zero_grad()

            for data in micro_batches:
# ...
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                log_prob=log_prob,
                                                                                advantages=advantages,
                                                                                eos_mask=response_mask,
                                                                                cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
# ...
                loss.backward()
# ...
            grad_norm = self._optimizer_step()
# ...
    self.actor_optimizer.zero_grad()
    return metrics
```

同样，这里的估计是错误的。

## 思路 1: KL 散度估计作为 loss 项

将 KL 作为 loss 项的设计背后，是一个自然的思路：先计算 $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$，再使用自动微分。

将 KL 作为 loss 项时，通常有一个隐藏的假设，即对所有与 $\theta$ 相关的量，都计算梯度，也即默认不使用 nograd。这也是目前 OpenRLHF 与 verl 实现 KL 作为 loss 项的方式。

如上文所说，几乎不可能直接计算 $`\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]`$，我们只能基于样本来估计，例如使用 Monte Carlo 估计

```math
\begin{aligned}
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta}}\left[\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}\right] \\
& \approx \frac{1}{N} \sum_{i=1}^{N} \left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}\right)
\end{aligned}
```

其中，$`\left(\mathbf{s}_{i,1}, \mathbf{a}_{i,1}, \cdots, \mathbf{s}_{i,T}, \mathbf{a}_{i,T}\right) \sim p_{\theta}`$。

上面还出现了 KL 散度的其他几种估计方法。

我们先介绍这些估计方法，再来分析其梯度。

### 插曲：如何尽可能准确地估计 KL 散度

前文提到，OpenRLHF 引入了 3 种 KL 散度的估计方法，分别称为 `k1`, `k2`, `k3`，这应该是来自[John Schulman 的博客 "Approximating KL Divergence"](http://joschu.net/blog/kl-approx.html)。

verl 则考虑了更多估计方法。实际上，verl 还考虑了直接计算条件 KL 散度（需要遍历整个词表）来估计，但目前还没有实现。对应代码如下：

```python
# https://github.com/volcengine/verl/blob/f8acd9017b4db4eead1f34beb39fce9c39143194/verl/trainer/ppo/core_algos.py#L351-L383
def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
```

统计意义上，估计的准确性可以由两种指标衡量，即偏差（Bias）和方差（Variance）。

John Schulman 的博客分析了 3 种估计方法的偏差和方差，并给出了结论：

考虑 $`\text{KL}[q,p]=\mathbb{E}_{x \sim q}[\log \frac{q(x)}{p(x)}] \approx \frac{1}{N} \sum_{i=1}^{N} k_j(x_i)`$，其中 $`x_i \sim q`$，令 $`r = \frac{q(x)}{p(x)}`$，则：

1. $`k_{1}= \log r`$ 是无偏估计，但方差较大。
2. $`k_{2}= \frac{1}{2} (\log r)^2`$ 是有偏估计，但方差较小。
3. $`k_{3}= (r - 1) - \log r`$ 是无偏估计，同时方差较小。

其还进行了简单的验证实验：

> Now let's compare the bias and variance of the three estimators for $`\mathrm{KL}[q, p]`$. Suppose $`q=N(0,1), p=N(0.1,1)`$. Here, the true KL is 0.005.

|     | bias/true | stdev/true |
| --- | --------- | ---------- |
| k1  | 0         | 20         |
| k2  | 0.002     | 1.42       |
| k3  | 0         | 1.42       |

> Note that the bias of k2 is incredibly low here: it's $`0.2 \%`$.
>
> Now let's try for a larger true KL divergence. $`p=N(1,1)`$ gives us a true KL divergence of 0.5.

|     | bias/true | stdev/true |
| --- | --------- | ---------- |
| k1  | 0         | 2          |
| k2  | 0.25      | 1.73       |
| k3  | 0         | 1.7        |

> Here, the bias of k2 is much larger. k3 has even lower standard deviation than k2 while being unbiased, so it appears to be a strictly better estimator.

### （on-policy 场景下）各种 KL 估计方法直接自动微分求得的梯度

#### 基于 k1 估计方法求得的梯度：期望为 0

基于 k1 估计方法求得的梯度样本值为

```math
\begin{aligned}
\nabla_{\theta} \sum_{t=1}^{T} \log \frac{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})} & = \nabla_{\theta} \sum_{t=1}^{T} \log \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t}) \\
& = \nabla_{\theta} \log \prod_{t=1}^{T} \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t}) \\
& = \nabla_{\theta} \log \prod_{t=1}^{T} \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t}) + \nabla_{\theta} \log \prod_{t=1}^{T-1} p(s_{i,t+1} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t}, a_{i,t}) + \nabla_{\theta} \log p(s_{i,1}) \\
& = \nabla_{\theta} \log p(s_{i,1}) \prod_{t=1}^{T} \pi_{\theta}(a_{i,t}, s_{i,t+1} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t}) \prod_{t=1}^{T-1} p(s_{i,t+1} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t}, a_{i,t}) \\
& = \nabla_{\theta} \log p_{\theta}(\tau)
\end{aligned}
```

对应的梯度期望为：

```math
\begin{aligned}
\mathbb{E}_{\mathbf{\tau} \sim p_{\theta}} \left[\nabla_{\theta} \log p_{\theta}(\mathbf{\tau})\right] & = \sum_{\mathbf{\tau} \in \mathcal{T}} p_{\theta}(\mathbf{\tau}) \nabla_{\theta} \log p_{\theta}(\mathbf{\tau}) \\
& = \nabla_{\theta} \sum_{\mathbf{\tau} \in \mathcal{T}} p_{\theta}(\mathbf{\tau}) \\
& = \nabla_{\theta} 1 \\
& = 0
\end{aligned}
```

所以基于 k1 估计方法求得的梯度期望为 0，平均意义上不会引起分布改变。

也就是说，如果 on-policy 地优化 k1 估计方法导出的 loss，平均意义上不会引起分布改变。

我们可以进一步考虑 off-policy 的场景，第一个 mini-batch 更新时梯度期望为 0，但由于随机性，仍然会略微改变分布，使得 $\pi_\theta != \pi_{\theta_{old}}$。

随后的 mini-batch 中，再在样本 $\tau \sim p_{\theta_{old}}$ 上计算梯度，则梯度期望变为 $\mathbb{E}_{\mathbf{\tau} \sim p_{\theta_{old}}} \left[\nabla_{\theta} \log p_{\theta}(\mathbf{\tau})\right]$，此时，减小 k1 估计值，就相当于增大来自采样分布 $p_{\theta_{old}}$ 的样本概率，即使模型向 $p_{\theta_{old}}$ 回退。

有趣的是，其作用与 PPO 中的 KL penalty 类似。注意，这并非其本意，因为其最初的计算中使用了 $\pi_{ref}$。

#### 基于 k2/k3 估计方法求得的梯度：暂时无法分辨其意义

基于 k2 估计方法求得的梯度样本值为

```math
\begin{aligned}
\nabla_{\theta} \sum_{t=1}^{T} \frac{1}{2} \left(\log \frac{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}\right)^2 & = \sum_{t=1}^{T} \log \frac{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})} \nabla_{\theta} \log \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})
\end{aligned}
```

基于 k3 估计方法求得的梯度样本为

```math
\begin{aligned}
& \nabla_{\theta}  \sum_{t=1}^{T} \left(\frac{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})} - 1 - \log \frac{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}\right) \\
= & \sum_{t=1}^{T} \left(\frac{\nabla_{\theta} \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})} - \nabla_{\theta} \log \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})\right) \\
= &  \sum_{t=1}^{T} \left(\frac{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})} - 1 \right)\nabla_{\theta} \log \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t}) \\
= &  \sum_{t=1}^{T} \left(\frac{1}{\pi_{r e f}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})} - \frac{1}{\pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})} \right) \nabla_{\theta} \pi_{\theta}(a_{i,t} \mid s_{i,1}, a_{i,1}, \cdots, s_{i,t})
\end{aligned}
```

基于 k2 和 k3 估计方法求得的梯度样本较为复杂，难以分辨其意义。

### 另一个问题：off-policy 场景下 KL 值的估计

如前文所述，在 off-policy 场景下估计当前策略的 KL over reference $\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]$ 时，我们会遇到一个困难：没有采样自当前策略 $\pi_\theta$ 的样本。GRPO 的公式没有处理这一点，OpenRLHF/verl 的 KL over reference loss 实现也忽略了这一点，所以即便不论先估计 KL 散度再求梯度的方法本身就存在问题，这里对 KL 散度的估计本身在 off-policy 场景下也是不准确的。

那么，有没有办法绕过这个困难呢？熟悉 off-policy PG 的朋友可能很容易想到，我们可以使用重要性采样（Importance Sampling, IS）来估计 $\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]$：

```math
\begin{aligned}
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta}}\left[\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}\right] \\
& = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta_{old}}}\left[\frac{p_{\theta}\left(\mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t, \mathbf{a}_t\right)}{p_{\theta_{old}}\left(\mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t, \mathbf{a}_t\right)} \sum_{t=1}^{T} \log \frac{\pi_{\theta}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}{\pi_{r e f}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}\right] \\
& = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta_{old}}}\left[ \left(\prod_{t=1}^{T} \frac{\pi_{\theta}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}{\pi_{\theta_{old}}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}\right) \sum_{t=1}^{T} \log \frac{\pi_{\theta}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}{\pi_{r e f}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}\right] \\
& = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta_{old}}}\left[ \exp \left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}{\pi_{\theta_{old}}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}\right) \sum_{t=1}^{T} \log \frac{\pi_{\theta}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}{\pi_{r e f}\left(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t\right)}\right] \\
\end{aligned}
```

代码则可以按如下方式改正：

OpenRLHF

```diff
kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
+ kl *= torch.exp(torch.sum(action_log_probs - old_action_log_probs, dim=-1))
```

verl

```diff
kld = core_algos.kl_penalty(logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type)
+ kld *= torch.exp(torch.sum(log_prob - old_log_prob, dim=-1))
```

然而，这里改正的只是 KL 散度的估计，而不是其梯度。

同时，显然，添加了 IS 系数的估计值的梯度已经极其复杂，难以分析了。

## 思路 2: 直接估计 KL 散度的梯度

由于我们使用的是梯度法，本质上，我们需要准确估计的是 KL 散度的梯度而非其本身。类似地，在 PG 中，我们需要最大化 $`\mathbb{E}_{\mathbf{\tau} \sim p_{\theta}}[r(\mathbf{\tau})]`$，估计的是其梯度 $`\nabla_{\theta} \mathbb{E}_{\mathbf{\tau} \sim p_{\theta}}[r(\mathbf{\tau})]=\mathbb{E}_{\mathbf{\tau} \sim p_{\theta}}[r(\mathbf{\tau}) \nabla_{\theta} \log p_{\theta}(\mathbf{\tau})]`$ 而不是 $`r(\mathbf{\tau})`$ 本身。

展开 KL over reference 的表达式：

```math
\begin{aligned}
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \mathbb{E}_{(\mathbf{s}_1, \mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) \sim p_{\theta}}\left[\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_t \mid  \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_t \mid \mathbf{s}_1, \mathbf{a}_1, \cdots,\mathbf{s}_t)}\right] \\
& = \sum_{(s_1, a_1, \cdots, s_T, a_T) \in \mathcal{T}} p_{\theta}(s_1, a_1, \cdots, s_T, a_T) \left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(a_t \mid  s_1, a_1, \cdots, s_t)}{\pi_{r e f}(a_t \mid s_1, a_1, \cdots, s_t)}\right) \\
& = \sum_{(s_1, a_1, \cdots, s_T, a_T) \in \mathcal{T}} p(s_1) \left(\prod_{t=1}^{T} \pi_{\theta}(a_t \mid  s_1, a_1, \cdots, s_t) \prod_{t=1}^{T-1} p(s_{t+1} \mid  s_1, a_1, \cdots, s_t, a_t)\right) \left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(a_t \mid  s_1, a_1, \cdots, s_t)}{\pi_{r e f}(a_t \mid  s_1, a_1, \cdots, s_t)}\right)
\end{aligned}
```

计算其梯度：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \nabla_{\theta} \sum_{(s_1, a_1, \cdots, s_T, a_T) \in \mathcal{T}} p(s_1) \left(\prod_{t=1}^{T} \pi_{\theta}(a_t \mid  s_1, a_1, \cdots, s_t) p(s_{t+1} \mid  s_1, a_1, \cdots, s_t, a_t)\right) \left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(a_t \mid  s_1, a_1, \cdots, s_t)}{\pi_{r e f}(a_t \mid  s_1, a_1, \cdots, s_t)}\right) \\
& = \sum_{(s_1, a_1, \cdots, s_T, a_T) \in \mathcal{T}} p(s_1) \left(\prod_{t=1}^{T} p(s_{t+1} \mid  s_1, a_1, \cdots, s_t, a_t)\right) \nabla_{\theta} \left(\left(\prod_{t=1}^{T} \pi_{\theta}(a_t \mid  s_1, a_1, \cdots, s_t) \right) \left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(a_t \mid  s_1, a_1, \cdots, s_t)}{\pi_{r e f}(a_t \mid  s_1, a_1, \cdots, s_t)}\right) \right)
\end{aligned}
```

### 在已知环境中简化 KL 梯度估计

但注意到，LLM 的许多任务中，环境中的状态转移概率分布均为已知的，有时还可能是确定性的（Deterministic）。

当状态转移概率分布已知时，$`\forall t, p_{\theta_i}(a_1, \cdots, s_t, a_t \mid s_1)`$ 都是可以计算的，则 KL over reference 可以直接写成：

```math
\begin{aligned}
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \sum_{(s_1, a_1, \cdots,s_T, a_T) \in \mathcal{T}} p(s_1) p_{\theta}(a_1,\cdots,s_T, a_{T} \mid s_1) \log \frac{p_{\theta}(a_1,\cdots, s_{T},a_{T} \mid s_1)}{p_{r e f}(a_1,\cdots, s_{T},a_{T} \mid s_1)}  \\
\end{aligned}
```

### 简写为 Contextual Bandit

为了方便书写，我们可以进一步将模型简化为 contextual bandit，即令 $`\mathbf{s}_1 = \mathbf{x} \in \mathcal{P}, (\mathbf{a}_1, \cdots, \mathbf{s}_T, \mathbf{a}_T) = \mathbf{y} \in \mathcal{R}`$，其中 $`\mathcal{P}, \mathcal{R}`$ 分别表示 prompt / response 空间，则 KL over reference 变为：

```math
\begin{aligned}
\mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim p_{\theta}}\left[\log \frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{r e f}(\mathbf{y} \mid \mathbf{x})}\right] \\
& = \sum_{(x, y) \in \mathcal{T}} p_{\theta}(x, y) \left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) \\
& = \sum_{(x, y) \in \mathcal{T}} p(s) \pi_{\theta}(y \mid x) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right)
\end{aligned}
```

其梯度变为：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \nabla_{\theta} \sum_{(x, y) \in \mathcal{T}} p(s) \pi_{\theta}(y \mid x) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) \\
& = \sum_{(x, y) \in \mathcal{T}} p(s) \nabla_{\theta} \left(\pi_{\theta}(y \mid x) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right)\right)
\end{aligned}
```

其中梯度项可以进一步展开为：

```math
\begin{aligned}
\nabla_{\theta} \left(\pi_{\theta}(y \mid x) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right)\right) & = \left(\nabla_{\theta} \pi_{\theta}(y \mid x)\right) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) + \pi_{\theta}(y \mid x) \nabla_{\theta} \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) \\
& = \left(\nabla_{\theta} \pi_{\theta}(y \mid x)\right) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) + \pi_{\theta}(y \mid x) \frac{1}{\pi_\theta(y \mid x)} \nabla_{\theta} \pi_{\theta}(y \mid x) \\
& = \left(\nabla_{\theta} \pi_{\theta}(y \mid x)\right) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) + \nabla_{\theta} \pi_{\theta}(y \mid x) \\
& = \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)} + 1\right) \nabla_{\theta} \pi_{\theta}(y \mid x)
\end{aligned}
```

代入回 KL 梯度表达式：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \sum_{(x, y) \in \mathcal{T}} p(s) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)} + 1\right) \nabla_{\theta} \pi_{\theta}(y \mid x) \\
& = \sum_{(x, y) \in \mathcal{T}} p(s) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)} + 1\right) \nabla_{\theta} \pi_{\theta}(y \mid x) \\
& = \sum_{(x, y) \in \mathcal{T}} p(s) \pi_{\theta}(y \mid x) \frac{\nabla_{\theta} \pi_{\theta}(y \mid x)}{\pi_{\theta}(y \mid x)} \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)} + 1\right) \\
& = \sum_{(x, y) \in \mathcal{T}} p(s) \pi_{\theta}(y \mid x) \left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)} + 1\right) \nabla_{\theta} \log \pi_{\theta}(y \mid x) \\
& = \mathbb{E}_{(x, y) \sim p_{\theta}} \left[\left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)} + 1\right) \nabla_{\theta} \log \pi_{\theta}(y \mid x)\right]
\end{aligned}
```

这里为了重新获得期望形式，引入了 $`1 = \pi_{\theta}(y \mid x) / \pi_{\theta}(y \mid x)`$，并利用了 $`\nabla_{\theta} \log \pi_{\theta}(y \mid x) = \frac{\nabla_{\theta} \pi_{\theta}(y \mid x)}{\pi_{\theta}(y \mid x)}`$。

注意到“对数概率梯度的期望为 0”，即：

```math
E_{\mathbf{\tau} \sim p_\theta}\left[b\nabla_\theta \log p_\theta(\tau) \right]=\sum_{\mathbf{\tau} \in \mathcal{T}} p_\theta(\mathbf{\tau}) b \nabla_\theta \log p_\theta(\mathbf{\tau}) = \sum_{\mathbf{\tau} \in \mathcal{T}} \nabla_\theta b p_\theta(\mathbf{\tau}) = b \nabla_\theta \sum_{\mathbf{\tau} \in \mathcal{T}} p_\theta(\mathbf{\tau}) = b \nabla_\theta 1 = 0
```

这意味着，从一个分布采样样本，再在样本上计算同一个分布的对数似然及其梯度，最终得到的梯度期望为 0，即统计意义上不会发生改变。也就是说，分布直接 on-policy 地通过对数似然蒸馏自己的输出，统计意义上不会改变分布。

注意，这与直接自我蒸馏（self-distillation）容易导致模型坍塌（collapse）的结论并不矛盾。因为直接自我蒸馏的工作很少保证 on-policy。

则 KL 梯度表达式可以进一步化简为：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & = \mathbb{E}_{(x, y) \sim p_{\theta}} \left[\left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)} + 1\right) \nabla_{\theta} \log \pi_{\theta}(y \mid x)\right] \\
& = \mathbb{E}_{(x, y) \sim p_{\theta}} \left[\left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) \nabla_{\theta} \log \pi_{\theta}(y \mid x)\right] + \mathbb{E}_{(x, y) \sim p_{\theta}} \left[\nabla_{\theta} \log \pi_{\theta}(y \mid x)\right] \\
& = \mathbb{E}_{(x, y) \sim p_{\theta}} \left[\left(\log \frac{\pi_{\theta}(y \mid x)}{\pi_{r e f}(y \mid x)}\right) \nabla_{\theta} \log \pi_{\theta}(y \mid x)\right]
\end{aligned}
```

进行 Monte Carlo 估计：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & \approx \frac{1}{N} \sum_{i=1}^{N} \left(\log \frac{\pi_{\theta}(y_i \mid x_i)}{\pi_{r e f}(y_i \mid x_i)}\right) \nabla_{\theta} \log \pi_{\theta}(y_i \mid x_i)
\end{aligned}
```

其中 $`(\mathbf{x}_i, \mathbf{y}_i) \sim p_{\theta}`$。

为了使用自动微分计算这一梯度估计式，我们需要构造对应的 loss 函数：

```math
\begin{aligned}
\mathcal{L}^{KL}_{\theta} & = \frac{1}{N} \sum_{i=1}^{N} \text{nograd}\left (\log \frac{\pi_{\theta}(y_i \mid x_i)}{\pi_{r e f}(y_i \mid x_i)}\right) \log \pi_{\theta}(y_i \mid x_i)
\end{aligned}
```

### 还原为已知环境决策过程

将上面的 KL 梯度表达式还原为已知环境决策过程建模的形式：

```math
\begin{aligned}
& \nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]\\
=& \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim p_{\theta}} \left[\left(\log \frac{\pi_{\theta}(\mathbf{y} \mid \mathbf{x})}{\pi_{r e f}(\mathbf{y} \mid \mathbf{x})}\right) \nabla_{\theta} \log \pi_{\theta}(\mathbf{y} \mid \mathbf{x})\right] \\
=& \mathbb{E}_{(\mathbf{s}_{1}, \mathbf{a}_{1}, \cdots, \mathbf{s}_{T}, \mathbf{a}_{T}) \sim p_{\theta}} \left[\left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{1}, \cdots, \mathbf{a}_{t-1}, \mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_{t} \mid \mathbf{s}_{1}, \cdots, \mathbf{a}_{t-1}, \mathbf{s}_t)}\right) \left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{1}, \cdots, \mathbf{a}_{t-1}, \mathbf{s}_t)\right)\right]
\end{aligned}
```

对应的 Monte Carlo 估计式为：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & \approx \frac{1}{N} \sum_{i=1}^{N}  \left(\sum_{t=1}^{T}\log \frac{\pi_{\theta}(a_{i, t} \mid s_{1, t}, \cdots, a_{i, t-1})}{\pi_{r e f}(a_{i, t} \mid s_{1, t}, \cdots, a_{i, t-1})}\right) \left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{i, t} \mid s_{1, t}, \cdots, a_{i, t-1})\right)
\end{aligned}
```

### 利用 Markov 性质化简 KL 梯度估计（减小方差）

前文我们提到过，令 $`s_t`$ 表示前 $`t`$ 个 token 组成的序列时，则自回归模型满足（一阶） Markov 性质，即

```math
p_{\theta_i}(s_{t+1} \mid s_1, \cdots, s_t, a_t) = p_{\theta_i}(s_{t+1} \mid s_t, a_t)
```

此时，我们可以利用 Markov 性质化简 KL 梯度估计式。

#### 参考 PG 如何利用 Markov 性质化简梯度估计

我们可以参考 MDP 建模的 PG 方法中利用 Markov 性质化简梯度估计式的技巧。

PG 的表达式为：

```math
\nabla_\theta J(\theta)=E_{\tau \sim p_\theta(\tau)}\left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right)\left(\sum_{t=1}^T r\left(\mathbf{s}_t, \mathbf{a}_t\right)\right)\right]
```

对应的估计式可以利用 Markov 性质化简为：

```math
\begin{aligned}
&\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\right)\left(\sum_{t=1}^T r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)\right)\\
\to &\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\left(\sum_{t^{\prime} = t}^T r\left(\mathbf{s}_{i, t^{\prime}}, \mathbf{a}_{i, t^{\prime}}\right)\right)
\end{aligned}
```

首先，让我们考虑原始策略梯度公式中对某个轨迹 $`i`$ 的单个时间步 $`t`$ 的贡献：

```math
\nabla_\theta \log \pi_\theta\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right) \sum_{t'=1}^T r\left(\mathbf{s}_{i, t'}, \mathbf{a}_{i, t'}\right)
```

现在，我们将总奖励分解为两部分：$`t' < t`$ 的奖励和 $`t' \geq t`$ 的奖励：

```math
\nabla_\theta \log \pi_\theta\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right) \left(\sum_{t'=1}^{t-1} r\left(\mathbf{s}_{i, t'}, \mathbf{a}_{i, t'}\right) + \sum_{t'=t}^T r\left(\mathbf{s}_{i, t'}, \mathbf{a}_{i, t'}\right)\right)
```

这里关键的洞察是，根据 Markov 性质，$`t' < t`$ 时的奖励 $`\sum_{t'=1}^{t-1} r\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right)`$ 与 $`t`$ 时的策略梯度 $`\nabla_\theta \log \pi_\theta\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)`$ 是独立的。

所以有：

```math
\begin{aligned}
\mathbb{E}_{r \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right) \sum_{t^{\prime}=1}^{t-1} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)\right] & = \mathbb{E}_{\mathbf{a}_t \sim \pi_\theta(\cdot \mid \mathbf{s}_t)}\left[\nabla_\theta \log \pi_\theta\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right] \cdot \mathbb{E}_{(\mathbf{s_1}, \mathbf{a_1}, \cdots, \mathbf{s_{t-1}}, \mathbf{a_{t-1}}) \sim p_\theta}\left[\sum_{t'=1}^{t-1} r\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right)\right] \\
& = 0 \cdot \mathbb{E}_{(\mathbf{s_1}, \mathbf{a_1}, \cdots, \mathbf{s_{t-1}}, \mathbf{a_{t-1}}) \sim p_\theta}\left[\sum_{t'=1}^{t-1} r\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right)\right] \\
& = 0
\end{aligned}
```

这里我们利用了对数概率梯度期望为 0 的性质。

代入 PG 的表达式，即可简化得到：

```math
\nabla_\theta J(\theta)=E_{\tau \sim p_\theta(\tau)}\left[\sum_{t=1}^T \left(\sum_{t'=t}^T r\left(\mathbf{s}_t, \mathbf{a}_t\right)\right)\nabla_\theta \log \pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right]
```

#### 利用 Markov 性质化简 KL 梯度估计性质化简 KL 梯度估计

类似地，对于我们前面得到的 KL 梯度表达式：

```math
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] =  \mathbb{E}_{(\mathbf{s}_{1}, \mathbf{a}_{1}, \cdots, \mathbf{s}_{T}, \mathbf{a}_{T}) \sim p_{\theta}} \left[\left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{1}, \cdots, \mathbf{a}_{t-1}, \mathbf{s}_t)}{\pi_{r e f}(\mathbf{a}_{t} \mid \mathbf{s}_{1}, \cdots, \mathbf{a}_{t-1}, \mathbf{s}_t)}\right) \left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{1}, \cdots, \mathbf{a}_{t-1}, \mathbf{s}_t)\right)\right]
```

我们可以利用 Markov 性质化简为：

```math
\mathbb{E}_{(\mathbf{s}_{1}, \mathbf{a}_{1}, \cdots, \mathbf{s}_{T}, \mathbf{a}_{T}) \sim p_{\theta}} \left[\left(\sum_{t=1}^{T} \log \frac{\pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{t})}{\pi_{r e f}(\mathbf{a}_{t} \mid \mathbf{s}_{t})}\right) \left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{t})\right)\right]
```

记

```math
k(s_t, a_t) = \log \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{r e f}(a_t \mid s_t)}
```

则同样地，根据 Markov 性质，$`t' < t`$ 时的 $`\sum_{t'=1}^{t-1} k\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right)= \sum_{t'=1}^{t-1} \log \frac{\pi_{\theta}(a_{t'} \mid s_{t'})}{\pi_{r e f}(a_{t'} \mid s_{t'})}`$ 与 $`t`$ 时的策略梯度 $`\nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{t})`$ 是独立的。

所以有：

```math
\mathbb{E}_{\mathbf{\tau} \sim p_\theta}\left[\nabla_\theta \log \pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right) \sum_{t^{\prime}=1}^{t-1} k\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)\right] = 0
```

代入 KL 梯度表达式，即可简化得到：

```math
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] =  \mathbb{E}_{\mathbf{\tau} \sim p_\theta}\left[\sum_{t=1}^{T} \left(\sum_{t'=t}^{T} k\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right) \right) \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{t}) \right]
```

对应的 Monte Carlo 估计式为：

```math
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \left(\sum_{t'=t}^{T} k\left(s_{i, t'}, a_{i, t'}\right) \right) \nabla_{\theta} \log \pi_{\theta}(a_{i, t} \mid s_{i, t})
```

此处，$`k\left(s_{i, t'}, a_{i, t'}\right) = \log \frac{\pi_{\theta}(a_{i, t'} \mid s_{i, t'})}{\pi_{r e f}(a_{i, t'} \mid s_{i, t'})}`$。

不难注意到 KL 估计样本值 $`k`$ 与 reward $`r`$ 在形式上的相似性，这也解释了为什么先前的工作要将 KL 放进 reward。但两者不同的是 $`k(s_{i, t'}, a_{i, t'})= \log \frac{\pi_{\theta}(a_{i, t'} \mid s_{i, t'})}{\pi_{r e f}(a_{i, t'} \mid s_{i, t'})}`$ 会随 $\pi_\theta$ 变化而变化，而 $`r(s_{i, t'}, a_{i, t'})`$ 不会。

类似地，我们可以利用 PG 的其他技巧，进一步减小该估计的方差，例如减去 baseline 等，具体可以参考 [UCB CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)。

注意，数学表达式中，各个值默认是不计算梯度的，只有 $\nabla_{\theta}$ 后的部分才计算梯度。所以不在 $\nabla_{\theta}$ 后的部分，但又与 $\theta$ 相关的值，需要使用 `torch.no_grad()` 显式地设置为不计算梯度。

同样，要使用自动微分计算该梯度估计式，我们需要构造对应的 loss 函数：

```math
\mathcal{L}^{KL}_{\theta} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \text{nograd}\left (\sum_{t'=t}^{T} k\left(s_{i, t'}, a_{i, t'}\right) \right) \log \pi_{\theta}(a_{i, t} \mid s_{i, t})
```

### 利用重要性采样处理 off-policy 场景

off-policy 场景下，我们无法使用样本 $`\mathbf{\tau} \sim p_{\theta}`$，而只能使用样本 $`\mathbf{\tau} \sim p_{\theta_{old}}`$。

利用重要性采样，前面的 KL 梯度表达式可以转化为：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & =  \mathbb{E}_{\mathbf{\tau} \sim p_{\theta}}\left[\sum_{t=1}^{T} \left(\sum_{t'=t}^{T} k\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right) \right) \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{t}) \right] \\
& =  \mathbb{E}_{\mathbf{\tau} \sim p_{\theta_{old}}}\left[ \frac{p_{\theta}(\mathbf{s}_{1}, \mathbf{a}_{1}, \cdots, \mathbf{s}_{T}, \mathbf{a}_{T})}{p_{\theta_{old}}(\mathbf{s}_{1}, \mathbf{a}_{1}, \cdots, \mathbf{s}_{T}, \mathbf{a}_{T})} \sum_{t=1}^{T} \left(\sum_{t'=t}^{T} k\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right) \right) \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{t}) \right] \\
& =  \mathbb{E}_{\mathbf{\tau} \sim p_{\theta_{old}}}\left[ \left(\prod_{t=1}^{T}\frac{\pi_{\theta}(\mathbf{a}_{t} | \mathbf{s}_{t})}{ \pi_{\theta_{old}}(\mathbf{a}_{t} | \mathbf{s}_{t})}\right) \sum_{t=1}^{T} \left(\sum_{t'=t}^{T} k\left(\mathbf{s}_{t'}, \mathbf{a}_{t'}\right) \right) \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{t} \mid \mathbf{s}_{t}) \right]
\end{aligned}
```

对应的 Monte Carlo 估计式为：

```math
\begin{aligned}
\nabla_{\theta} \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right] & \approx \frac{1}{N} \sum_{i=1}^{N} \left(\prod_{t=1}^{T}\frac{\pi_{\theta}(\mathbf{a}_{i, t} | \mathbf{s}_{i, t})}{ \pi_{\theta_{old}}(\mathbf{a}_{i, t} | \mathbf{s}_{i, t})}\right) \sum_{t=1}^{T} \left(\sum_{t'=t}^{T} k\left(s_{i, t'}, a_{i, t'}\right) \right) \nabla_{\theta} \log \pi_{\theta}(a_{i, t} \mid s_{i, t}) \\
& = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \left(\left(\prod_{t=1}^{T}\frac{\pi_{\theta}(\mathbf{a}_{i, t} | \mathbf{s}_{i, t})}{ \pi_{\theta_{old}}(\mathbf{a}_{i, t} | \mathbf{s}_{i, t})}\right)\sum_{t'=t}^{T} k\left(s_{i, t'}, a_{i, t'}\right) \right) \nabla_{\theta} \log \pi_{\theta}(a_{i, t} \mid s_{i, t})
\end{aligned}
```

对应的 loss 函数为：

```math
\mathcal{L}^{KL}_{\theta} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \text{nograd}\left(\left(\prod_{t=1}^{T}\frac{\pi_{\theta}(\mathbf{a}_{i, t} | \mathbf{s}_{i, t})}{ \pi_{\theta_{old}}(\mathbf{a}_{i, t} | \mathbf{s}_{i, t})}\right)\sum_{t'=t}^{T} k\left(s_{i, t'}, a_{i, t'}\right) \right) \log \pi_{\theta}(a_{i, t} \mid s_{i, t})
```

注意，其中

```math
k\left(s_{i, t'}, a_{i, t'}\right) = \log \frac{\pi_{\theta}(a_{i, t'} \mid s_{i, t'})}{\pi_{r e f}(a_{i, t'} \mid s_{i, t'})}
```

仍然需要通过 $`\pi_{\theta}`$ 计算。

### 小结：KL 梯度估计的正确实现

KL 梯度估计的核心问题在于：究竟对什么量计算梯度。这里需要注意的是，并非所有与 $\theta$ 相关的量，都需要计算梯度。

1. 公式中的 nograd 操作，该操作在 PyTorch 中可以通过 `torch.no_grad()` 实现。将 KL 放入 reward 中，通常会自然地实现这一点，即不带梯度计算（当然，也可以实现为 loss 形式，但需要注意手动 nograd）。
2. 在 off-policy 场景下，则还需要注意：从第二个 mini-batch 开始，$\pi_\theta != \pi_{\theta_{old}}$
   1. 添加重要性采样系数 $\frac{p_{\theta}(\mathbf{s}_{1}, \mathbf{a}_{1}, \cdots, \mathbf{s}_{T}, \mathbf{a}_{T})}{p_{\theta_{old}}(\mathbf{s}_{1}, \mathbf{a}_{1}, \cdots, \mathbf{s}_{T}, \mathbf{a}_{T})}=\prod_{t=1}^{T}\frac{\pi_{\theta}(\mathbf{s}_{t}, \mathbf{a}_{t})}{ \pi_{\theta_{old}}(\mathbf{s}_{t}, \mathbf{a}_{t})}$。将 KL 估计样本值放入 reward 中，并重新进行重要性采样，则可以自然地实现这一点。
   2. 使用当前策略 $`\pi_{\theta}`$ 重新计算 $`k(s_{i, t'}, a_{i, t'})=\log \frac{\pi_{\theta}(a_{i, t'} \mid s_{i, t'})}{\pi_{r e f}(a_{i, t'} \mid s_{i, t'})}`$。将 KL 放入 reward 中时，很容易忽略这一点。

综上所述，基于目前主流 LLM RL 框架中的实现，最简单的修正方式应当是，

1. 保持将 KL 估计值
2. 每轮更新时，使用当前策略 $`\pi_{\theta}`$ 重新计算 $`k(s_{i, t'}, a_{i, t'})=\log \frac{\pi_{\theta}(a_{i, t'} \mid s_{i, t'})}{\pi_{r e f}(a_{i, t'} \mid s_{i, t'})}`$ 。

### 替换 k 是否对 KL 梯度估计同样有效？（TODO）

上文的推导是从定义出发的，$k$ 的形式与 k1 一致。如果将 $k$ 替换为 k2，k3 或其他估计样本值，是否能更准确地估计 KL 梯度？

### KL-regularized RL 的理论优势（TODO）

Wei Xiong et al. 证明了 KL-regularized RL 的 regret 只有 $\mathcal{O}(\log T)$。

## 致谢

感谢生广明 @PeterSH6、Wei Xiong @WeiXiongUST、 Weixun Wang、Yiming Liu、Haibin Lin @eric-haibin-lin 等的有益讨论。

感谢 Cursor 和 Mathpix 在书写 LaTeX 时发挥的巨大作用。

## 参考

https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-5.pdf
