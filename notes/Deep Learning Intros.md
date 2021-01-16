## Deep Learning Intros

---

#### Expo Weighed Average:

$$
V_t=\beta V_{t-1}+(1-\beta)\theta_t
$$
​		where, $V_{t-1}$ denotes the value of result in stage[t-1]. $\theta_t$ is the current observation. With this recursive equation, it can be easily derived that:
$$
V_t=\beta V_{t-1}+(1-\beta)\theta_t=(1-\beta)\theta_t+\beta(1-\beta)\theta_{t-1}+\beta^2V_{t-2}=...
$$
​		$\beta$ is smaller than 1. This is a weighed (moving-average-like) average method.

​		**<u>Why do we use this?</u>** Moving average needs a queue to store all the data and perform average. In deep learning where massive data could occur, this will not be efficient. Therefore, we can simply use this expo weighed average, which only requires merely one storage unit.

#### Momentum

One method to keep track of the old gradients. (Assuming that the current direction should not drift too far.).

#### AdaGrad

Keeps track of the squared gradients.
$$
w = \frac{\partial f}{\partial x},v=w^2\\
S = S+v\\
x = x-w/\sqrt{S}
$$
Notice that: square operation. In the part where grads are big, corresponding term of x will be damped because of division, while smaller parts might experience acceleration. Making the ill-conditioned problem (Steep in some directions while flat in others.) not so outrageous.

#### RMSProp

Leaky AdaGrad, which means, the gradients which are too old will decay. (I think this is better, always making the new ones take over.) Using Expo weighed average for AdaGrad S.

#### Adam

One of the best default method for optimization. It has:

- Bias correction: preventing the gradients (or steps) from becoming too small when initialized with a small coefficient ($\beta$ is small) expo weighed average. Using self division.
- EWA for gradients: Momentum.
- EWA for squared gradients and decay mechanism: RMSProp-like.

Common in Practice!

---

### CNN



