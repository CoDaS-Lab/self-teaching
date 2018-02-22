# Notes on self-teaching

## Teaching model

In the teaching model, the teacher presents pairs of feature and label pairs to the learner using the following set of recursive equations:

$$P_L(h|x, y) \propto P_T(x, y|h)P_L(h)$$

$$P_T(x, y|h) \propto P_L(h|x, y)P_T(x, y)$$

We assume that the teacher's prior over feature and label pairs $P_T(x, y)$ is uniform, and run these equations until they converge. Once this is reached, the teacher selects data points according to the following equation:

$$P_L(h^*|x, y) \propto P_T(x, y|h^*)P_L(h)$$

### Direct instruction model

The direct instruction model is a variant of the teaching model where the teacher provides only features to a learner, and not any labels. This is the version of the teaching model that is used in the simulations for our paper. This results in a slight modification of the teaching equations as follows:

$$P_L(h|x, y) \propto P(y|x, h)P_T(x|h)P_L(h)$$

Now the teacher only selects the feature $x$ to teach, and $y$ is observed from the environment instead. Note: this equation doesn't look correct to me as the right term when simplified is equal to $P(h, x)$

$$P_T(x|h) \propto \sum_y P_L(h|x, y) P_T(x, y)$$

In the code, the learner's posterior is updated in the following manner:

$$P_L(h|x, y) = \frac{P(y|x, h)P_T(x|h)P_L(h)}{\sum_h' P(y|x, h')P_T(x|h')P_L(h')}$$ 

Where $P_L(h)$ is initialized to be $1/|h|$ and $P_T(x|h)$ is initialized to be $1/|x||y|$.

Then, the teacher's posterior is updated in the following manner:

$$P(x, y) = \frac{1}{|x||y|}$$

$$P(h, x, y) = P_L(h|x, y)P(x, y)$$

$$P(h, x) = \sum_y P(h, x, y)$$

$$P_T(x|h) = \frac{P(x, h)}{P(h)} = \frac{P(x, h)}{\sum_x P(x, h)}$$

Is there a simpler way to calculate the teacher's posterior in the direct instruction model? One possibility is to ignore the prior in the teaching equation, and use the learner's posterior directly as done in Shafto and Goodman (2008) by setting:

$$P_T(x|h) \propto (P_L(h|x, y))^\alpha$$

And then setting $\alpha = 1$. Using this equation, we can calculate the exact teaching probability as follows:

$$P_T(x|h) = \frac{P_L(h|x, y)}{\sum_x' P_L(h|x', y)}$$

TODO: double check this is correct!

## Self-teaching model

The self-teaching model can be described by first showing how direct instruction can be rewritten:

$$P_L(h|x, y) \propto P(y|x, h)P_T(x|h)P(h)$$

$$P_T(x|h') \propto \sum_y P_L(h'|x, y) P_T(x, y)$$

$$P_T(x|h) = \sum_{h'} P_T(x|h') \delta(h'|h)$$

where $h$ is the true hypothesis, and $h'$ is the learner's guess of the true hypothesis. In these equations, $\delta(h|h')$ refers to the knowledge of the teacher, which we can modify to become $P_L(h')$, which is now independent of the true hypothesis $h$. Thus, the self-teaching model can be written as:

$$P_L(h|x, y) \propto P(y|x, h)P(h)$$ and $P_T(x|h)$ 

as the term $P_T(x)$ occurs in both the numerator and denominator and does not depend on $h'$, it cancels out. 

The other two equations calculate the self-teaching probability for a particular hypothesis $h'$ which is denoted as $P_T(x|h')$, and the self-teaching probability $P_T(x)$ which marginalizes over $h'$

$$P_T(x|h') \propto \sum_y (h'|x, y) P_T(x, y)$$

$$P_T(x) = \sum_h' P_T(x|h') P_L(h')$$


### Self-teaching code

The likelihood of observing a particular label is given by:

$$P(y|x, h') = \begin{cases}
0 \quad \text{if } h(x) = 0 \\
1 \quad \text{if } h(x) = 1
\end{cases}$$

To update the learner's posterior, we run update_learner_posterior which calculates:

$$P_L(h|x, y) = \frac{P(y|x, h) P_T(x|h)}{\sum_h P(y|x, h') P_T(x|h')}$$

To update the self-teaching posterior, we run update_self_teaching_posterior which calculates:

prob_joint_data: $P(x, y) = \frac{1}{|x||y|} \quad \forall x, y$

prob_joint: $P(h', x, y) = P(h'|x, y)P(x, y)$

prob_joint_hyp_features: $P(h', x) = \sum_y P(h', x, y)$

prob_conditional_features: $P(x|h') = \frac{P(h', x)}{P(h')} = \frac{P(h', x)}{\sum_x P(h', x)}$

self_teaching_posterior: $P(x) \propto \sum_{h'} \sum_y P(x|h) P(h'|x, y)$ 

Right now the self-teaching posterior code uses $P(h'|x, y)$ at the end, but according to the equations this should be replaced with the prior $P(h')$ instead. Discussion with Scott resolved this, it should use the prior instead and when done so it shows that the beahviour of the self-teaching model acts like an active learner instead. 

In order to learn, the self-teacher uses their own self-teaching posterior to sample a feature $x^*$ to observe, as described in sample_self_teaching_posterior, where:

$$x^* \sim P_T(x)$$

$$y = h(x^*)$$

Then, the learner's posterior is updated using the above equation.

## Batch self-teaching model

In the batch version of the self-teaching model, rather than evaluating which features $x$ to self-teach and select by simulating one-step ahead, the model instead evaluates a set of features $x_1, \dots, x_n$ and corresponding features $y_1, \dots, y_n$ using the same equations as above in the self-teaching model.

However, once we have obtained a self-teaching posterior, instead of it looking having the form $P_T(x)$, it is of the form $P_T(x_1, \dots, x_n)$. To recover the probability of teaching each individual feature, we sum across all sets of teaching points that contain that feature and divide by the total number of features as follows:

$$ P_T(x) = \sum_{x \in (x_1, \dots, x_n)} \frac{P_T(x_1, \dots, x_n)}{n}$$

## Incremental self-teaching model
In the incremental self-teaching model, we extend the self-teaching model

## Teaching example: 3 features and 2 data points

Here, I consider what it is like to teach in the boundary game with the following hypotheses:

$$\mathcal{H} = \{111, 011, 001, 000\}$$

### Batch self-teaching

The batch self-teacher selects two different values of $\mathbf{x}$ from the set $\{01, 02, 12\}$, with each of the hypotheses having the possible sets of $\mathbf{y}$ values: $\{00, 01, 10, 11\}$.

We assume a uniform prior over hypotheses and teaching sets:

$$P(h) = 1/4 \quad \forall h$$

$$P(\mathbf{x}) = 1/3 \quad \forall \mathbf{x}$$

The likelihood $P(\mathbf{y}|\mathbf{x}, h)$ is the following:
