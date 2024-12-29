# Regularization


## The choice of our model complexity will either:
$$
\Large \text{increase the bias $\longrightarrow$ decrease the variance}
$$

$$
\Large \text{increase the variance $\longrightarrow$ decrease the bias}
$$

<div align="center">

| Term | High | Low |
| ------| ------- | ---------|
|Variance| Risk of overfitting | Generalization |
|Bias| Risk of underfitting | Fitting |
|Noise| Challenge | Easy |

</div>

#### __Goal__: Find a model complexity that has both low bias (able to fit well) and low variance (able to generalize)

## Weight Decay L1/L2 Regularization

### Idea:

Add a penalty term to the emperical risk:

$$
\Large \text{$argmin_{w}[\frac{1}{N} \sum_{i=1}^{N} L(y_{i}, f(x_{i}; w))] + \alpha \Omega(w)$}
$$

The regularization penalizes high parameter values:

$$
\Large \text{L1: $\Omega(w) = \frac{1}{|w|} \sum_{i=1}^{|w|} |w_{i}|$}
$$

$$
\Large \text{L2: $\Omega(w) = \frac{1}{|w|} \sum_{i=1}^{|w|} w_{i}^2$}
$$