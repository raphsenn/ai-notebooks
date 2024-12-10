# Ensembles

## Motivation: Ensembles are aggregations of multiple base models

![image](../../images/ensembles.png)


# Variance Reduction

The Bias-Variance Decomposition:

# Variance Reduction

The Bias-Variance Decomposition:

$$
\mathbb{E}_{x, y, \mathcal{D}} \left[ (\hat{f}(x; \mathcal{D}) - y)^2 \right] =
\underbrace{\mathbb{E}_{x, \mathcal{D}} \left[ (\hat{f}(x; \mathcal{D}) - \bar{f}(x))^2 \right]}_{\text{Variance}}
+ \underbrace{\mathbb{E}_{x, y} \left[ (\bar{f}(x) - \bar{y}(x))^2 \right]}_{\text{Bias}^2}
+ \underbrace{\mathbb{E}_{x, y} \left[ (\bar{y}(x) - y)^2 \right]}_{\text{Noise}}
$$

where the expected prediction model \( \bar{f}(x) \) is defined as:

$$
\bar{f}(x) \coloneqq \mathbb{E}_{\mathcal{D} \sim P_N} \left[ \hat{f}(x; \mathcal{D}) \right] = \int_{\mathcal{D}} \hat{f}(x; \mathcal{D}) \, p(\mathcal{D}) \, d\mathcal{D}
$$

Therefore, variance is reduced if:

$$
\hat{f}(x; \mathcal{D}) \to \bar{f}(x), \, \forall \mathcal{D}
$$

# Ensemble of Multiple I.I.D. Training Sets

- Given multiple (i.i.d.) training sets \( \mathcal{D}^{(1)}, \dots, \mathcal{D}^{(K)} \), where:

$$
\mathcal{D}^{(k)} \in (\mathcal{X} \times \mathcal{Y})^N, \, \forall k \in \{1, \dots, K\}
$$

- Train one prediction model on each training set:

$$
\hat{f}^{(k)}(x) = \hat{f}(x, \theta^{(k)}), \, \text{s.t.} \, \theta^{(k)} = \arg \min_{\theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}\left(y_i^{(k)}, \hat{f}(x_i^{(k)}, \theta)\right)^2
$$

- Compute an ensemble model:

$$
\hat{f}(x) = \frac{1}{K} \sum_{k=1}^K \hat{f}^{(k)}(x)
$$

---

### Ensemble Reduces Variance

- Following the law of large numbers:

$$
\bar{z} = \frac{z_1 + z_2 + \dots + z_K}{K} \to \mathbb{E}[z] \, \text{as} \, K \to \infty
$$

- For the ensemble model:

$$
\left[ \hat{f}(x) = \frac{1}{K} \sum_{k=1}^K \hat{f}^{(k)}(x) \right] \to \left[ \bar{f}(x) \coloneqq \mathbb{E}_{\mathcal{D} \sim P_N} [\hat{f}(x; \mathcal{D})] \right]
$$

- Replacing the single model with an averaged ensemble reduces the variance:

$$
\mathbb{E}_{x, \mathcal{D}} \left[ (\hat{f}(x; \mathcal{D}) - \bar{f}(x))^2 \right] = \mathbb{E}_x \left[ (\hat{f}(x) - \bar{f}(x))^2 \right] \to 0 \, \text{as} \, K \to \infty
$$

