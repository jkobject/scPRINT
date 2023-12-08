so we have a problem

$c_j$ is a cell

$\bar{G}$ is the true graph of gene regulation and $m*m$ matrix of $m$ genes

$\bar{e}$ is the true cell expression

$\hat{e} \sim ZiNB({n},{p},{\gamma})$ is the recovered cell expression

$\exists \bar{G}, f$ such that $f: \bar{G} \rightarrow \bar{e}$

given a $G_0 = 0^{m*m}$ , $\hat{e}$ we want to find an update scheme:

 $G_{i+1} = G_i + \text{update}(\hat{e})$

such that:

$\min \text{D}(G_i, \bar{G})$ here $D$ itself is non trivial as some connections have more importance than others. Since we don't know either $D$ nor $\bar{G}$ we settle for a proxy task:

$\min_{f, G_i} \text{d}(\bar{e}, f(G_t))$

we can also see the update as a gradient flow:

$\frac{dG_t}{dt} = -\nabla F(G_t)$

where $\frac{dG_t}{dt}$ is the rate of change of the graph with respect to time, and $-\nabla F(G_t)$ is the negative gradient of the potential function at $G_t$

in our context the update is a transformer neural network model and the $i$ represent the index of its different layers.

the update itself is a $MLP({Q^iK^{iT}\over{\sqrt{d}}})$

where: $Q^i = W_Q^iR^i$, $K^i = W_K^iR^i$, d is a normalizing factor and $R^i \in \R^{d,m}$ a representation of the cell's expression $e$

in this current approach we can see that

(WIP: need to help say that G's update here is coupled to R's but only weekly and that the learning signal could be improved by doing something like finding E = SDE(G,R)) making G strongly tied to the downstream expression


(TODO: need to talk also about the graph laplacian and how this diffusivity helps define a similarity between graphs?)

---

(WIP: here would want to say that in fine if we had a known Gt, Gt+1 we could learn this distance metric such that the distance is decreased between graphs leading to a similar expression output)

Let's just have a quick look at this unknown distance $D$ between graphs within the optimal transport framework:

Finding the right distance itself can be seen as a task that involves finding a transport plan P:

$\min_{P \in U(a,b)} \langle P, M_{XY} \rangle - \gamma E(P)$

$\arg\min_{P^*} \epsilon(P)$

with

$\epsilon(P) := \sum_{i,i',j,j'} P_{i,j} P_{i',j'} \left( C_1(x_i, x_{i'}) - C_2(y_j, y_{j'}) \right)^2$

giving an -entropy regularised- update rule such that

$P_{t+1} \leftarrow \arg\min_{P_{1m}=a, P_{1n}=b} \langle \nabla \epsilon(P_t), P \rangle - \gamma \langle E(P) \rangle$

with an approximatation of $\nabla \epsilon(P)$:


$\nabla \epsilon(P) \propto -C_1 P C_2$

(WIP: these equations are not finished and taken from other papers.)

given that each gene is represented with a vector that too will evolve through the network. we can represent it as the fused GW by adding a term:

$\sum_{i,j} P_{i,j} C_3(f_{x_i}, f_{y_j})$
