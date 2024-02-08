# Problem definition

Let $c_j$ be a cell, $\bar{G}$, the true graph of gene regulation and $m*m$ matrix of $m$ genes and $\bar{e}$ is the true cell expression such that $\exists \bar{G}, f$ such that $f: \bar{G} \rightarrow \bar{e}$.


let $\hat{e} \sim ZiNB({n},{p},{\gamma})$ be the recovered cell expression.

Given a $G_0 = 0^{m*m}$ , $\hat{e}$ we want to find an update scheme:
$G_{i+1} = G_i + \text{update}(\hat{e})$
s.t. we $\min \text{D}(G_i, \bar{G})$
(here $D$ itself is non trivial as some connections have more importance than others).

Since we don't know either $D$ nor $\bar{G}$ we settle for a proxy task: 
$\min_{f, G_i} \text{d}(\bar{e}, f(G_t))$

# Current approach

In our context we would want the update to be akin to a transformer neural network model:
simplified as: $MLP(softmax({Q^iK^{iT}\over{\sqrt{d}}})A^i)$

where: $Q^i = W_Q^iR^i$, $K^i = W_K^iR^i$, $A^i = W_A^iR^i$ d is a normalizing factor and $R^i \in \R^{d,m}$ a representation of the cell's expression $e$
and the $i$ represent the index of its different layers.

in this current approach [1] (sinkformer) shows that with a modification of the softmax() to be a row / column normalization. We can see this update as a gradient flow:
$\frac{dR_t}{dt} = -\nabla F(R_t)$

where $\frac{dR_t}{dt}$ is the rate of change of the point cloud with respect to time, and $-\nabla F(R_t)$ is the negative gradient of the potential function at $G_t$



# WIP

## TODO:
---

TODO: need to help say that G's update here is coupled to R's but only weekly and that the learning signal could be improved by doing something like finding E = SDE(G,R) making G strongly tied to the downstream expression

TODO: need to talk also about the graph laplacian and how this diffusivity helps define a similarity between graphs?

TODO: compare expressivity with MPNN

TODO: talk about the model's equivariance. since there is not position per se. the model is equivariant to edge's position (E3 class?)

TODO: talk about evolution of both nodes and edges and adjacencies

long term TODO: think about 
  - efficiency of the evolution 
  - about universality of the functions that are computed

---

## 14/12/23

we can see the update as a gradient flow:
$\frac{dG_t}{dt} = -\nabla F(G_t)$

where $\frac{dG_t}{dt}$ is the rate of change of the graph with respect to time, and $-\nabla F(G_t)$ is the negative gradient of the potential function at $G_t$

can be seen as an update $G_{t+1} = G_t - \epsilon \nabla E(G_t)$

where $E()$ is an energy function for the graph.

we now also update both:
$r_i \leftarrow r_i + \sum_{j}softmax(C_{i,j}+u_{i,j})W_Ar_j$

$u_{i,j} \leftarrow u_{i,j} + softmax(C_{i,j}+u_{i,j})$

where $C_{i,j} = r_i^TW_K^TW_Qr_j$ the attention of the model, which can be seen as the cost matrix between nodes $i$ and $j$ and $u_{i,j}$ are the edge weights/embeddings of the graph.

We would want to go in the same direction as what [1] (sinkformer) did but with this new view of a system of 2 equations.

---

$\min \sum_{i} || x_i - x'_{\sigma(j)} ||^2 + \sum_{i,j} || u_{i,j} - u'_{\sigma(i),\sigma(j)} ||^2$

$\min <C,P> + ||\bar{G}-PGP^T||^2$

P is the coupling matrix between the nodes of the graph and the nodes of the graph at the next time step, C is the cost matrix. (Kantorovich formulation)

Q: can we see what EGT does as solving an OT problem? where we don't know the cost matrix but we update the couplings across the layers?

Issue: can we make it work as a GW problem?


## 05/12/23
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
