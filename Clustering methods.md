# Clustering methods
## K-means
...
##  Affinity propagation
Affinity propagation (AP) is a clustering algorithm based on the concept of "message passing" between data points. It does not require the number of clusters to be determined or estimated before running the algorithm. 

The cluster centers are the original data points in dataset (exemplars). AP should find the exemplars and decide other data points belong to which exemplar.

### Some important definitions
**Exemplars:** A subset of input dataset, representative of clusters (centers of cluster.)

**Similarity Matrix($S$):** $s(i,k) i \neq k$ indicates how well the data point with index $k$ is suited to be the exemplar for data point $i$. If the input is feature vector, then the similarity between points is the negative squared euclidean distance. In our task, I use the negative ...

**Preference:** The diagnal elements $s(k,k)$ in $S$ has another name: preference. It indicates how well $x_k$ chosen as exemplar. Larger values of s(k,k), data point $k$ is more likely to be chosen as exemplars. If it is not provided, all $s(k,k)$ shared the median or the minimum of the input similarities. Higher shared preference value result in more clusters.

**Responsibility ($R$):** 
The "responsibility" matrix R has values $r(i, k)$ that quantify how well-suited $x_k$ is to serve as the exemplar for $x_i$, relative to other candidate exemplars for $x_i$.

**Availability ($A$):**
The "availability" matrix $A$ contains values $a(i, k)$ that represent how "appropriate" it would be for $x_i$ to pick $x_k$ as its exemplar, taking into account other points' preference for $x_k$ as an exemplar.

**Converge** 
### Algorithm
* $A$ and $R$ initialize to zero

* While not converge 
   1. $r(i, k) \leftarrow s(i, k) - max [ a(i, \acute{k}) + s(i, \acute{k}) \forall \acute{k} \neq k ]$
   2. $a(i, k) \leftarrow min [0, r(k, k) + \sum_{\acute{i}~s.t.~\acute{i} \notin \{i, k\}}{r(\acute{i}, k)}]$

### Explanation of algorithm
We can treat this process as an election. 
All the people participate in the election to select some representatives. All people are both voters and candidates. 

s(i, k) is the prior preference for $x_i$ to choose $x_k$ as his representative. 

r(i, k) indicates the advantage of $x_k$ compared with other candicates on the side of voter $x_i$. It is computed by $s(i,k)$ minus the score of the strongest competitor on the side of $x_i$. The score for other candicates $x_\acute{k}$ is $a(i, \acute{k}) + s(i, \acute{k})$.

The update process for r(i, k) corresponds to the selection of the candidates by $x_i$. The most attractive one will be selected in the end.

$a(i, k)$: from the formula, we can find all values ​​of $r(\acute{i}, k)> 0$ have positive influence to $a(i, k)$. Just like the online poll during the election. If I（$x_i$） heared others($x_\acute{i}$) say a candicate($x_k$) is good ($r (i', k)> 0$), I'll have a good impression and have a high probability vote to him. 

The update process of $a(i, k)$ corresponds to the influence of online polls on candidate $x_k$ for $x_i$. Candicate who have more followers will have more probability to be chosen as a representive (exemplar)

The loop process of alternating update $R$ and $A$ just like a lot of round of elections. The election results will be converged after some rounds. 

$r(i, k)$ reflects the competition, $a(i, k)$ is to make the cluster more intense.
 

## Spectral clustering
Spectral clustering treats the data clustering as a graph partitioning problem. This note is based on the [Introduction to spectral clustering](http://lagis-vi.univ-lille1.fr/~lm/classpec/reunion_28_02_08/Introduction_to_spectral_clustering.pdf) and [A Tutorial on Spectral Clustering](http://engr.case.edu/ray_soumya/mlrg/Luxburg07_tutorial_spectral_clustering.pdf).

 
### Similarity matrix $W$
The dataset is $X= {x_1, …, x_i, …, x_n}$. For each pair of $(x_i, x_j)$, a similarity value should be computed. Then a similarity matrix $W \in R^{n*n}$ is constructed. There are different ways to compute this similarity value:

1. Cosine (our task and in most natural language processing tasks)
2. Fuzzy 
3. Gaussian types
4. Output of neural networks(Our method)

Gaussian types is much more used in clustering approaches
and is defined by:
$$...$$

There are also different ways to construct a graph representing
the relationships between data points:

1. Fully connected graph: All vertices having non-null similarities are connected each other. 
2. r-neighborhood graph: Each vertex is connected to vertices falling inside a ball of radius r where r is a real value that has to be tuned in order to catch the local structure of data. 
3. k-nearest neighbor graph: Each vertex is connected to its k-nearest
neighbors where k is an integer number which controls the local relationships of data. 
4. r-neighborhood and k-nearest neighbor combined.

### Some important definition in Graph Cut
#### A overlook of graph cut
A graph is $G(V, E)$, $V$ is the set of vertices and $E$ is the set of edges. If we want to split a graph in two parts $A$ and $B$ (graph partition), then 
$$A \cup B = V$$
$$A \cap B = \Phi $$
It is easy to expand to $K$ parts:
$$S_1 \cup ... \cup S_k \cup ... \cup S_K = V$$
$$S_1 \cap ... \cap S_k \cap ... \cap S_K = \Phi $$

#### Degree matrix $D$
We define the degree $d_i$ of a vertex $i$ as the sum of edges weights incident to it: 
$$d_i=\sum_{j=1}^{n} w_{i,j}$$
The degree matrix ($D$) of the graph $G$ denoted by will be a diagonal matrix having the elements on its diagonal and the off-diagonal elements having value 0. 

#### Weights connections between clusters
1. The sum of weight connections between two clusters:$\mathrm{Cut}(A,B)=\sum_{i \in A, j \in B} w_{i,j}$
2. The sum of weight connections within cluster A: $\mathrm{Cut}(A,A)=\sum_{i \in A, j \in A} w_{i,j}$
3. The total weights of edges originating from cluster A: $\mathrm{Vol}(A)=\sum_{i \in A} d_{i}$

#### Objective function
Given a similarity graph with adjacency matrix $W$, the simplest and most direct way to construct a partition of the graph is to solve the mincut problem. To define it, please recall the notation
$W(A, B)=\sum_{i \in A, j \in B} w_{i,j}$
and $\bar{A}$ for the complement of A. For a given number $k$ of subsets, the mincut approach simply consists in choosing a partition $A_1,..., A_k$ which minimizes:
$$\mathrm{cut}(A_1,...,A_k)=\frac{1}{2}\sum_{i=1}^{k}W(A_i,\bar{A_i})$$
In the real implementation of $k=2$, the solution of mincut simply separates one individual vertex from the rest of the graph. Normaly, clusters should be reasonably large groups of points. So there are two other objective function:
$$\mathrm{RatioCut}(A_1,...,A_k)=\frac{1}{2}\sum_{i=1}^{k} \frac{W(A_i,\bar{A_i})}{\left | A_i \right |}=\sum_{i=1}^{k} \frac{\mathrm{cut}(A_i,\bar{A_i} )}{\left | A_i \right |}$$

$\left | A_i \right |$ is the number of vertices in subgraph $A$.
$$\mathrm{Ncut}(A_1,...,A_k)=\frac{1}{2}\sum_{i=1}^{k} \frac{W(A_i,\bar{A_i})}{\mathrm{vol}(A_i)}=\sum_{i=1}^{k} \frac{\mathrm{cut}(A_i,\bar{A_i} )}{\mathrm{vol}(A_i)}$$
$\mathrm{Vol}(A)$ is the total weights of edges originating from cluster A: $\mathrm{Vol}(A)=\sum_{i \in A} d_{i}$


### Laplacian matrix in spectral clustering
Laplacian matrix is a matrix representation of a graph. The unnormalized graph Laplacian matrix is defined as：
$$L=D-W$$
where $D$ is the degree matrix and $W$ is the similarity matrix. Laplacian matrix has some important properties:

1. For every vector $f \in \mathbb{R}^{n}$ we have
$$f'Lf=\frac{1}{2}\sum_{i,j=1}^{n}w_{i,j}(f_i-f_j)^{2}$$
2. $L$ is symmetric and positive semi-definite.
3. The smallest eigenvalue of $L$ is $0$, the corresponding eigenvector is the constant one vector $1$.
4. L has n non-negative, real-valued eigenvalues $0 = \lambda_1 ≤ \lambda_2 ≤ . . . ≤ \lambda_n$.

There are two matrices which are called normalized graph Laplacians in the literature. Both matrices are closely related to each other and are defined as：
$$L_{\mathrm{sym}}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}=I-D^{-\frac{1}{2}}WD^{-\frac{1}{2}}\\
L_{\mathrm{rw}}=D^{-1}L=I-D^{-1}W
$$


Their properties are similar to unnormalized graph Laplacian matrix. 

### Algorithm
Input: Similarity matrix $S \in R^{n×n}$, number $k$ of clusters to construct.

* Construct a similarity graph. Let $W$ be its weighted adjacency matrix.
* Compute the unnormalized Laplacian $L$ or $L_{\mathrm{sym}}$ or $L_{\mathrm{rw}}$.
* Compute the first k eigenvectors $u_1, . . . , u_k$ of $L$
* Let $U \in R^{n×k}$ be the matrix containing the vectors $u_1, . . . , u_k$ as columns.
* For $i = 1, . . . , n$, let $y_i ∈ R^k$ be the vector corresponding to the $i$-th row of $U$.
* Cluster the points $(y_i)_{i=1,...,n}$ in $R^k$ with the $k$-means algorithm into clusters $C_1, . . . , C_k$.

Output: Clusters $A_1, . . . , A_k$ with $A_i = \{j| y_j \in C_i\}$.

### Conclusion
Spectral clustering transform clustering problem to a graph cut task, and solve it by eigenvector factorization on Laplacien matrix. Random walk clustering can also treated as a special case. The algorithm is simple, there're also some prove and deduce for this algorithm in [A Tutorial on Spectral Clustering](http://engr.case.edu/ray_soumya/mlrg/Luxburg07_tutorial_spectral_clustering.pdf). Matrix factorization is a important tool for clustering. Maybe I can spend some times on it. 


## Label propagation
This note is a summary of [semi-supervised](https://en.wikipedia.org/wiki/Semi-supervised_learning) and [learning from labeled and unlabeled data with label probagaion](http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf).

Label propagation is a semi-supervised learning algorithm. In a dataset for semi-supervised learning, there is a small amount of labeled data with a large amount of unlabeled data. Semi-supervised learning algorithm can make advantage of unlabeled data to explore the dsitribution of all the dataset. Semi-supervised learning algorithms make use of at least one of the following assumptions:

* **Smoothness assumption:** Points which are close to each other are more likely to share a label.
* **Cluster assumption:** The data tend to form discrete clusters, and points in the same cluster are more likely to share a label.
* **Manifold assumption:** The data lie approximately on a manifold of much lower dimension than the input space. In this case we can attempt to learn the manifold using both the labeled and unlabeled data to avoid the curse of dimensionality. Then learning can proceed using distances and densities defined on the manifold.

In fact, label probagation will label the unlabeled data based on a similarity matrix. Close data points have similar labels. So it has two important steps:

1. Compute the similarity matrix $W$ and transition matrix $T$
2. Do the propagation

### Similarity matrix $W$ and transition matrix $T$
In dataset, $(x_1, y_1)...(x_l,y_l)$ are labeled data, where $Y_L=\{ y_1,...,y_l\}$ are the class labels. We assume all classes are present in the labeled data. $(x_{l+1}, y_{l+1})...(x_{l+u}, y_{l+u})$ are unlabeled data where $Y_U = \{ y_{l+1},...,y_{l+u}\}$ are unobserved, usually $l<<u$. Now the problem is to estimate $Y_U$ from $X$ and $Y_L$. 

There are different ways to compute the similarity between two data points $(x_i, x_j)$. The most used is local Euclidean distance, which is controled by a parameter $\sigma$:
$$w_{i,j}=\mathrm{exp}(-\frac{d_{i,j}^{2}}{\sigma ^{2}})=\mathrm{exp}(-\frac{\sum_{d=1}^{D}(x_{i}^{d}-x_{j}^{d})^{2}}{\sigma ^{2}})$$
where $D$ is the dimension of $x$.

The probabilistic transition matrix $T \in R^{(l+u)*(l+u)}$ is defined as follows:
$$T_{i,j}=P(j \rightarrow i)=\frac{w_{i,j}}{\sum_{k=1}^{l+u}w_{k,j}}$$
where $T_{i,j}$ is the probability to jump from node j to i. 

The we define a $(l+u)*C$ label matrix $F$, the $i$th row represents the label probability distribution of node $x_i$. And $F$ can be split into 2 parts $F=[Y_{L};Y_{U}]$. $Y_L$ is fixed, and the initialition of $Y_U$ is not important.

### Algorithm
input is the label matrix $F$ and transition matrix $T$

* propagation: $F=TF$
* Reset the $F$ corresponding to the labeled data points: $F_L=Y_L$
* Repeat until converge

Step 2 is very important, we should prevent the initially labeled nodes fade away. 


#### Variant
In the step 2 of this algorithm, we should always reset $F_L$, but we just care about $Y_U$. The following algorithm can help us just compute $Y_U$. 

First we should split transition matrix $T$ as follows:
$$
\begin{bmatrix}
T _{LL}&T _{LU}  & \\ 
T _{UL} &T _{UU}  & 
\end{bmatrix}
$$

Then:
$$f_{U} \leftarrow T_{UU}f_{U}+T_{UL}Y_{L}$$

$f_{U}$ will converge at $(I-T_{UU}^ {-1})T_ {UL}Y_L$.

more information in [Learning from labeled and unlabeled data with label propagation](http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf) and [Semi-Supervised Learning with Graphs](http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf).