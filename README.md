# MATLAB implementation of the EARS algorithm

0. Corresponding author: Qi Zhang, Department of Applied Mathematics and Statistics, Stony Brook University, Stony Brook, NY 11794-3600

1. The EARS algorithm by Qi Zhang and Jiaqiao Hu [1] is implemented for solving single-objective box-constrained expensive deterministic optimization problems. Note that the implementation could be speed-up significantly by simplifying calculations and exploiting parallel computing. The current version aims at the readability.

2. EARS (enhancing annealing random search) is a random search algorithm for solving Lipschitz continuous optimization problems. The algorithm samples candidate from a parameterized probability distribution over the solution space and uses the previously sampled data to fit a surrogate model of the objective function. The surrogate model is then used to modify the parameterized distribution in a way that concentrates the search on the set of high quality solutions. We prove the global convergence of the algorithm and provide numerical examples to illustrate its performance.

3. In the implementation, the algorithm samples candidate solutions from a sequence of independent multivariate normal distributions that recursively  approximiates the corresponding Boltzmann distributions [2].

4. In the implementation, the surrogate model is constructed by the radial basis function (RBF) method [3].

### Reference:
1. Qi Zhang and Jiaqiao Hu (2019): Enhancing Random Search with Surrogate Models for Continuous Optimization. Proceedings of the IEEE 15th International Conference on Automation Science and Engineering, forthcoming.
2. Jiaqiao Hu and Ping Hu (2011): Annealing adaptive search, cross-entropy, and stochastic approximation in global optimization. Naval Research Logistics 58(5):457-477.
3. Gutmann HM (2001): A radial basis function method for global optimization. Journal of Global Optimization 19:201-227.
