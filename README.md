# jaxipm

We present our jaxipm work from our paper: [Scaling Nonlinear Optimization: Many Problems, One GPU](https://arxiv.org/abs/2606.26341)

# Installation



# Research FAQ's

## Magic Numbers

The only "magic numbers" we have in the results presented is in the batch size and number of optimization results expected for each problem (beyond the parameters for the optimization, which we borrow from IPOPT, and have proven robust in IPOPTs case). We chose the number of optimization results to be small enough such that it wouldnt take forever on our hardware (1x L40s GPU), and chose the batch sizes to be roughly 1/4 of this. We chose the batch size to be roughly 1/4 to demonstrate the necessity of the iteration-level batching, which has a more pronounced effect when optimizations have diverged a bit, which is more the case when we need to do a few resets along the way. In my experimentation the throughput results were'nt very sensitive to batch size, but feel free to experiment (just takes a while to compile and run - but I am working on the compilation side of things ;) )!

## Why is Throughput Higher than GPU-Accelerated Sequential MadNLP?

This may not be obvious to the outsider so I just wanted to take a moment to mention it here. MadNLP does a great job GPU accelerating both the evaluation of sparse derivatives which constitute the KKT matrices (via examodels), and in the solving of the sparse symmetric positive definite condensed KKT matrix reformulation via cuDSS. We also use cuDSS in this work, but we solve the symmetric indefinite KKT matrix formulation used by IPOPT. When MadNLP calls cuDSS, a given problem is not guaranteed to fully saturate the GPU - leaving performance on the table. However, when we batch solve our symmetric indefinite KKT matrix (the same form as IPOPT uses), we almost certainly fully saturate the GPU (and if we don't then we can simply increase the batch size until we do). This means we get more linear solves per second from the GPU than the non GPU-batched (but still GPU-accelerated) MadNLP implementation.

## Why Don't we use the MadNLP Positive Definite KKT Formulation?

For this work we decided to use cuDSS for our sparse linear solves, as it is the fastest currently available direct sparse set of linear solvers on GPU that exist today. Both the IPOPT and MadNLP KKT matrix formulations require inertia correction (perturb the diagonal such that we find directions to local minima instead of saddles and maxima). MadNLP does this by detecting failure of positive definite symmetric linear solves in cuDSS, and inertia correcting. This does not work in the batched case because a batch of linear solves on cuDSS is seen as a block diagonal set of linear systems, and if any one of them fails, the whole batch fails, without a way to detect which element of the batch failed. Therefore we use the cuDSS symmetric indefinite linear solver, which gives us explicit matrix inertias for each linear solve in the batch, and allows us to inertia correct them individually. That being SAID - I would like to implement something like BaSpaCho to enable the MadNLP KKT formulation in (GPU-)batch in the future!
