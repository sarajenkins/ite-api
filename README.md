Tasks:
 - preprocess twins data: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/ganite/data_preprocessing_ganite.py
 - run GANITE and CMGP on Twins dataset
 - Create a unified API which allows users to at least train both algorithms with specified hyperparameters and contains predict and test methods. Predict methods should allow users to obtain potential outcomes for a new patient. Test method should let users evaluate the algorithm and compute appropriate error metrics ((e.g. precision in estimating heterogeneous effect (PEHE))https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/68e4f7d13e4368eba655132a73ff9f278da5d3af/alg/ganite/ganite.py)
 - 
3. Run the unified API on the dataset (train + evaluation)
4 [Bonus]. Perform hyper-parameter tuning 
5 [Bonus]. Implement GANITE in PyTorch or a newer version of TensorFlow.

Example of unified API: 
 Code:
https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/02edab3b2b6d635470fa80184bbfd03b8bf80
82d/app/clairvoyance/
 Paper: https://openreview.net/forum?id=xnC8YwKUE3k
Code for methods:
 GANITE: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/ganite/ 
 CMGP:
https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/causal_multitask_gaussian_proce
sses_ite/ 
Note: CMGP training data needs to be downsampled to have at most 1000 training examples. 
Data:
 Twins: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/data/twins/
Note: details for pre-processing the Twins dataset:
https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/ganite/data_preprocessing_ga
nite.py
All required code and data can also be found in the following GitHub repository:
https://github.com/ZhaozhiQIAN/technical_interview