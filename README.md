# API for Training and Evaluation of ITE models
    Supported ITE models:
        - GANITE : https://openreview.net/forum?id=ByKWUeWA-
        - CMGP : https://arxiv.org/pdf/1704.02801.pdf

## Installation
```
pip install -r requirements.txt
```

## Example useage

```
from api.metrics import PEHE
from api import Model

# ganite = Model('GANITE', num_iterations, num_kk, _alpha, _mini_batch_size, int(_h_dim))
cmgp = Model('CMGP', dim=num_features)

# fit model
cmgp.fit(train_X, train_Y, train_T)

# predict on new patients
result = cmgp.predict(test_X)

# evaluate results with defined metric (here PEHE)
cmgp.test(test_X, test_T, PEHE)
```

## Outstanding Tasks
    - Implement GANITE in PyTorch or a newer version of TensorFlow.
    - `GPCoregionalizedRegression` "is a thin wrapper around the models.GP class with a set of sensible defaults" https://gpy.readthedocs.io/en/deploy/_modules/GPy/models/gp_coregionalized_regression.html . Using models.GP limits the amount of data can efficiently train on to 1000 samples. To train on a large sample size, implement `SVGPCoregionalizedRegression` which would inherit from models.SVGP https://gpy.readthedocs.io/en/deploy/_modules/GPy/core/svgp.html. 
    - Early stopping on GANITE hyperparameter tuning was performed by verying a set of static inputs for `num_iterations`. This is a naive approach and could be improved using a more advanced method e.g. https://gist.github.com/ryanpeach/9ef833745215499e77a2a92e71f89ce2
    