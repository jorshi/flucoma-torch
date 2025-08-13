# flucoma-torch

Moel training and hyperparameter searches for FluCoMa MLPs in PyTorch.

## Getting Started

Recommend making a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install `flucoma-torch`:
```bash
pip install flucoma-torch
```

Export your datasets / labelsets from FluCoMA.

Train a regressor:
```
ft-train-regressor source=source-data.json target=target-data.json
```

Train a classifier:
```
ft-train-classifier source=source-data.json target=target-labels.json
```

Search for hyperparameters for a regressor:
```
ft-optimize-regressor source=source-data.json target=target-data.json
```

Search for hyperparmaters for a classifier:
```
ft-optimize-classifier source=source-data.json target=target-labels.json
```

Load the resulting `model.json` / `best_model.json` and associated scaler files back into FluCoMa.


## Training FluCoMa MLPs
`flucoma-torch` comes with command line functions for training MLPRegressors and MLPClassifiers. These scripts read json files written by FluCoMa dataset and labelset objects.

**Regressor training**:
```
ft-train-regressor source=source-data.json target=target-data.json
```

Where `source-data.json` is the input data to the model and `target-data.json` is the ground truth output data.

**Classifier training**:
```
ft-train-classifier source=source-data.json target=target-labels.json
```

Where `source-data.json` is the input data to the model and `target-labels.json` is the ground truth data from a FluCoMa label set.

### Outputs

All trained models and logs will be stored in an output directory, for example:
```bash
outputs/train_classifier/<date>/<time>
```
The `model.json` file can be loaded directly into either an MLPRegressor or MLPClassifier, depending on what you trained. (e.g., a read message into the max/msp `fluid.mlpclassifier~` object)

### Data Scaling

Data scaling may also have been applied to your data, the source and target (for regressors) scaling configs will also be stored in this folder and can be read directly into the corresponding FluCoMa object. The scaling method is written into the filename:

- `_normalize` -> `fluid.normalize~`
- `_standardize` -> `fluid.standardize~`
- `_robustscale` -> `fluid.robustscale~`

To select which scaling method to use during training see command line args below.

### Arguments

See [FluCoMa Neural Network Parameters](https://learn.flucoma.org/learn/mlp-parameters/) for more info on the mlp parameters.

| Argument              | Value Type                           |
|-----------------------|--------------------------------------|
| mlp.activation        | int {0, 1, 2, 3}                     |
| mlp.batch_size        | int                                  |
| mlp.hidden_layers     | list (in quotes), e.g. "[2,2]"       |
| mlp.learn_rate        | float                                |
| mlp.max_iter          | int                                  |
| mlp.momentum          | float [0.0,1.0]                      |
| mlp.output_activation | int {0, 1, 2, 3}                     |
| mlp.validation        | float [0.0,1.0]                      |
| scaler                | normalize, robustscaler, standardize |

If you do not want data scaling at all pass in the argument `"~scaler"` with the quotes. 

**Activations:**

The acivation types are:

- 0: identity
- 1: sigmoid
- 2: relu
- 3: tanh

Sigmoid is always used as the `output_activation` for classifiers and cannot be changed.

**Validation:**

When `mlp.validation > 0.0` this ratio of the dataset will be randomly selected and withheld for validation. Early stopping will be applied after 20 epochs of no improvement on the validation set.

**Example Usage:**
```
ft-train-classifier source=source-data.json target=target-data.json mlp.activation=2 mlp.hidden_layers="[2,4,2]" scaler=standarize
```

## Hyperparameter Searches

Don't know what MLP parameters to use? Hyperparameter searches can be performed using [optuna](https://optuna.org/). This command line functions work the same as the training scripts, but have a few more arguments and will train multiple models with different parameters to find what works the best for your data.

**Classifier:**
```
ft-optimize-classifier source=source-data.json target=target-labels.json
```

**Regressor:**
```
ft-optimize-regressor source=source-data.json target=target-data.json
```

### Outputs
All trained models and logs will be stored in an output directory, for example:
```bash
outputs/optimize_classifier/<date>/<time>
```

The best model found during the search is saved in `best_model.json`, and can be loaded in FluCoMa. 

The best resulting MLP parameters are logged in `best_hyperparameters.json`.

### Arguments
This script adds additional hyperparameters in addition to those available for regular training. **Note:** all mlp parameters (except for `max_iter` and `validation`) will be overwritten during the search.

| Argument         | Value Type      | Description                                                                    |
|------------------|-----------------|--------------------------------------------------------------------------------|
| mlp.max_iter     | int             | Number of epochs to train for each trial                                       |
| mlp.validation   | float [0.0,1.0] | Validation dataset size (ratio), will perform early stopping if > 0.0          |
| n_trials         | int             | Number of search trials to run                                                 |
| n_startup_trials | int             | Number of trials to perform before starting to prune                           |
| n_warmup_steps   | int             | Number of optimization steps before a trial will be considered for pruning     |
| study_name       | str             | Rename the study (how it's labeled if using sqlite)                            |
| sqlite           | bool            | Save the study data in a sqlite database (can be viewed with optuna-dashboard) |
| storage_name     | str             | Storage name, name of sqlite database                                          |

## Train your own model

You can also write your own training routines in python using the FluidMLP class, which you can then export to a json file. For regressors this json can be loaded directly.

```python
from flucoma_torch.model import FluidMLP

mlp = FluidMLP(
    input_size=2,
    hidden_layers=[2,4,2],
    output_size=2,
    activation=2,
    output_activation=2,
)

# ... train the model

mlp.save("model.json")
```

For classifiers you also need labels. For example, training a classifier with data scaling:

```python
import json
from flucoma_torch.data import load_classifier_dateset
from flucoma_torch.model import FluidMLP
from flucoma_torch.scaler import FluidNormalize

normalizer = FluidNormalize()

# This returns a torch Dataset, the scaler fit to the data, and the classification labels
dataset, fit_normalizer, labels = load_classifier_dateset(
    "source-data.json",
    "target-labels.json",
    normalizer
)

# Get the first item to see input/output shape
x, y = dataset[0]

mlp = FluidMLP(
    input_size=x.shape[0],
    hidden_layers=[2,4,2],
    output_size=y.shape[0],
    activation=2,
    output_activation=1, # sigmoid output
)

# ... train the model

# Save model with labels
classifier_dict = {
    "labels": {
        "labels": labels,
        "rows": len(labels)
    }
    "mlp": mlp.get_as_dict()
}
with open("model.json", "w") as fp:
    json.dump(classifier_dict, fp)

# Save normalizer
fit_normalizer.save("normalizer.json")
```
