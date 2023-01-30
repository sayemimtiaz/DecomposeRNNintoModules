## RQ3: Replaceability of modules
This paper introduces a method for breaking down a trained RNN model into separate modules, one for each output class. The research question evaluated in this paper examines the efficacy of this decomposition technique and its applicability in various scenarios. This research question demonstrates how a faulty decomposed module can be replaced in different situations. Our evaluation centers on two types of replace: a) replacing the modules within the same input-output type (referred to as **intra-replace**), and b) replacing the modules from different types (referred to as **inter-replace**).

For every dataset, we train and decompose four models for each model type. We note the models with the lowest (denoted as **faulty model**) and highest accuracy (denoted as **best model**). In the first scenario, we replace every module of **faulty model** with one from **best model** in the hope of replacing the faulty logic in the **faulty model**. In the second scenario, we evaluate the same across different tasks using different datasets.

The following steps provide some examples for running experiments to answer this RQ. Please read the [RQ1 evaluation documentation](/tutorial/evaluate_rq1.md) before running the following experiments.

## General instructions
To evaluate the replacement scenarios, the below steps should be followed in general:
- In replace scenarios, two models are involved: *faulty model*, which is to be replaced with a modules from *best model*.
- Both the *faulty* and *best* models must be decomposed following the documentation for [RQ1](/tutorial/evaluate_rq1.md). In the *intra-replace* mode, replace is evaluated within the same architecture and dataset, and in the *inter-replace* mode, modules are replaced from different architectures and datasets. 
- Each architecture should be decomposed in a specific mode. For example:
  - **one-to-one**: It should be decomposed in *rolled* mode.
  - **many-to-one**: It should be decomposed in *unrolled* mode.
  - **one-to-many**: It should be decomposed in *unrolled* mode.
  - **many-to-many-same**: It should be decomposed in *unrolled* mode.
  - **many-to-many-different**: It should be decomposed in *rolled* mode.
In our experiment, we found those modes to be the best performing. Hence the replacement evaluation is done on modules decomposed in the best mode.

- Once the target model(s) is decomposed, to evaluate **intra-replace**, `compare_replace` script should be executed with this command: `python3 -m replace.intra.{X}.compare_replace` and, for **inter-replace**, run: `python3 -m replace.inter.{X}.compare_replace`, where *X refers to different input-output architectures, such as one_to_one, many_to_one, one_to_many, many_to_many_same, and many_to_many_different*. 
- Before running the `compare_replace` script, the following variables should be checked to ensure the intended model is being evaluated:
```
activation = '{X}' #X = {gru_models, lstm_models, vanilla_rnn_models, relu_models}
bestModelName = 'mybestmodel' #name of the best model
faultyModelName = 'myfaultymodel' #name of the faulty model
```

### Example 1: replace faulty parts of a model with four GRU stacked layers with a module of better performing model with one GRU layer within the same dataset (intra-replace many-to-one)

In our experiment on many-to-one intra-replace, we found that **model1** is better performing than **model4** for GRU. Hence, in this example, **model1** is the **best model**, and **model4** is the **faulty model**. To replace them as described, they must first be decomposed following the documentation provided for [RQ1](/tutorial/evaluate_rq1.md). Please decompose them in **unrolled** mode. Once decomposed, we have to ensure that the following variables point to `model1`, `model4`, and `gru_models` in this script: *replace/intra/many_to_one/compare_replace.py*:
```
activation = 'gru_models'
bestModelName = 'model1'
faultyModelName = 'model4'
```
If it doesn't, please change it to the one you are looking to evaluate intra-reuse on. Once this basic sanity checking is done, please follow the below steps to evaluate it:

1. Open the terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run the following:
```
python3 -m replace.intra.many_to_one.compare_replace
```
The script will attempt to read and write from the directory. Please make sure that it has such permissions. 

4. Depending on the task, it may run for a while. It will print step-by-step replace results, which will be saved as a `csv` file in the same directory with the name *result_gru_models.csv*. In every scenario, it will print composed module accuracy after replacement which can be compared with monolithic faulty model accuracy previously found. A sample output for the illustration is provided below:
```
After replacing 0 Accuracy: 0.9631490787269682
After replacing 1 Accuracy: 0.96214405360134
After replacing 2 Accuracy: 0.9634840871021776
After replacing 3 Accuracy: 0.9614740368509213
After replacing 4 Accuracy: 0.9651591289782244
After replacing 5 Accuracy: 0.9608040201005025
... ...
```
Here the accuracy ranges from 0 to 1.0.

### Example 2: replace a module between different tasks (inter-replace)
Please see the inter-reuse example described for [RQ2](/tutorial/evaluate_rq2.md) first, as both are similar in the experimental setup.
For this example, we want to replace modules of a faulty model from **many-to-one** with a module from the best model in **many-to-many-same**. Let's assume, in both cases, **model1_combined.h5** is the responsible party. In other words, **lstm_models/many_to_one/h5/model1_combined.h5** performs poorly for **many-to-one** and **lstm_models/many_to_many_same/h5/model1_combined.h5** performs best for **many-to-many-same**. 

Similar to inter-reuse, we must first decompose *model1_combined.h5* in both **lstm_models/many_to_one** and **lstm_models/many_to_many_same** following the documentation provided for [RQ1](/tutorial/evaluate_rq1.md). Please decompose both in *unrolled* mode. 

Once decomposed, we have to ensure that the following variables point to `model1_combined` and `lstm_models` in this script: *replace/inter/manyMany_manyOne/compare_replace.py*:
```
activation = 'lstm_models'
bestModelName = 'model1_combined'
faultyModelName = 'model1_combined'
```
If it doesn't, please change it to the one you want to evaluate inter-replace on. Once this basic sanity checking is done, please follow the below steps to evaluate it:

1. Open the terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run the following:
```
python3 -m replace.inter.manyMany_manyOne.compare_replace
```
The script will attempt to read and write from the directory. Please make sure that it has such permissions. 

4. Depending on the task, it may run for a while. It will print step-by-step replace results, which will be saved as a `csv` file in the same directory with the name *result_lstm_models_model1_combined.csv*. In every scenario, it will print the module accuracy after replacement.

