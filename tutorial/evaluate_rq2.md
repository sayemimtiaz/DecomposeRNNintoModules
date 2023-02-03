## RQ2: Reusability of modules
This paper introduces a method for breaking down a trained RNN model into separate modules, one for each output class. The research question evaluated in this paper examines the efficacy of this decomposition technique and its applicability in various scenarios. Through this research question, we demonstrate how the decomposed modules can be reused in different situations. Our evaluation centers on two types of reuse: a) utilizing the modules within the same input-output type (referred to as **intra-reuse**), and b) using the modules from different types to create a new problem (referred to as **inter-reuse**).

In the first scenario, we demonstrate how to create a new problem within the same task by training a binary classifier using only two classes from a given dataset without the need to start from scratch by reusing existing modules. In the second scenario, we evaluate the effectiveness of reusing modules across different tasks using different datasets.

The following steps provide some examples for running experiments to answer this RQ. Please read the [RQ1 evaluation documentation](/tutorial/evaluate_rq1.md) before running the following experiments.

## General instructions
To evaluate the reuse scenarios, the below steps should be followed in general:
- First, the target model on which reuse is to be evaluated must be decomposed following the documentation for [RQ1](/tutorial/evaluate_rq1.md). In the *intra-reuse* mode, reuse is evaluated within the same architecture and dataset; hence, only the target model needs to be decomposed. However, in *the inter-reuse* mode, modules are reused from different architectures and datasets. So, in this case, target models from both should be decomposed.
- Each architecture should be decomposed in a specific mode. For example:
  - **one-to-one**: It should be decomposed in *rolled* mode.
  - **many-to-one**: It should be decomposed in *unrolled* mode.
  - **one-to-many**: It should be decomposed in *unrolled* mode.
  - **many-to-many-same**: It should be decomposed in *unrolled* mode.
  - **many-to-many-different**: It should be decomposed in *rolled* mode.
In our experiment, we found those modes to be the best performing. Hence the reuse evaluation is done on modules decomposed in the best mode.

- Once the target model(s) is decomposed, to evaluate **intra-reuse**, `compare_reuse` script should be executed with this command: `python3 -m reuse.intra.{X}.compare_reuse`, where *X refers to different input-output architectures, such as one_to_one, many_to_one, one_to_many, many_to_many_same, and many_to_many_different*. 
- For **inter-reuse**, run: `python3 -m reuse.inter.{Z}.compare_reuse`, where *Z refers to oneOne_oneMany, and manyOne_manyMany*. Here, *oneOne_oneMany* indicates that modules are reused from *one-to-one* and *one-to-many* model, and *many-to-one* and *many-to-many-same* for *manyOne_manyMany*. 
- Before running the `compare_reuse` script, the following variables should be checked to ensure the intended model is being evaluated:
```
activation = '{X}' #X = {gru_models, lstm_models, vanilla_rnn_models, relu_models}
modelName = 'mymodel' #name of the intended model.
```

### Example 1: evaluate intra-reuse scenario for a one-to-one model with one GRU layer 
In this case, we want to evaluate intra-reuse for one-to-one **model1.h5** GRU model. To do so, it must first be decomposed following the documentation provided for [RQ1](/tutorial/evaluate_rq1.md). Once decomposed, we have to ensure that the following variables point to `model1` and `gru_models` in this script: *reuse/intra/one_to_one/compare_reuse.py*:
```
activation = 'gru_models'
modelName = 'model1'
```
If it doesn't, please change it to the one you are looking to evaluate intra-reuse on. Once this basic sanity checking is done, please follow the below steps to evaluate it:

1. Open the terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run the following:
```
python3 -m reuse.intra.one_to_one.compare_reuse
```
The script will attempt to read and write from the directory. Please make sure that it has such permissions. 

4. Depending on the task, it may run for a while. It will print step-by-step reuse scenarios, which will be saved as a `csv` file in the same directory with the name *result_gru_models_model1.csv*. In every scenario, it will print both module reuse accuracy and accuracy when trained from scratch. A sample output for the illustration is provided below:
```
Start Time:22:52:37
Modularized Accuracy (Class 0 - Class 1): 0.9849381387842926
Trained Model Accuracy (Class 0 - Class 1): 0.9838622915545993
Modularized Accuracy (Class 0 - Class 2): 0.9973614775725593
Trained Model Accuracy (Class 0 - Class 2): 0.9973614775725593
Modularized Accuracy (Class 0 - Class 3): 0.989041095890411
... ...
```
Here the accuracy ranges from 0 to 1.0.

### Example 2: evaluate inter-reuse scenario between (one-to-one)-to-(one-to-many) with one LSTM layer 
In this, case reuse is evaluated between a model of **one-to-one** and **one-to-many** with one LSTM layer. Note that, in this case, we need to combine the vocabulary of both datasets for the *Embedding* layer to work since the same input from both datasets will go to modules from both models. Therefore, a model must be trained with a common vocabulary. We have provided a pre-trained model trained with a common vocabulary for convenience. The model is named after *model1_combined.h5*. However, a model from scratch can be trained, too, using the *model.py* script as described in [RQ1 documentation](/tutorial/evaluate_rq1.md).

Similar to intra-reuse, we must first decompose *model1_combined.h5* in both *lstm_models/one_to_one* and *lstm_models/one_to_many* following the documentation provided for [RQ1](/tutorial/evaluate_rq1.md). Please note that *lstm_models/one_to_one/h5/model1_combined.h5* must be decomposed in *rolled* mode, and *lstm_models/one_to_many/h5/model1_combined.h5* in *unrolled* mode for this to work as we found those models to be best performing in their respective architecture. Similarly, if one were to evaluate inter-reuse between (many-to-one)-to-(many-to-many-same), both models in *lstm_models/many_to_many_same/h5/model1_combined.h5* and *lstm_models/many_to_one/h5/model1_combined.h5* shuld be decomposed in *unrolled* mode. This is true for other models, too, i.e., *gru_models*, *relu_models*, etc.

Once decomposed, we have to ensure that the following variables point to `model1_combined` and `lstm_models` in this script: *reuse/inter/oneOne_oneMany/compare_reuse.py*:
```
activation = 'lstm_models'
modelName = 'model1_combined'
```
If it doesn't, please change it to the one you are looking to evaluate inter-reuse on. Once this basic sanity checking is done, please follow the below steps to evaluate it:

1. Open a terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run the following:
```
python3 -m reuse.inter.oneOne_oneMany.compare_reuse
```
The script will attempt to read and write from the directory. Please make sure that it has such permissions. 

4. Depending on the task, it may run for a while. It will print step-by-step reuse scenarios, which will be saved as a `csv` file in the same directory with the name *result_lstm_models_model1_combined.csv*. In every scenario, it will print both module reuse accuracy and accuracy when trained from scratch in a similar fashion.

