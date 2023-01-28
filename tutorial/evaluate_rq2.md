## RQ2: Reusability of modules
This paper introduces a method for breaking down a trained RNN model into separate modules, one for each output class. The research question evaluated in this paper examines the efficacy of this decomposition technique and its applicability in various scenarios. Through this research question, we demonstrate how the decomposed modules can be reused in different situations. Our evaluation centers on two types of reuse: a) utilizing the modules within the same input-output type (referred to as **intra-reuse**), and b) using the modules from different types to create a new problem (referred to as **inter-reuse**).

In the first scenario, we demonstrate how to create a new problem within the same task by training a binary classifier using only two classes from a given dataset, without the need for starting from scratch by reusing existing modules. In the second scenario, we evaluate the effectiveness of reusing modules across different tasks, using different datasets.

The following steps provide some examples for running experiments to answer this RQ. Please read the [RQ1 evaluation documentation](/tutorial/evaluate_rq1.md) before procedding to run following experiments.

### Example 1: evaluate intra-reuse scenario for a one to one model with one GRU layer 
In this case, we want to evaluate intra-reuse for one to one **model1.h5** GRU model. To do so, it must first be decomposed following the documentation provided for [RQ1](/tutorial/evaluate_rq1.md). Once decomposed, we have to ensure that following variables point to `model1` and `gru_models` in this script: *reuse/intra/one_to_one/compare_reuse.py*:
```
activation = 'gru_models'
modelName = 'model1'
```
If it doesn't, please change it to the one you are looking to evaluate intra-reuse on. Once, this basic sanity checking is done, please follow below steps to evaluate it:

1. Open terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run following:
```
python3 -m reuse.intra.one_to_one.compare_reuse
```
The script will attempt to read and write from directory. Please make sure that it has such permissions. 

4. Depending on the task, it may run for a while. It will print step by step reuse scenarios, which will be saved as a `csv` file in the same directory with name *result_gru_models_model1.csv*. In every scenario, it will print both module reuse accuracy vs. the accuracy when trained from te scratch. A sample output for the illustration is provided below:
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
In this, case reuse is evaluated between a model of **one-to-one** and **one-to-many** with one LSTM layer. Note that, in this case, we need to combine the vocabulary of both the dataset for *Embedding* layer to work since same input from both the dataset will go to modules from both models. Therefore, a model must be trained with common vocabulary. We have provided a pre-trained model trained with common vocabulary for the convenience. The model is named after *model1_combined.h5*. However, a model from scratch can be trained too using the *model.py* script as described in [RQ1 documentation](/tutorial/evaluate_rq1.md).

Similar to intra-reuse, we must first decompose *model1_combined.h5* in both *lstm_models/one_to_one* and *lstm_models/one_to_many* following the documentation provided for [RQ1](/tutorial/evaluate_rq1.md). Please note that *lstm_models/one_to_one/h5/model1_combined.h5* must be decomposed in *rolled* mode, and *lstm_models/one_to_many/h5/model1_combined.h5* in *unrolled* mode for this to work as we found those mode to be best performing in their respective architecture. Similarly, if one were to evaluate inter-reuse between (many-to-one)-to-(many-to-many-same), both models in *lstm_models/many_to_many_same/h5/model1_combined.h5* and *lstm_models/many_to_one/h5/model1_combined.h5* shuld be decomposed in *unrolled* mode. This is true for other types of models too, i.e., *gru_models*, *relu_models*, etc.

Once decomposed, we have to ensure that following variables point to `model1_combined` and `lstm_models` in this script: *reuse/inter/oneOne_oneMany/compare_reuse.py*:
```
activation = 'lstm_models'
modelName = 'model1_combined'
```
If it doesn't, please change it to the one you are looking to evaluate inter-reuse on. Once, this basic sanity checking is done, please follow below steps to evaluate it:

1. Open terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run following:
```
python3 -m reuse.inter.oneOne_oneMany.compare_reuse
```
The script will attempt to read and write from directory. Please make sure that it has such permissions. 

4. Depending on the task, it may run for a while. It will print step by step reuse scenarios, which will be saved as a `csv` file in the same directory with name *result_lstm_models_model1_combined.csv*. In every scenario, it will print both module reuse accuracy vs. the accuracy when trained from te scratch in a similar fashion.

