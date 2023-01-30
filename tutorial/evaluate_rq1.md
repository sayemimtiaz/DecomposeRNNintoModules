## RQ1 - Decomposition quality
This paper presents a technique for decomposing a trained RNN model into individual modules, one for each output class. The research question (RQ) evaluated in this paper assesses the effectiveness of this decomposition technique. The RQ measures the impact of decomposition on model accuracy by comparing the accuracy of the composed model using decomposed modules to that of the original monolithic model from which the modules were derived. Additionally, the RQ evaluates the similarity between the decomposed modules and the monolithic model using the Jaccard index. The following steps provide some examples for running experiments to answer this RQ. Please read the [project structure documentation](/tutorial/structure.md) before running the following experiments.

## General instructions
To decompose an RNN model, the below steps, in general, will have to be followed:
1. A model to be decomposed must already be trained and placed in the `{X}\{Y}\h5` directory where **X refers to either of lstm_models, gru_models, vanilla_rnn_models, or relu_models**, and **Y refers to different input-output architectures, such as one_to_one, many_to_one, one_to_many, many_to_many_same, and many_to_many_different**. For the user's convenience, pre-trained models are already provided in each *h5* directory.
2. Each model can be trained in two modes: *rolled* and *unrolled* except for *one-to-one* models, which can only be decomposed in *rolled* mode.
3. To decompose in a *rolled* mode, open the terminal in the root directory of the cloned repository and type: `python3 -m {X}.{Y}.decomposer_rolled` (See above for the meaning of *X* and *Y*)
4. To decompose in an *unrolled* mode, open the terminal in the root directory of the cloned repository and type: `python3 -m {X}.{Y}.decomposer_unrolled` (See above for the meaning of *X* and *Y*)


### Example 1: decompose and evaluate a one-to-one model with two stacked GRU layers in *rolled* mode
To decompose and evaluate such a model, we first need to ensure that such an already trained model exists. For this case, we will have to check that `model2.h5` exists in the directory *gru_models/one_to_one/h5*. If not, it can be trained via the `model.py` script provided in the *gru_models/one_to_one* directory. Furthermore, we have to ensure that the following variable points to `model2.h5` in this script: *gru_models/one_to_one/decomposer_rolled.py*:
```
model_name = os.path.join(root, 'h5', 'model2.h5')
```
If it doesn't, please change it to the one you want to decompose. Once this basic sanity checking is done, please follow the below steps to decompose and evaluate it:

1. Open the terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run the following:
```
python3 -m gru_models.one_to_one.decomposer_rolled
```
The script will attempt to read and write from the directory. Please make sure that it has such permissions. 

4. Depending on the task, it may run for a while. Usually, *rolled* mode will run faster than *unrolled* mode. It will print step-by-step module generation, which will be saved in: the *gru_models/one_to_one/modules* directory. In the end, it will show the accuracy of both the monolithic model and modularized model, including the Jaccard index. A sample output will look like this: 
```
Start Time:22:52:37
#Module 0 in progress....
#Module 1 in progress....
#Module 2 in progress....
#Module 3 in progress....
#Module 4 in progress....
#Module 5 in progress....
evaluating rolled: /Users/sayem/Documents/Research/RNN/DecomposeRNNintoModules/lstm_models/one_to_one/h5/model1.h5
Modularized Accuracy: 0.9715242881072027
Model Accuracy: 0.9701842546063651
mean jaccard index: 0.6620745716232255
```
Here both accuracy and index range from 0 to 1.0.

### Example 2: decompose and evaluate a many-to-one model with one LSTM layers in *unrolled* mode
To decompose and evaluate such a model, we first need to ensure that such an already-trained model exists. For this case, we will have to check that `model1.h5` exists in the directory *lstm_models/many_to_one/h5*. If not, it can be trained via the `model.py` script provided in the *lstm_models/many_to_one* directory. Furthermore, we have to ensure that the following variable points to `model1.h5` in this script: *lstm_models/many_to_one/decomposer_unrolled.py*:
```
model_name = os.path.join(root, 'h5', 'model1.h5')
```
If it doesn't, please change it to the one you want to decompose. Once this basic sanity checking is done, please follow the below steps to decompose and evaluate it:

1. Open the terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run the following:
```
python3 -m lstm_models.many_to_one.decomposer_unrolled
```
The script will attempt to read and write from the directory. Please make sure that it has such permissions. 

4. It will generate similar output as shown above.

### Example 3: decompose and evaluate a many-to-many language model with three LSTM layers in *rolled* mode
First, please download the corresponding pre-trained language model from [here](https://doi.org/10.57967/hf/0307). For this example, we want this downloaded model (*language_models/lstm/rq1 models/model_LSTM_3layer.h5*) placed inside *lstm_models/many_to_many_different/many_to_many_different_rolled/h5* directory in the cloned repository. Furthermore, one can train a language model from the script provided [here](https://doi.org/10.57967/hf/0307). In the next step, we have to ensure that the following variable points to `model_LSTM_3layer.h5` in this script: *lstm_models/many_to_many_different/many_to_many_different_rolled/decomposer_rolled.py*:
```
model_name = os.path.join(root, 'h5', 'model_LSTM_3layer.h5')
```
If it doesn't, please change it to the one you want to decompose. Once this basic sanity checking is done, please follow the below steps to decompose and evaluate it:

1. Open the terminal in the root directory of the cloned repository. 

2. Activate the environment:
```
source rnnenv/bin/activate
```
3. Run the following:
```
python3 -m lstm_models.many_to_many_different.many_to_many_different_rolled.decomposer_rolled
```
4. It will generate similar output as shown above. Similarly, other language models can be decomposed too.
