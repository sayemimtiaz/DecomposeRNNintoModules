# Project structure
This document outlines the structure and organization of the project. The repository includes scripts, paper results, and pre-trained models, with the exception of large language models which can be found at [here](https://doi.org/10.57967/hf/0307). 
The specific artifacts included in this repository are listed as follows:

## Code: 
We have provided all Python scripts for decomposing and evaluating RNN models. We provide a description of important directories/scripts below: 

- *gru_models*: scripts for decomposing *GRU* models and evaluating them. It's organized in the following way:
	- *gru_models/one_to_one*: decomposition scripts for *GRU* one to one model. This directory in turn contains following contents:
		- *h5*: This directory includes several pre-trained monolithic models for the user's convenience. These pre-trained models can be used as examples for decomposing the model. This directory contains models in this format: `modelX.h5`, where `X` ranges from 1 to 4. For example, `model2.h5` would mean a GRU one to one model with two stacked GRU layers.
		- *decomposer_rolled.py*: This is the python script that needs to be executed for decomposing a one to one GRU model from *h5* directory in *rolled* mode. 
		- *decomposer_unrolled.py*: In addition, this file may exist where it's appropriate. This is the python script that needs to be executed for decomposing a one to one GRU model from *h5* directory in *unrolled* mode. 
		- *model.py*: This script provides a model defintion to be trained from scratch. 
	All other model types (i.e., one to many, many to many etc.) follows the similar structure, hence we will skip describing it repetitively.
	
	- gru_models/one_to_many: decomposition scripts for *GRU* one to many model.
	- gru_models/many_to_one: decomposition scripts for *GRU* many to one model.
	- gru_models/many_to_many_same: decomposition scripts for *GRU* many to many model.
	- gru_models/many_to_many_different: decomposition scripts for *GRU* many to many(Encoder-Decoder) language model. 
 
- *lstm_models*: scripts for decomposing *LSTM* models and evaluating them. It's organized in a similar way as *gru_models*.


- *vanilla_rnn_models*: scripts for decomposing *Vanilla RNN* models with *Tanh* activation and evaluating them. It's organized in a similar way as *gru_models*.

- *relu_models*: scripts for decomposing *Vanilla RNN* models with *Relu* activation and evaluating them. It's organized in a similar way as *gru_models*.

- *reuse*: Scripts for evaluating RQ2: Reusability.

- *replace*: Scripts for evaluating RQ3: Replaceability.

- *motivating_example*: Recreates motivating examples.

- All other directories are helper scripts required to run these experiments.


## Results: 
The [results directory](/results) contains all the results. Results are organized in following ways:

1. results/{X}: Results for RNN variant X, X={Vanilla RNN, LSTM, GRU}
2. results/X/rq1: Results for RQ1
3. results/X/rq2/intra/{Y}: Results for RQ2:intra-reuse for Y model, Y={one_to_one, one_to_many, many_to_one, many_to_many_same, many_to_many_different}.
4. results/X/rq2/inter/{Z}: Results for RQ2:inter-reuse for Z model, Z={manyOne_manyMany, oneOne_oneMany}.
5. results/X/rq3/intra/{A}: Results for RQ3:intra-replace for A model, A={one_to_one, one_to_many, many_to_one, many_to_many_same}.
6. results/X/rq3/inter/{Z}: Results for RQ3:intra-replace for Z model.
7. results/X/motivating example/problem 1: Results for motivating example experiments problem 1. 
8. results/X/motivating example/problem 2: Results for motivating example experiments problem 2. 
