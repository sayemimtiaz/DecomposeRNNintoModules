# DecomposeRNNintoModules
**Title** Decomposing a Recurrent Neural Network into Modules for Enabling Reusability and Replacement

**Authors** Sayem Mohammad Imtiaz, Fraol Batole, Astha Singh, Rangeet Pan, Breno Dantas Cruz, and Hridesh Rajan

**PDF** https://arxiv.org/pdf/2212.05970.pdf

Modularity and decomposition are central ideas in software engineering that enable better software evolution. One of the important benefits of decomposing software into modules is the ability to reassemble and replace modules flexibly. This paper proposes an approach to decompose a trained model into modules to enable such benefits for recurrent neural networks. In particular, it introduces a method for breaking down a trained RNN model into separate modules, one for each output class. The research question evaluated in this paper examines the efficacy of this decomposition technique and its applicability in various scenarios. In particular, the paper asks three research questions: 
- **RQ1 - Decomposition Quality**: Does decomposing RNN model into modules incur cost?
- **RQ2 - Reusability**: Can Decomposed Modules be Reused to Create a New Problem?
- **RQ3 - Replaceability**: Can Decomposed Modules be Replaced?




This repository exposes all the code, data, and instrcutions to reproduce the resukts presented in the paper.

## Project structure
The structure of this repository has been detailed [here](/tutorial/structure.md).

## Installation and Usage

Please follow this [documentation](/INSTALL.md) to install the system and make it runnable.


## Running experiments
The paper evaluates three research questions (RQ). A detailed instruction to reproduce results for every RQs are listed in following files:
1. **RQ1:** The quality of the RNN decomposition (in terms of accuracy), similairty between model and modules (in terms of Jaccard index) are evaluated in this RQ. The detailed instructions can be found [here](/tutorial/evaluate_rq1.md). 
2. **RQ2:** This RQ evaluates different reuse scenarios. The detailed instructions can be found [here](/tutorial/evaluate_rq2.md). 
3. **RQ3:** This RQ evaluates different replace scenarios. The detailed instructions can be found [here](/tutorial/evaluate_rq3.md). 


Please cite the paper as:

```
@inproceedings{imtiaz23rnn,
  author = {Sayem Mohammad Imtiaz and Fraol Batole and Astha Singh and Rangeet Pan and Breno Dantas Cruz and Hridesh Rajan},
  title = {Decomposing a Recurrent Neural Network into Modules for Enabling Reusability and Replacement},
  booktitle = {ICSE'23: The 45th International Conference on Software Engineering},
  location = {Melbourne, Australia},
  month = {May 14-May 20},
  year = {2023},
  entrysubtype = {conference},
  abstract = {
    Can we take a recurrent neural network (RNN) trained to translate between languages and augment it to support a new natural language without retraining the model from scratch? Can we fix the faulty behavior of the RNN by replacing portions associated with the faulty behavior? Recent works on decomposing a fully connected neural network (FCNN) and convolutional neural network (CNN) into modules have shown the value of engineering deep models in this manner, which is standard in traditional SE but foreign for deep learning models. However, prior works focus on the image-based multiclass classification problems and cannot be applied to RNN due to (a) different layer structures, (b) loop structures, (c) different types of input-output architectures, and (d) usage of both nonlinear and logistic activation functions. In this work, we propose the first approach to decompose an RNN into modules. We study different types of RNNs, i.e., Vanilla, LSTM, and GRU. Further, we show how such RNN modules can be reused and replaced in various scenarios. We evaluate our approach against 5 canonical datasets (i.e., Math QA, Brown Corpus, Wiki-toxicity, Clinc OOS, and Tatoeba) and 4 model variants for each dataset. We found that decomposing a trained model has a small cost (Accuracy: -0.6%, BLEU score: +0.10%). Also, the decomposed modules can be reused and replaced without needing to retrain.
  }
```
