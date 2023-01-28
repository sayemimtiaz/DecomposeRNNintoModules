# DecomposeRNNintoModules
Modularity and decomposition are central ideas in software engineering that enable better software evolution. One of the important benefits of decomposing software into modules is the ability to reassemble and replace modules flexibly. This paper proposes an approach to decompose a trained model into modules to enable such benefits for recurrent neural networks. In particular, it introduces a method for breaking down a trained RNN model into separate modules, one for each output class. The research question evaluated in this paper examines the efficacy of this decomposition technique and its applicability in various scenarios. In particular, the paper asks three research questions: 
- **RQ1 - Decomposition Quality**: Does decomposing RNN model into modules incur cost?
- **RQ2 - Reusability**: Can Decomposed Modules be Reused to Create a New Problem?
- **RQ3 - Replaceability**: Can Decomposed Modules be Replaced?

This repository exposes all the code, data, and instrcutions to reproduce the resukts presented in the paper.

## Project structure
The structure of this repository has been detailed [here](/tutorial/structure.md).

## Installation and Usage

Please follow this [docmentation](/INSTALL.md) to install the system and make it runnable.


## Running experiments
The paper evaluates three research questions (RQ). A detailed instruction to reproduce results for every RQs are listed in following files:
1. **RQ1:** The quality of the RNN decomposition (in terms of accuracy), similairty between model and modules (in terms of Jaccard index) are evaluated in this RQ. The detailed instructions can be found [here](/tutorial/evaluate_rq1.md). 
2. **RQ2:** This RQ evaluates different reuse scenarios. The detailed instructions can be found [here](/tutorial/evaluate_rq2.md). 
3. **RQ3:** This RQ evaluates different replace scenarios. The detailed instructions can be found [here](/tutorial/evaluate_rq3.md). 
