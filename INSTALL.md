## RQ1
This paper presents a technique for decomposing a trained RNN model into individual modules, one for each output class. The research question (RQ) evaluated in this paper assesses the effectiveness of this decomposition technique. The RQ measures the impact of decomposition on model accuracy by comparing the accuracy of the composed model using decomposed modules to that of the original monolithic model from which the modules were derived. Additionally, the RQ evaluates the similarity between the decomposed modules and the monolithic model using the Jaccard index. The following steps provide some examples for running experiments to answer this RQ.

### preliminaries 

## Decompose a

This tutorial assumes a Python 3.7 or 3.8 is installed in the user system. Additionally, the project requires several packages which are listed in the [requirements.txt](/requirements.txt) file.  In the following steps, we will provide a detailed guide on how to set up a virtual environment and install the necessary packages for running the project:

1. Clone the repository in your local system.
```
git clone https://github.com/sayemimtiaz/DecomposeRNNintoModules.git
```

2. Open terminal in the root directory of the cloned repository. 

3. Create a virtual ennvironment. Run on command line (all commands are written assuming a bash terminal):
```
python3 -m venv rnnenv
```
4. Activate the environment:
```
source rnnenv/bin/activate
```
 You can exit this virtual environment by `deactivate`.

5. Install required packages:
```
pip install -r requirements.txt
```

6. Test the installation by running any script within project. For instance, following will decompose a LSTM model, showing modularized accuracy, model accuracy, and jaccard index as a final summary of the run:
```
python3 -m lstm_models.one_to_one.decomposer_rolled
```

# Project structure
The structure of this repository has been detailed [here](/tutorial/structure.md).

# Running experiments
The paper evaluates three research questions (RQ). A detailed instruction to reproduce results for every RQs are listed in following files:
1. **RQ1:** The quality of the RNN decomposition (in terms of accuracy), similairty between model and modules (in terms of Jaccard index) are evaluated in this RQ.
2. **RQ2:** This RQ evaluates different reuse scenarios. 
3. **RQ3:** This RQ evaluates different replace scenarios. 

