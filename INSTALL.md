## Disclaimer
This project has been tested on an Intel-based Mac OS using Python 3.7. We highly recommend using a similar setup for optimal performance. Additionally, to ensure a clean and isolated environment, we suggest utilizing a virtual Python environment. This tutorial will guide you through the setup process, including the creation of a virtual environment.

# Installation and Usage

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
