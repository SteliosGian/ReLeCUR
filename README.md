# ReLeCUR

Recommendation Systems: A Reinforcement Learning Approach of the Cold-User Problem

## How to Run

### Creating the Environment

Everything is packaged in a Conda environment. You can create the environment with all the necessary packages using the ```environment.yml``` file. To create the environment, run:

```bash
conda env create -f environment.yml
```

Make sure you have conda installed on your machine.

### Active Learning Strategies

The Active Learning Strategies are in the ```active_learning.py``` or by using jupyter notebook in ```active_learning_notebook.ipynb```. To run, either use Jupyter Notebook to run the cells or use the following command:

```bash
python active_learning.py
```

Make sure you're using the conda virtual environment and the dataset is present.

### Reinforcement Learning

The Reinforcement Learning approach code is in the ```relecur.py``` or by using jupyter notebook in ```relecur_notebook.ipynb```. Again, either use Jupyter Notebook to run the cells or use the following command:

```bash
python relecur.py
```

### RL Environments

The Reinforcement Learning environments are in the ```environment_al.py``` and in ```environment_items.py``` files. The former contains the environment for the AL-based method, and the latter contains the environment for the Item-based method. These environment are imported in the ```relecur.py``` file.
