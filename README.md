## CRL-TCP: Continual Reinforcement Learning-based Test Case Prioritization

CRL-TCP is a test case prioritization technique based on continuous reinforcement learning, introducing the concept of
continuous reinforcement learning during the retraining process.

### Execution

Run.py is the main entry file. You can run the CRL-TCP method using ``python Run.py``. During execution, a **CRL-TCP.log** file will be generated, and the execution results will be stored in the **results** folder at the same
level as the src directory. You can use ``python Statistic.py`` to analyze the execution results, and the statistical
results will be
saved
in **CRL-TCP.xlsx**.

* Run.py is executed repeatedly for each project, and the project execution list can be set in the hyperparameters.
* Statistic.py generates statistical information for the project execution results.
* Agent.py is related to the continuous reinforcement learning agent.
* Env.py is related to the environments.
* Model.py is about the model.
* Reward.py is about the reward functions.

### Environment

Development environment

```
Python: 3.10.9
PyTorch: V2.2.0
OS: Ubuntu 20.04.6 LTS
```

Third-party libraries

```
numpy: 1.23.5
pandas: 1.5.3 
scipy: 1.10.0 
openpyxl: 3.0.10
```

### Dataset

The preprocessed dataset is placed under `dataset.zip`. Please ensure that the path to these .csv files is correctly set
in the code. 
