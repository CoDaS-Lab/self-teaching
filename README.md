# Self-teaching

## Directory structure

- `causal_learning`: Contains the different models for the causal graph task
- `concept_learning`: Contains the different models for the concept learning task
- `notebooks`: Contains various jupyter notebooks with worked examples
- `original_causal_learning_code`: Matlab code of Pat's original pedagogical sampling model for the causal graph task
- `simulations`: Code to run simulations (currently only self-teaching for the concept learning task)
- `tests`: Directory for test code

## Running the code

To run the concept learning simulations:

```bash
python simulations/self_teaching_simulations.py
```

To run the causal learning simulations (both active learning and self-teaching):

```bash
python causal_learning/graph_simulations.py
```

To run the tests:

```bash
pytest
```

The tests require `pytest` on your machine, which can be installed with the following:

```bash
pip install -U pytest
```
