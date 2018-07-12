# Self-teaching

Code associated with:

Yang, S. C-H., Vong, W.K., Yu, Y., & Shafto, P. *A unifying computational framework for teaching and active learning* (submitted)

## Directory structure

- `models`: Contains the different models (active learning, teaching and self-teaching) used in the simulations
- `notebooks`: Contains various jupyter notebooks with worked examples
- `old_models`: Original matlab code of Pat's original pedagogical sampling model for the causal graph task and unused models
- `simulations`: Code to run simulations (currently only self-teaching for the concept learning task)
- `tests`: Directory for test code
- `run_simulations.py`: Main file to run the simulations and generate the figures from the paper

## Running the code

To run the simulations and produce the figures in the paper:

```bash
python run_simulations.py
```

To run the tests:

```bash
pytest
```

The tests require `pytest` on your machine, which can be installed with the following:

```bash
pip install -U pytest
```
