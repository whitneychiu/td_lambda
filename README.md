# Learning to Predict by the Methods of Temporal Differences
This repository contains the source code, results graphs for my implementation of the random walk example demonstrated in the paper **Learning to Predict by the Methods of Temporal Differences** by Richard S. Sutton in 1988. [[paper](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)]

## Dependencies
Python==3.9.16 

numpy==1.23.5 

matplotlib==3.7.1 

tqdm==4.65.0 

## Repository Structure
- `main.py` contains the script to train a random walk agent for a single pass and also until the algorithm converges to replicate the original Figure 3, 4, and 5, in the original paper.
- To train the model from scratch and generate the figures, simply run
```python
python main.py
```
