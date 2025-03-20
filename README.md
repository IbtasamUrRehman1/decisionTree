# Fair Information Gain Decision Tree

This repository contains a Python implementation of a modified decision tree algorithm that incorporates fairness considerations through the use of Fair Information Gain (FIG).

## Overview

Traditional decision trees use Information Gain (IG) as a criterion for splitting nodes. However, IG does not account for fairness, potentially leading to biased outcomes. This implementation introduces Fair Information Gain (FIG), which combines IG with Fairness Gain (FG) to promote more equitable decision-making.

## Files

* `fair_information_gain_tree.py`: Contains the Python code implementing the Fair Information Gain decision tree algorithm, along with a comparison to a traditional decision tree.

## Requirements

* Python 3.x
* NumPy
* Scikit-learn

You can install the required packages using pip:

```bash
pip install numpy scikit-learn
