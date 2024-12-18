# Algorithm Toolkit

A set of cognitive neuroscience inspired agents and learning algorithms.

These consist of implementations of the canonical Successor Representation algorithms and more.

The algorithms included here are all tabular. Tabular algorithms work with observations that are integer representations of the state of the agent (e.g., which grid the agent is in a grid world). This corresponds to the `index` observation type.



## Included algorithms

| Algorithm | Function(s) | Update Rule(s) | Reference | Description | Code Link |
| --- | --- | --- | --- | --- | --- |

| TD-SR | ψ(s, a), ω(s) | one-step temporal difference | [Dayan, 1993](https://ieeexplore.ieee.org/abstract/document/6795455) | A basic successor representation algorithm | [Code](./td_agents.py) |

| Dyna-SR | ψ(s, a), ω(s) | one-step temporal difference, replay-base dyna | [Russek et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) | A dyna successor representation algorithm | [Code](./dyna_agents.py) |




## Algorithm hyperparameters

Below is a list of the common hyperparameters shared between all algorithms and agent types. Typical value ranges provided are meant as rough guidelines for generally appropriate learning behavior. Depending on the nature of the specific algorithm or task, other values may be more desirable.

* `lr` - Learning rate of algorithm. Typical value range: `0` - `0.1`.
* `gamma` - Discount factor for bootstrapping. Typical value range: `0.5` - `0.99`.
* `poltype` - Policy type. Can be either `softmax` to sample actions proportional to action value estimates, or `egreedy` to sample either the most valuable action or random action stochastically.
* `beta` - The temperate parameter used with the `softmax` poltype. Typical value range: `1` - `1000`.
* `epsilon` - The probablility of randomly acting using with the `egreedy` poltype. Typical value range: `0.1` - `0.5`.
