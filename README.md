# Portfolio-Optimizer-
Constrained optimizer using Mean–variance framework 

# Portfolio Optimizer (CVXPY)

A constrained mean–variance portfolio optimizer for daily equity long–short strategies.

This repository implements a clean, production-style optimizer that converts
predicted returns (alphas) into portfolio weights under common used constraints.

## Features

- Mean–variance optimization
- Dollar neutrality
- Gross leverage constraint
- Per-asset position limits
- Optional factor neutrality
- Optional turnover penalty
- CVXPY-based quadratic programming

## Objective

Maximize:

    μᵀw − (λ / 2) wᵀΣw

Subject to:

- ∑ w = 0 (Market Neutral constraint)
- ∑ |w| ≤ L (Leverage constraint)
- |wᵢ| ≤ w_max (Concentration constraint)
- Bᵀw = 0 (factor neutrality)
