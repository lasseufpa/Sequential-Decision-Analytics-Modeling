# Sequential Decision Analytics and Modeling

Python implementations of selected problems from **Sequential Decision Analytics and Modeling** by Warren B. Powell. Each chapter is modeled using the Universal Modeling Framework (UMF) for sequential decisions under uncertainty.

## Course Goal

Learn and apply the five core elements of the UMF to a variety of sequential decision problems:

| Element | Notation | Description |
|---|---|---|
| State | $S^n$ | All information needed to make a decision at time $n$ |
| Decision | $x^n$ | The action taken based on a policy $X^\pi(S^n)$ |
| Exogenous Information | $W^{n+1}$ | The uncertainty arriving after the decision |
| Transition Function | $S^{n+1} = S^M(S^n, x^n, W^{n+1})$ | How the state evolves |
| Objective Function | $\max_\pi \mathbb{E}\left[\sum C(S^n, x^n, W^{n+1})\right]$ | Maximize cumulative contribution |

## Chapters

| # | Topic | Folder | Key Concepts |
|---|---|---|---|
| 1 | Modeling Sequential Decision Problems | [ch01_modeling](chapters/ch01_modeling/) | UMF, Five Elements, Policies |
| 2 | An Asset Selling Problem | [ch02_asset_selling](chapters/ch02_asset_selling/) | Threshold Policies, Time Series Prices |
| 3 | Adaptive Market Planning | [ch03_adaptive_market](chapters/ch03_adaptive_market/) | Uncertainty Modeling, Adaptive Decisions |
| 4 | Learning the Best Diabetes Medication | [ch04_diabetes](chapters/ch04_diabetes/) | Multi-armed Bandits, Bayesian Learning, Exploration vs. Exploitation |
| 5 | Stochastic Shortest Path Problems — Static | [ch05_shortest_path_static](chapters/ch05_shortest_path_static/) | Shortest Paths, ADP, Post-decision State |
| 6 | Stochastic Shortest Path Problems — Dynamic | [ch06_shortest_path_dynamic](chapters/ch06_shortest_path_dynamic/) | Deterministic Lookahead, Parameterized Policies |
| 7 | Applications, Revisited | [ch07_applications_revisited](chapters/ch07_applications_revisited/) | Four Policy Classes, Online vs. Offline, Policy Search |
| 8 | Energy Storage I | [ch08_energy_storage_1](chapters/ch08_energy_storage_1/) | Dynamic Programming, VFA, Buy-Low Sell-High |
| 9 | Energy Storage II | [ch09_energy_storage_2](chapters/ch09_energy_storage_2/) | Gaussian Process, Deterministic Lookahead |
| 10 | Supply Chain Management I: Two-Agent Newsvendor | [ch10_supply_chain_newsvendor](chapters/ch10_supply_chain_newsvendor/) | Multi-Agent Decisions, Newsvendor Problem |
| 11 | Supply Chain Management II: The Beer Game | [ch11_supply_chain_beer_game](chapters/ch11_supply_chain_beer_game/) | Multi-Agent, Anchor-and-Adjust, Lookahead |
| 12 | Ad-Click Optimization | [ch12_ad_click](chapters/ch12_ad_click/) | Bayesian Learning, Value of Information |
| 13 | Blood Management Problem | [ch13_blood_management](chapters/ch13_blood_management/) | Resource Allocation, Myopic vs. VFA Policies |
| 14 | Optimizing Clinical Trials | [ch14_clinical_trials](chapters/ch14_clinical_trials/) | Stopping Rules, Patient Enrollment |

## Getting Started

```bash
git clone https://github.com/lasseufpa/Sequential-Decision-Analytics-Modeling.git
cd Sequential-Decision-Analytics-Modeling
pip install -r requirements.txt
```

To open a chapter notebook:

```bash
jupyter notebook chapters/ch04_diabetes/chapter_04.ipynb
```

## Repository Structure

```
common/              Shared base classes (State, Simulator, Policy)
chapters/
  ch01_modeling/     Overview of the Universal Modeling Framework
  ch02_asset_selling/
  ch03_adaptive_market/
  ch04_diabetes/     Model, policies, evaluation, and interactive notebook
  ch05_shortest_path_static/
  ch06_shortest_path_dynamic/
  ch07_applications_revisited/
  ch08_energy_storage_1/
  ch09_energy_storage_2/
  ch10_supply_chain_newsvendor/
  ch11_supply_chain_beer_game/
  ch12_ad_click/
  ch13_blood_management/
  ch14_clinical_trials/
data/                Reference data
notebooks/           Sandbox for quick exploration
```

## Bibliography

- Powell, W. B. (2024). *Sequential Decision Analytics and Modeling*. Now Publishers.
