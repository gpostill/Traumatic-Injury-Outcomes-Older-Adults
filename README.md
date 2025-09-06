# Traumatic Injury Outcomes Older Adults


This repository hosts three related projects that study long-term outcomes after injury in older adults. All projects draw on the same data source and use a shared codebase for ingestion, validation, preprocessing, metrics, and figure/table exports. The structure keeps methods consistent across projects while allowing each to evolve independently.

----
### Repository Goals
-  Provide a single, consistent pipeline for cohort construction, feature engineering, model training, and reporting across all projects.
-  Ensure compliance with privacy, data-sharing, and ethics requirements (no identifiable data leaves the secure environment).
-  Make outputs manuscript-ready (stable file names for tables/figures, vector graphics).
-  Standardize fairness, calibration, and clinical utility assessment across projects.

----
### Thesis Project Overview

| Project                                      | Branch                  | Core question                                                                                                                                                            | Primary methods                                                                                                                    | Status         |
| -------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| **A. Pre-injury Health Clusters & Outcomes** | `preinjury_clusters`  | Can unsupervised ML on **pre-injury** data reveal clinically meaningful clusters of older adults that predict survival and functional independence?                      | Feature construction from 2-yr lookback; clustering (e.g., OPTICS/K-Prototypes/UMAP); cluster stability; association with outcomes | 🟢 active      |
| **B. Long-term Outcome Prediction**          | `longterm_prediction` | Can we predict **alive and at home** / **functional independence** at 6–24 months post-injury?                                                                           | Logistic regression, gradient boosting; calibration; decision curves                                       | 🟡 in progress |
| **C. Equity & Clinical Utility**             | `equity_utility`      | Do predictions and clusters **generalize** and perform equitably across subgroups (e.g., age bands, sex, comorbidity/frailty strata), and do they offer **net benefit**? | Subgroup performance; PROGRESS-Plus analyses; decision curve analysis; sensitivity analyses                                        | 🟡 in progress |

The three projects share: cohort definition, feature registry, evaluation suite, and export conventions.

----
### Project Scripts

**A. Pre-injury Health Clusters & Outcomes** 
- **Aim1_OG_KProto_Frail.py** - code to determine the optimal number of clusters and generate the optimal clusters on the *frail* training data.
- **Aim1_OG_Cluster_Description_Frailty_Final.py** - code to determine the optimal number of clusters and generate the optimal clusters on the *frail* data (entire cohort). These are the final clusters. 
- **Aim1_OG_Cluster_Description_Frailty_Final.py** - code to determine the optimal number of clusters and generate the optimal clusters on the *non-frail* training data.
- **Aim1_OG_Cluster_Description_Frailty_Final.py** - code to determine the optimal number of clusters and generate the optimal clusters on the *non-frail* data (entire cohort). These are the final clusters. 
- **Aim1_OG_Clusters_Outcomes_Final.py** - code to evaluate the outcomes (survival and functional independence) across all clusters generated. 
- **Aim1_OG_Cluster_Description_Frailty_Final.py** - code to describe the projects included across all clusters generated. 

*Note that the project scripts for the remaining two thesis projects are not yet published*

----
### Data Access & Governance

Data Source: Provincial/Regional trauma registry linked with health administrative data (housed at ICES) for community-dwelling adults ≥65 years admitted with moderate/severe injury (ISS > 9), 2015–2023. Access is restricted and analyses are performed within an approved secure environment (e.g., a data custodian’s enclave).

This repository includes:
- Configurations and schema contracts (column dictionaries, expected ranges).
- Parameterized code that runs inside the secure environment.
- Synthetic/toy examples for documentation.

This repository does not include:
- Raw data or row-level outputs.
- Secure paths 

----
### License
- Code: 
- Docs (text/figures): CC-BY 4.0
- Data: Not included; governed by data custodian agreements and ethics approvals
  
----
### Contact
**Gemma Postill** — MD/PhD Candidate  |  gemma.postill@utoronto.ca

  https://orcid.org/0000-0001-6185-995X
