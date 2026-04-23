# Modelling Chronic Respiratory Disease Burden in China  
### A Multi-Stage Feature Selection and ARDL-ECM Analysis

## Overview

This project develops a time-series analytical framework to model the dynamics of chronic respiratory disease (CRD) burden in China. Using national-level data from the Global Burden of Disease (GBD) study and World Bank indicators, the analysis integrates feature selection, econometric modelling, and visual analytics.

The objective is to identify key risk factors associated with CRD burden, evaluate their short-run effects, and assess long-run equilibrium relationships.

---

## Data Sources

- Global Burden of Disease (GBD): CRD DALY rates and risk factor exposures  
- World Bank: Socioeconomic indicators (GDP per capita, population ageing, government health expenditure)

All data used are publicly available.

---

## Methodology

The modelling framework follows a four-stage pipeline:

### 1. Data Processing and Prechecks
- Construction of a national time-series dataset  
- First differencing to reduce non-stationarity  
- Lag feature generation (lag 0–2)  
- Augmented Dickey–Fuller (ADF) tests  
- Variance Inflation Factor (VIF) analysis  

### 2. Feature Selection
Multiple approaches are compared under small-sample constraints:
- Elastic Net regression  
- Backward elimination (p-value based)  
- Random Forest importance ranking  
- VIF-based filtering  

### 3. Model Evaluation
- Ordinary Least Squares (OLS) models  
- Leave-One-Out Cross-Validation (LOOCV)  
- Comparison of predictive performance across model specifications  

### 4. Dynamic Modelling
- Autoregressive Distributed Lag (ARDL) models  
- Error Correction Model (ECM) representation  
- Bounds testing for cointegration

## Visual Analytics and Dashboard
It does not present causal conclusions, but instead communicates:
- Observed trends  
- Model-implied relationships  
- Short-run vs long-run dynamics  

---

## Key Findings

- Household air pollution and population ageing emerge as key predictors of CRD burden  
- Short-run dynamics are captured through differenced models  
- ARDL-ECM analysis indicates a stable long-run equilibrium relationship  
- Model results highlight the importance of delayed environmental effects and demographic trends  

---

## Repository Structure
