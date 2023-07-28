# Analysing the Belgian Housing Market

<p align="center">
<img src="https://cdn1.iconfinder.com/data/icons/real-estate-set-1/512/3-1024.png"/ width="200" height="200"/>
</p>

<h2 id="table-of-contents"> :book: Table of Contents</h2>
<details open="open">
  <summary>Table of Contents</summary>
<ol>
<li><a href="#Goal-of-the-Analysis"> ➤ Goal of The Analysis</a></li>
<li><a href="#Requirements"> ➤ Requirements</a></li>
<li><a href="#Dataset"> ➤ Dataset</a></li>
<li><a href="#Descriptive-Analysis"> ➤ Descriptive Analysis</a></li>
<li><a href="#Prediction-Modeling"> ➤ Prediction Modeling</a></li>
<li><a href="#Example-of-Usage"> ➤ Example of Usage</a></li>
<li><a href="#Interactive API Documentation"> ➤ Interactive API Documentation</a></li>
<li><a href="#Time-Schedule"> ➤ Time Schedule</a></li>
</ol>
</details>

## Goal of The Analysis 
The main idea behind analysing the Belgian housing market, is to grasp a better understanding of the main features that could be influencing the relative prices. After analysing the data at hand, a prediction analysis had been created based on several comparitive models, namely: linear and non-linear models.


## Requirements
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>

> - Numpy
> - Seaborn
> - Matplotlib
> - Sklearn
> - XGBoost
> - Uvicorn
> - FastApi

## Dataset
The data collected was extracted from a Belgian real estate platform ([Immoweb.be](https://www.immoweb.be/en/search/house/for-sale)), which conglomerates many rental and selling units from real estate agencies.

## Descriptive Analysis
Based on the extracted data, an more in depth review had been made, which can be found in the `Data_Exploration` folder. 

## Prediction Modeling
To investigate whether the price of houses is influenced by sevral features, several regression models had been used. The variables  with the strongest correlation with the dependant variable *Price* were (*Bedrooms, Bathrooms, Toilets, Surface of Plots*). Hence, these features had been tested to see to what extent can these elements predict the changes in house prices.

Ultimatley when running the three models, we can compare which model is most efficient in predicting the housing prices.

## Example of Usage
__1. Download the dataset:__
Download the csv file from the `Data`` folder which consists of the scraped and cleaned data from 
[Immoweb.be](https://www.immoweb.be/en/search/house/for-sale).

__2. Open the Python script:__
Download the Pipeline_Regression file which can be found in the `src` folder. Herein, you can introduce new prediction modeling techniques to optimize the corresponding results.

__3. Running the script:__
Lastly, you can run the script by running the main.py file, to view the results.

## Interactive API Documentation
<img src="assets/fastapi-logo.png" alt="fastapi-logo" height="60" />

To create an interacive tool which presents the valuation of a house, FastApi was used to showcase the individual results (see below images as examples).

![Alt text](<Screenshot 2023-07-28 at 14.23.38.png>)

![Alt text](<Screenshot 2023-07-28 at 14.23.47.png>)

## Time Schedule

This project took place between:

- Data Exploration: 10/07/2023-14/07/2023
- Prediction Modeling: 17/07/2023-20/07/2023
- FastApi: 26/07/2023-28/07/2023




