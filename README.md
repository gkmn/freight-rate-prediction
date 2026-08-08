# Freight Rate Prediction - ML Assessment

## Overview
This repository contains the Machine Learning project developed for the Freight Rate ML Assessment. The objective is to predict truckload freight rates based on historical data using data exploration, feature engineering, and a validated machine learning model.

## 🎥 Video Presentation
[**Click here to watch the 2-3 minute Loom Video Walkthrough**](https://www.loom.com/share/3c332eadabcb4fd0925fe061ab4fa014)

## 📁 Repository Structure
* `freight_rate.py`: The complete solution code containing data preprocessing, feature engineering, validation, and modeling.
* `requirements.txt`: The exact dependencies needed to run the solution.
* `validation_predictions.csv`: The final predictions containing exactly `load_id,predicted_rate` for the 12,000 loads.
* `december_predictions.csv`: The predictions generated for the December chart inputs.
* `Report.pdf`: The assessment report detailing the validation/split approach and the generated December prediction chart.

## 🚀 Run Instructions
To reproduce the environment and predictions:

1. Clone this repository:
   git clone <https://github.com/gkmn/freight-rate-prediction.git>
2. Install the required dependencies:
   python -m pip install -r requirements.txt
3. Place the provided data files (`train_test.csv`, `validation.csv`, `december_chart_inputs.csv`) inside a `data/` folder in the root directory.
4. Run the main pipeline script to generate the prediction CSVs:
   python final_code.py
5. (Optional) To verify the chart output using the provided scorer, run:
   python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

## 👤 Author
**Gökmen İnce**
