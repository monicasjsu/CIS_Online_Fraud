# CIS_Online_Fraud

Steps to run:

* 1. All input files are located in `dataset` folder
* 2. First run the `preprocess.py` to in `data_preprocessing` folder. This will generate the 
`train_cleaned.csv` and saves the file into Amazon RDS asn= a table.
* 3. All visualiations and data insights can be found in `visualizations` folder.
* 4. Feature engineering and missing value imputing strategies can be found in `feature_engineering` folder.
* 5. Export the `CreditCardPCAAutoML.csv` from the above step. 
* 6. This csv file will be used as an input file to `TransmorgrifAI` AutoML framework 
