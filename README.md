# HW-5-Product-demand-data-set
repository created for HW5: forecasting models
Notebook: LabProject#5_Product-demand-Forecasting.ipynb Purpose: Deduced Additional Information and ARIMA time series forecasting to
predict sales/demand for wahrehouseA.

Loaded the Historical Product Demand.csv in a dataframes.

* **Additional Information #1: To find total number of products that belong to each category**.
* **Viusalization:** Plotted a graph, number of products against each category of products
* Converted the above new data into a new data frame (df_prod_ProductPerCategory).
* Saved the resulted dataframe in a csv (AdditionalInfo#1_NoOfProductEachCategory.csv.csv)

* **Additional Information #2: To find total number of product category that belong to each warehouse.**
* Converted the above new data into a dataframe (df_prod_CategoryPerWHouse).
* **Viusalization:** Plotted a graph, number of category that belongs to each warehouse.

* **Additional Information #3: For each Warehouse to find corresponsing Product_Category and for each product category**
  **find total product. **
* Converted the above new data into a dataframe (df_prod_WHouse_Category_product)
* Saved the resulted dataframe in a csv (AdditionalInfo#3_WHouse_Category_product.csv.csv)

* **Additional Information #4: To find the date on which maximum and minimum orders where placed.**
* Converted the above new data into a dataframe (df_prod_OrderPerDay)
* **Viusalization:** Date wise order demand distribution

* **Additional Information #5: Created 3 new columns namely year, month and date for each of the order placed.**

* **Additional Information #6: To find for each year how many orders were placed. **
* Converted the above new data into a dataframe (df_prod_new_Year )
* **Viusalization:** Year wise order demand distribution

* **Additional Information #7: To find for each month how many orders were placed. **
* Converted the above new data into a dataframe (df_prod_new_Month )
* **Viusalization:** Month wise order demand distribution

* **Additional Information #8: To find for each day how many orders were placed. **
* Converted the above new data into a dataframe (df_prod_new_date )
* **Viusalization:** Day wise order demand distribution

**My findings -->**

*  There are total 2160 unique products.
# The most popular product is Product_1359 with 16936 frequency.
# Only 4 unqique warehouses
# All in all total 33 unique Product_Category

* On dates 2013-04-17, 2016-11-25, 2014-04-16, 2009-01-02, 2011-04-13	maximum number of 15 matches were played.
* For season 2014 and 2016 maximum number of matches were playes i.e. 1319 matches.
* Among all the team1, DNA has the maximum mean score of 125.132653
* Among all the team2, WSA has the maximum mean score of 120.421053
* In year 2012, maximum number of 1474 matches were played.
* For all the seasons, the maximum number of matches were played in March i.e. 11877 matches.
* Maximum number of matches were played/held during start or end of the month.

Random Forest: Regression Analysis

* In order to apply random forest, changed the datatype of team1 and team2 column (object type) to int
* Find the labels and stored them separately i.e. the score we wanted to predict.
* Remove the labels from the features
* Saving feature names for later use.
* To convert the dataframe to numpy array.
* Used Skicit-learn to split data into training and testing sets.
* Imported the random forest model.
* Instantiated the model with 1000 decision trees
* Trained the model on training data
* Used the forest's predict method on the test data
* Calculated the absolute errors
* Printed the mean absolute error (mae) i.e. 5.16 degree
* Calculated mean absolute percentage error (MAPE)

*Accuracy: 94.71 %.*
