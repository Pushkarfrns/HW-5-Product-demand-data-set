# HW-5-Product-demand-data-set
repository created for HW5: forecasting models
Notebook: LabProject#5_Product-demand-Forecasting.ipynb Purpose: Deduced Additional Information and ARIMA time series forecasting to
predict sales/demand for wahrehouseA.

Loaded the Historical Product Demand.csv in a dataframes.

* **Additional Information #1: To find total number of products that belong to each category**.
* **Visualization:** Plotted a graph, number of products against each category of products
* Converted the above new data into a new data frame (df_prod_ProductPerCategory).
* Saved the resulted dataframe in a csv (AdditionalInfo#1_NoOfProductEachCategory.csv.csv)

* **Additional Information #2: To find total number of product category that belong to each warehouse.**
* Converted the above new data into a dataframe (df_prod_CategoryPerWHouse).
* **Visualization:** Plotted a graph, number of category that belongs to each warehouse.

* **Additional Information #3: For each Warehouse to find corresponsing Product_Category and for each product category**
  **find total product. **
* Converted the above new data into a dataframe (df_prod_WHouse_Category_product)
* Saved the resulted dataframe in a csv (AdditionalInfo#3_WHouse_Category_product.csv.csv)

* **Additional Information #4: To find the date on which maximum and minimum orders where placed.**
* Converted the above new data into a dataframe (df_prod_OrderPerDay)
* **Visualization:** Date wise order demand distribution

* **Additional Information #5: Created 3 new columns namely year, month and date for each of the order placed.**

* **Additional Information #6: To find for each year how many orders were placed.**
* Converted the above new data into a dataframe (df_prod_new_Year )
* **Visualization:** Year wise order demand distribution

* **Additional Information #7: To find for each month how many orders were placed.**
* Converted the above new data into a dataframe (df_prod_new_Month )
* **Visualization:** Month wise order demand distribution

* **Additional Information #8: To find for each day how many orders were placed.**
* Converted the above new data into a dataframe (df_prod_new_date )
* **Visualization:** Day wise order demand distribution

**My findings -->**

* There are total 2160 unique products.
* The most popular product is Product_1359 with 16936 frequency.
* Only 4 unqique warehouses
* All in all total 33 unique Product_Category
* Category_019 has maximum of 470266 produtcs.
* Whse_J has maximum and Whse_C has minimum products respectively.
* On date 2013/9/27 maximum of 2075 orders were placed.
* In year 2013, maximum number of 218298 orders were placed.
* Most/ maximum of 96619 orders were demand/placed in the month of October.
* On every 1st of month, maximum orders were demand.

**ARIMA: Time Series Forecasting**

* Forecasted for the warehouseA demand.
* Removed the unnecessary columns from the dataframe.
* Indexing with Time Series Data
* Visualizing WarehouseA Sales Time Series Data
* Visualize the data using time-series decomposition: decompose our time series into three distinct components: trend, seasonality, and noise
* ARIMA, which stands for Autoregressive Integrated Moving Average.
* ARIMA(p, d, q): parameters account for seasonality, trend, and noise in data
* parameter Selection for our warehouseA demand ARIMA Time Series Model.
* Fitting the ARIMA model
* Validating forecasts
* Mean Squared Error of our forecasts 


