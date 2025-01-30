import sys
customer_id = 1900
csp_customerId = 3
cspId = 110
print(customer_id)
print(csp_customerId)
print(cspId)

######################### Data Importation ########################## 

import pandas as pd
import pyodbc
import sqlalchemy as sa
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
import logging
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

### Data Import ###
print('Data importation begins...')
print('Starting connection with SQL Server database')


# MSSQL connection parameters
connection_url = sa.URL.create(
    "mssql+pyodbc",
    username="adminUser",
    password="finops@123",
    host="sqlazuresingledatabaseserver.database.windows.net",
    database="IDD_PBI",
    query={"driver": "ODBC Driver 18 for SQL Server"},
)

# Create the SQLAlchemy engine
engine = sa.create_engine(connection_url)

# SQL query to fetch data
query = """
SELECT * 
FROM MC_FOCUS_OPTICS_BILLING_FORECAST 
WHERE csp_customerId = ? AND csp_customerId IN (
    SELECT csp_customerId 
    FROM MC_OPTICS_CSP_CUSTOMERS 
    WHERE customer_id = ?
);
"""

# Read data into a DataFrame using pandas
try:
    # Create an empty list to store chunks
    testl = []
    
    # Using SQLAlchemy engine directly, passing csp_customerId and customer_id as parameters
    for chunk in pd.read_sql(query, engine, params=(csp_customerId, customer_id), chunksize=50000):
        print(f"Got dataframe with {len(chunk)} rows")
        testl.append(chunk)
    
    # Concatenate all chunks into a final DataFrame
    data = pd.concat(testl, ignore_index=True)
    
    # Output the shape and a preview of the data
    print(f"Data imported successfully! Shape: {data.shape}")
    print(data.head())

except Exception as e:
    print(f"Error occurred while reading data: {e}")

# Save the DataFrame to an Excel file
output_file = "dataset.xlsx"
data.to_excel(output_file, index=False)  # index=False avoids saving the row index in the file
print(f"DataFrame has been saved to {output_file}")

data.shape
data.head()
print('Data Imported successfully !')


################################### Data Exploration ####################################
print('Data exploration begins...')
data.shape
data.columns
data.info()
data.isnull().sum()
data.isnull().sum()/data.shape[0]*100
round((data.isnull().sum()/data.shape[0])*100,2)
data.describe()
data.describe(percentiles=[.10,.25,.50,.75,.90])
print('Data Exploration successful !')

################################ Data Cleaning ###############################
print('Data cleaning begins...')
data.columns
data.isnull().sum()
unnecessary_columns = ['billing_id',
'BillingAccountId',
'BillingAccountName',
'csp_customerId',
'SkuId',
'SkuPriceId',
'SubAccountId',
'SubAccountName',
'ResourceId',
'BillingCurrency',
'ChargeFrequency',
'InvoiceIssuerName',
'Operation',
'ProviderName',
'PublisherName',
'PricingCategory',
'PricingUnit',
'ResourceName',
'ResourceType',
'ChargeDescription',
'CommitmentDiscountId',
'CommitmentDiscountName',
'CommitmentDiscountStatus',
'BillingPeriodStart',
'BillingPeriodEnd','EffectiveCost','ServiceCategory','ListCost','createdDate','ChargeClass',
'CommitmentDiscountCategory','CommitmentDiscountType','ContractedCost','ContractedUnitPrice',
'ChargeSubcategory','AvailabilityZone','ListUnitPrice','PricingQuantity']
data.drop(unnecessary_columns,axis=1,inplace=True)
data.info()
data.head()
data.isnull().sum()
data.isnull().sum()

import pandas as pd
from scipy.stats import chi2_contingency

# List of categorical columns to test
categorical_columns = ['Region','ChargeCategory', 'ServiceName', 'ConsumedUnit']

# Convert 'EffectiveCost' to binary (0 or 1) if not done already
data['EffectiveCost_binary'] = (data['BilledCost'] > 0).astype(int)

# Loop through each categorical column and perform the Chi-Square test
for col in categorical_columns:
    # Create the contingency table for the categorical feature and the binary 'EffectiveCost'
    contingency_table = pd.crosstab(data[col], data['EffectiveCost_binary'])

    # Perform the Chi-Square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

    # Print the results
    print(f"\nChi-Square Test for '{col}':")
    print(f"Chi-Square Stat: {chi2_stat}")
    print(f"P-Value: {p_val}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies:\n{expected}")

    # Interpret the p-value
    if p_val < 0.05:
        print(f"The relationship between '{col}' and 'BilledCost' is statistically significant (p-value < 0.05).")
    else:
        print(f"The relationship between '{col}' and 'BilledCost' is not statistically significant (p-value >= 0.05).")

data.isnull().sum()
import pandas as pd
data['ConsumedUnit'] = data['ConsumedUnit'].ffill()
data['ServiceName'] = data['ServiceName'].ffill()
data['Region'] = data['Region'].fillna(data['Region'].mode()[0])
data.isnull().sum()
data.columns
data.head()
data = data.drop('EffectiveCost_binary', axis=1)
data.columns
print('Data cleaning successful !')


########################### Data Analysis #################################

print('Data Analysis begins...')
data.describe(percentiles=[.10,.25,.50,.75,.90])
data.head()
data.columns

# Date vs cost
import pandas as pd

# Convert 'ChargePeriodStart' and 'ChargePeriodEnd' to datetime if not already
data['ChargePeriodStart'] = pd.to_datetime(data['ChargePeriodStart'])
data['ChargePeriodEnd'] = pd.to_datetime(data['ChargePeriodEnd'])

# Create a new column that contains the year-month for the start and end dates
data['Start_YearMonth'] = data['ChargePeriodStart'].dt.to_period('M')
data['End_YearMonth'] = data['ChargePeriodEnd'].dt.to_period('M')

# Create an empty dictionary to store Log_BilledCost for each month
log_billedcost_monthly = {}

# Loop over each row and sum the Log_BilledCost for each month in the date range
for _, row in data.iterrows():
    start_month = row['Start_YearMonth']
    end_month = row['End_YearMonth']

    # Iterate over months in the range
    for month in pd.period_range(start_month, end_month, freq='M'):
        if month not in log_billedcost_monthly:
            log_billedcost_monthly[month] = 0
        log_billedcost_monthly[month] += row['BilledCost']

# Convert the result to a DataFrame for easier visualization
log_billedcost_monthly_data = pd.DataFrame(list(log_billedcost_monthly.items()), columns=['Month', 'BilledCost'])
log_billedcost_monthly_data = log_billedcost_monthly_data.sort_values('Month')

print(log_billedcost_monthly_data)

# Date vs Log_ConsumedQuantity
# Create an empty dictionary to store Log_ConsumedQuantity for each month
log_consumedquantity_monthly = {}

# Loop over each row and sum the Log_ConsumedQuantity for each month in the date range
for _, row in data.iterrows():
    start_month = row['Start_YearMonth']
    end_month = row['End_YearMonth']

    # Iterate over months in the range
    for month in pd.period_range(start_month, end_month, freq='M'):
        if month not in log_consumedquantity_monthly:
            log_consumedquantity_monthly[month] = 0
        log_consumedquantity_monthly[month] += row['ConsumedQuantity']

# Convert the result to a DataFrame for easier visualization
log_consumedquantity_monthly_data = pd.DataFrame(list(log_consumedquantity_monthly.items()), columns=['Month', 'ConsumedQuantity'])
log_consumedquantity_monthly_data = log_consumedquantity_monthly_data.sort_values('Month')

print(log_consumedquantity_monthly_data)

# Log_ConsumedQuantity and Log_BilledCost Relationship
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation
correlation = data[['ConsumedQuantity', 'BilledCost']].corr()
print(correlation)


#ServiceName counts
# Count the occurrences of each ServiceName
service_counts = data['ServiceName'].value_counts()

# Find the most used ServiceName
most_used_service = service_counts.idxmax()
most_used_count = service_counts.max()

print(f"Most used ServiceName: {most_used_service}, Count: {most_used_count}")
print(service_counts)  # This will show the count of each service

#Region with most used Services
# Group by Region and ServiceName, then count the occurrences
region_service_counts = data.groupby(['Region', 'ServiceName']).size().reset_index(name='Count')

# Find the Region-ServiceName pair with the highest count
most_used_region_service = region_service_counts.loc[region_service_counts['Count'].idxmax()]

print(f"The Region with the most used ServiceName is {most_used_region_service['Region']} with ServiceName {most_used_region_service['ServiceName']}. Count: {most_used_region_service['Count']}")
print('Data Analysis successful !')


################################## Data Preparation ###################

print('Data preparation begins...')

import pandas as pd
import numpy as np

data['ChargePeriodStart'] = pd.to_datetime(data['ChargePeriodStart'])
data['ChargePeriodEnd'] = pd.to_datetime(data['ChargePeriodEnd'])

# Initialize an empty list to collect rows for the new DataFrame
rows = []

# Iterate through each row
for _, row in data.iterrows():
    start_date = row['ChargePeriodStart']
    end_date = row['ChargePeriodEnd']
    cost = row['BilledCost']

    # Calculate the total number of days and cost per day
    num_days = (end_date - start_date).days + 1
    daily_cost = cost / num_days

    # Generate rows for each day in the range
    for single_date in pd.date_range(start_date, end_date):
        rows.append({'Date': single_date, 'BilledCost': daily_cost})

# Create a new DataFrame from the expanded rows
expanded_data = pd.DataFrame(rows)

# Group by the date to handle duplicates and sum the daily costs
final_data = expanded_data.groupby('Date', as_index=False).sum()

print(final_data)
final_data = pd.DataFrame(final_data)

final_data.shape
final_data.head()
final_data.columns

# Filter rows where 'daily_cost' is NaN
nan_rows = final_data[final_data['BilledCost'].isna()]
# Display the rows with NaN values
print(nan_rows)

train_data = final_data[:-10]  # Training data (all data except the last 10 days)
test_data = final_data[-10:]  # Testing data (last 10 days)
print("Training Data:")
print(train_data.head())
print("Testing Data:")
print(test_data.head())

############################## Handling outliers - Feature engineering ###########
# Log Transformation
print('Feature engineering begins...')

final_data['Log_BilledCost'] = np.log1p(final_data['BilledCost'])
test_data = pd.DataFrame(final_data)
test_data.columns
prophet_data = test_data[['Date', 'Log_BilledCost']].rename(columns={'Date':'ds','Log_BilledCost': 'y'})
prophet_data.head()

print('Feature engineering successful !')

########################## Model Training ######################
print('Model Training begins...')

from prophet import Prophet
model = Prophet(daily_seasonality=True)
model.fit(prophet_data)

# Save the trained model using joblib
joblib.dump(model, 'aws_billing_forecast_model.pkl')

print("Model saved successfully!")

print('Model Training successful !')

######################### Forecasting ############################

print('Forecast begins...')
import pandas as pd
import numpy as np
import joblib  # For loading the saved model
from dateutil.relativedelta import relativedelta

model = joblib.load('aws_billing_forecast_model.pkl')
last_date = prophet_data['ds'].max()
end_date = (last_date + relativedelta(months=4)).replace(day=1) + pd.offsets.MonthEnd(0)
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=end_date, freq='D')
future = pd.DataFrame({'ds': future_dates})

forecast = model.predict(future)
forecast['ForecastedCost'] = np.expm1(forecast['yhat'])
forecasted_values = forecast[['ds', 'ForecastedCost']].rename(columns={'ds': 'date','ForecastedCost': 'predicted_cost'})
print(forecasted_values)
forecasted_values.to_csv('forecasted_values.csv', index=False)

last_3_months = final_data[final_data['Date'] >= (last_date - pd.DateOffset(months=3))]
last_3_months_renamed = last_3_months[['Date', 'BilledCost']].rename(columns={'BilledCost': 'predicted_cost','Date':'date'})
combined_data = pd.concat([last_3_months_renamed, forecasted_values], axis=0, ignore_index=True)
combined_data.to_csv('combined_forecast_and_historical.csv', index=False)
# Print the final combined data
print("Combined data (historical + forecasted):")
print(combined_data)
print('Data saved to combined_forecast_and_historical.csv')

print('Forecast Successful !')

################################### Model evaluation ###############

print('Model evaluation begins...')

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Predict using the fitted model on the test set (last 10 days)
forecast_test = model.predict(prophet_data)

# Extract actual values and predicted values for evaluation
y_true = prophet_data['y'].values  # Actual values (from the test set)
y_pred = forecast_test['yhat'].values  # Predicted values (from the forecast)

# Calculate evaluation metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_true, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

print('Model Evaluation successful !')


############################## Results to the database ###################################3
print('Forecast results saving to the database...')

##################################### with csp and customer_id in table ######################
import sqlalchemy as sa
import pandas as pd

connection_url = sa.URL.create(
    "mssql+pyodbc",
    username="adminUser",            # Your SQL Server username
    password="finops@123",           # Your SQL Server password
    host="sqlazuresingledatabaseserver.database.windows.net",  # Your SQL Server host
    database="IDD_PBI",           # Your database name
    query={"driver": "ODBC Driver 18 for SQL Server"},  # ODBC driver
)

# Create a SQLAlchemy engine
engine = sa.create_engine(connection_url)


try:
    # Step 1: Connect to the database
    with engine.connect() as connection:
        # Step 2: Iterate through the DataFrame and insert/update each row
        for _, row in combined_data.iterrows():
            # Check if the combination of date, cspId, and customer_id exists in the table
            check_query = """
            SELECT COUNT(*) 
            FROM forecast_results 
            WHERE date = :date AND cspId = :cspId AND customer_id = :customer_id;
            """
            result = connection.execute(
                sa.text(check_query),
                {
                    "date": row["date"],
                    "cspId": cspId,  # Use predefined cspId
                    "customer_id": customer_id,  # Use predefined customer_id
                },
            )
            exists = result.scalar()  # Get the count of matching rows

            if exists > 0:
                # Row exists, update the predicted_cost
                update_query = """
                UPDATE forecast_results
                SET predicted_cost = :predicted_cost
                WHERE date = :date AND cspId = :cspId AND customer_id = :customer_id;
                """
                connection.execute(
                    sa.text(update_query),
                    {
                        "date": row["date"],
                        "predicted_cost": row["predicted_cost"],
                        "cspId": cspId,  # Use predefined cspId
                        "customer_id": customer_id,  # Use predefined customer_id
                    },
                )
                print(f"Updated predicted_cost for date={row['date']}, cspId={cspId}, customer_id={customer_id}.")
            else:
                # Row doesn't exist, insert a new row
                insert_query = """
                INSERT INTO forecast_results (date, predicted_cost, cspId, customer_id)
                VALUES (:date, :predicted_cost, :cspId, :customer_id);
                """
                connection.execute(
                    sa.text(insert_query),
                    {
                        "date": row["date"],
                        "predicted_cost": row["predicted_cost"],
                        "cspId": cspId,  # Use predefined cspId
                        "customer_id": customer_id,  # Use predefined customer_id
                    },
                )
                print(f"Inserted new row for date={row['date']}, cspId={cspId}, customer_id={customer_id}.")

            # Commit the transaction
            connection.commit()

        print(f"{len(combined_data)} rows processed.")

except sa.exc.SQLAlchemyError as e:
    print(f"Error while connecting to SQL Server: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("Forecast results saved to the database successfully!")
