#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


import pandas as pd

# Load the data from the uploaded file
file_path = "C:\\Users\\USER\\OneDrive\\Desktop\\retail.csv"
retail_data = pd.read_csv(file_path)

# first few rows of the dataset
retail_data.head()


# In[13]:


# statistics
retail_data.describe()


# In[14]:


# Calculating descriptive statistics for relevant columns
descriptive_stats = retail_data[['price', 'retail_price', 'units_sold', 'rating', 'rating_count', 'merchant_rating']].describe()

descriptive_stats


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Setting the aesthetics for the plots
sns.set(style="whitegrid")

# Plotting distributions for price and retail price
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(retail_data['price'], bins=30, kde=True)
plt.title('Distribution of Product Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(retail_data['retail_price'], bins=30, kde=True, color='orange')
plt.title('Distribution of Retail Prices')
plt.xlabel('Retail Price')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[15]:


# Plotting units sold vs. rating
plt.figure(figsize=(7, 5))
sns.scatterplot(data=retail_data, x='rating', y='units_sold')
plt.title('Units Sold vs. Product Rating')
plt.xlabel('Rating')
plt.ylabel('Units Sold')
plt.show()


# In[17]:


# Handling missing values
retail_data['has_urgency_banner'] = retail_data['has_urgency_banner'].fillna('Unknown')
retail_data['urgency_text'] = retail_data['urgency_text'].fillna('Unknown')
retail_data['merchant_profile_picture'] = retail_data['merchant_profile_picture'].fillna('Unknown')
retail_data['product_color'] = retail_data['product_color'].fillna(retail_data['product_color'].mode()[0])
retail_data['product_variation_size_id'] = retail_data['product_variation_size_id'].fillna(retail_data['product_variation_size_id'].mode()[0])
retail_data['origin_country'] = retail_data['origin_country'].fillna(retail_data['origin_country'].mode()[0])

# Addressing Outliers
cap_threshold = retail_data['units_sold'].quantile(0.95)
retail_data['units_sold'] = retail_data['units_sold'].clip(upper=cap_threshold)


# In[19]:


#demand forecasting model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Data Preparation
features = ['price', 'retail_price', 'rating', 'rating_count', 'uses_ad_boosts']
X = retail_data[features]
y = retail_data['units_sold']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model Evaluation
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(y_pred)
print(mae)
print(rmse)


# In[20]:


#inventory optimization model

# Assumptions for demonstration purposes
average_lead_time = 10  # days
average_holding_cost = 0.05  # per unit per day
average_ordering_cost = 50  # per order
desired_service_level = 0.95  # 95% service level

# Estimating average demand during lead time
average_demand_during_lead_time = y_train.mean() * average_lead_time

# Estimating the variability in demand (standard deviation)
demand_std = y_train.std()


# In[21]:
df.isna().sum()
# Safety Stock Calculation using z-score for 95% service level (approximately 1.645)
z_score = 1.645
safety_stock = z_score * demand_std * (average_lead_time ** 0.5)

# Reorder Point Calculation
reorder_point = average_demand_during_lead_time + safety_stock

# Select relevant features
feature_cor = ['price', 'retail_price', 'units_sold', 'uses_ad_boosts', 'rating', 'rating_count', 'units_sold']
data = df[feature_cor]

plt.figure(figsize=(16, 5))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap of the variables', fontdict={'fontsize': 12}, pad=12);

# Train a random forest regression model
model=RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred=model.predict(X_test)
# Evaluate the model
mse=mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# In[ ]:




