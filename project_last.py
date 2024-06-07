import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt


from sklearn.neighbors import LocalOutlierFactor


from sklearn.preprocessing import RobustScaler


from sklearn.metrics import r2_score


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


from lightgbm import LGBMRegressor



import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

"""
In this study, there are three different datasets named cleandata, business, and economy. 
The provider merged the economy and business datasets with some additions and subtractions to create the cleandata dataset.

The summary of the study includes:

Accessing the cleandata from the beginning to make the data more meaningful,
Applying exploratory data analysis (EDA) on the data to understand it better,
Building a model to predict flights.

"""

# Section 1
"""
Creating cleandata from business and economy datasets and add extra features into cleandata.
"""

df_cle = pd.read_csv("MIULL/fight_pre/Clean_Dataset.csv")

df_bus = pd.read_csv("MIULL/fight_pre/business.csv")

df_eco = pd.read_csv("MIULL/fight_pre/economy.csv")

df_bus["class"] = "business"
df_eco["class"] = "economy"

df_con = pd.concat([df_eco, df_bus], axis=0)

df_cle.info()

df_con.info()

"""
There are 300,153 records in df_cle and 300,261 records in df_con. 
There is a difference of 108 records. Let's find these differences.

"""

df_cle["airline"].value_counts()
df_con["airline"].value_counts()

"""
As a result, df_con contains 61 extra records from StarAir and 41 extra records from Truejet, 
totaling 102 extra records. This means the provider deleted these values from the data.
"""

df_con = df_con[~df_con['airline'].isin(['StarAir', 'Trujet'])]
df_con.reset_index(drop=True, inplace=True)

"""
I find rest of 6 differences. I find this differences with Excel match method.
"""

df_con.iloc[[563, 6181, 96486, 104676, 111315, 154007], :]

df_con = df_con.drop([563, 6181, 96486, 104676, 111315, 154007])
df_con.reset_index(drop=True, inplace=True)
"""
I add  travel date inside to cleandata from con data.
"""
df_cle["travel_date"] = df_con["date"]

"""
Using the Travel_date column and the day_dif column, the dates of ticket purchases were obtained.
"""

df_cle[['day', 'month', 'year']] = pd.DataFrame(df_cle["travel_date"].str.split("-", expand=True).to_numpy().
                                                astype(int),columns=["day", "month", "year"])

df_cle["day"]=df_cle["day"].astype(np.int64)
df_cle["month"]=df_cle["month"].astype(np.int64)


df_cle["travel_date"] = pd.to_datetime(df_cle["travel_date"], format='%d-%m-%Y')

df_cle[df_cle["month"] == 3].sample(1)
df_cle[df_cle["month"] == 2].sample(1)

df_cle["ticket_buy_date"] = np.where(df_cle["month"] == 3, df_cle["day"] - df_cle["days_left"] + 28,
                                 np.where(df_cle["month"] == 2, df_cle["day"] - df_cle["days_left"], df_cle["day"]))

df_cle["ticket_buy_date"].value_counts()

"""
As a result, 300,153 tickets were sold on the 10th of February. This means the data is for tickets sold on the 10th of February, 2022.
"""

"""
I don't need  this values more.
"""

df_cle=df_cle.drop(["Unnamed: 0","ticket_buy_date","day","month","year"], axis=1)


"""
I add  departure time, arrival time and stop inside to cleandata from con data.
"""
df_cle[["dep_time", "arr_time","stops"]] = df_con[["dep_time", "arr_time","stop"]]

df_cle['dep_time'] = df_cle['dep_time'].apply(lambda x: x.split('-')[0])
df_cle['arr_time'] = df_cle['arr_time'].apply(lambda x: x.split('-')[0])

"""
Let's clean up the data in the "stops" column
"""

def delete_strings(main_string, sub_strings):
    for item in sub_strings:
        main_string = main_string.replace(item, '')
    return main_string


df_cle['stops'] = df_cle['stops'].apply(lambda x: delete_strings(x, ['1-stop', '\n', '\t']))

df_cle["stops"]=df_cle["stops"].replace("", "Via Unknown")
df_cle['stops'].value_counts()

"""
For easy understanding i edit data.
"""

df_cle = df_cle[["airline", "flight", "source_city","dep_time", "departure_time", "stops","arr_time","arrival_time",
         "destination_city","class", "duration", "days_left","travel_date", "price"]]

# Section 2

"""
First encounter with clean data.
"""

df=df_cle
df.columns = df.columns.str.upper()
df.head()
df.info()

###################
df_business = df[df["CLASS"] == "Business"]
df_economy = df[df["CLASS"] == "Economy"]

"""
Checking Data
"""

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
check_df(df_business)
check_df(df_economy)

####################
for column in df.columns:
    print(f"Name of feature: {column:_<18}   Number of nunique: {df[column].nunique():>7}   Dtype of feature: "
          f"{str(df[column].dtype):<10}")


"""
Grap columns for categoric and numeric
"""

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Step 3
"""
Exploratory data analysis (EDA)
"""

"""
Visualizing the numeric data for the Economy class using a histogram.
"""


plt.figure(figsize=(17, 6))
colors = ['blue', 'red', 'orange']

for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df_economy[col], kde=True, color=colors[i % len(colors)])
    plt.title("Histogram of " + col)

plt.tight_layout()
plt.show()

"""
Visualizing the numeric data for the Business class using a histogram.
"""

plt.figure(figsize=(17, 6))
colors = ['blue', 'red', 'orange']

for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df_business[col], kde=True, color=colors[i % len(colors)])
    plt.title("Histogram of " + col)

plt.tight_layout()
plt.show()

"""
Visualizing the categorical data for the Economy class using a pie graph.
"""

fig, axs = plt.subplots(2, 3, figsize=(18, 12))

for i, col in enumerate(cat_cols):
    ax = axs[i // 3, i % 3]
    df_economy[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                    colors=sns.color_palette('bright', len(df_economy[col].unique())), ax=ax)
    ax.set_ylabel('')
    ax.set_title("Distribution of " + col)

plt.tight_layout()
plt.show()

"""
Visualizing the categorical data for the Business class using a pie graph.
"""

fig, axs = plt.subplots(2, 3, figsize=(18, 12))

for i, col in enumerate(cat_cols):
    ax = axs[i // 3, i % 3]
    df_business[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                    colors=sns.color_palette('bright', len(df_business[col].unique())), ax=ax)
    ax.set_ylabel('')
    ax.set_title("Distribution of " + col)

plt.tight_layout()
plt.show()

"""
Correlation analysis for Economy class.
"""
corr = df_economy[num_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

"""
Correlation analysis for Business class.
"""

corr = df_business[num_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

"""
Mean value of Economy class prices with group by categorical columns
"""

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df_economy, "PRICE", col)

"""
Mean value of Business class prices with group by categorical columns
"""
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df_business, "PRICE", col)

"""
Mean value of Business and Economy classes prices by airline
"""

def get_mean_prices(data):
    return data.groupby("AIRLINE").agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)


merged_prices = pd.concat([get_mean_prices(df_economy), get_mean_prices(df_business)]
                          , axis=1, keys=['Economy', 'Business'])

print(merged_prices)

# Section 4

"""
Feature Engineering
"""
"""
Outlier Analysis
"""

"""
Box plot for Economy class
"""

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, len(num_cols), i)
    sns.boxplot(y=df_economy[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel('Values')
plt.tight_layout()
plt.show()

"""
Box plot for Business class
"""

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, len(num_cols), i)
    sns.boxplot(y=df_business[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel('Values')
plt.tight_layout()
plt.show()

##################

df_economy["DURATION"].sort_values(ascending=False).head(10)
df_economy["PRICE"].sort_values(ascending=False).head(10)

df_economy = df_economy.drop([193889, 194359, 119508, 193926,197809 ,94386])

#####################


df_business["DURATION"].sort_values(ascending=False).head(10)
df_business["PRICE"].sort_values(ascending=False).head(10)
df_business["PRICE"].sort_values(ascending=True).head(10)

df_business = df_business.drop([261152,293606,261377])

df = pd.concat([df_economy, df_business], axis=0).reset_index(drop=True)

"""
Missing value analysis
"""

df.isnull().values.any()


"""
Adding new columns
"""
df_economy["PRICE_RANGE"] = pd.qcut(df_economy['PRICE'], q=5, labels=['e1', 'e2', 'e3', 'e4', 'e5'])
df_business["PRICE_RANGE"] = pd.qcut(df_business['PRICE'], q=5, labels=['b1', 'b2', 'b3', 'b4', 'b5'])

df_price= pd.concat([df_economy["PRICE_RANGE"], df_business["PRICE_RANGE"]], ignore_index=True)

df_price = pd.DataFrame(df_price, columns=["PRICE_RANGE"])

df["PRICE_RANGE"]=df_price["PRICE_RANGE"]

df["PRICE_RANGE"].value_counts()

# Step 5
"""
Encoding
"""


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


ohe_cols = [col for col in df.columns if 40 >= df[col].nunique() >= 2]

df= one_hot_encoder(df, ohe_cols)
df = pd.get_dummies(df, columns=["DEP_TIME","ARR_TIME"], drop_first=True, dtype=int)

df.head(1)

# Step 6
"""
Scaling
"""



rs = RobustScaler()
df["DURATION"] = rs.fit_transform(df[["DURATION"]])
df["DAYS_LEFT"] = rs.fit_transform(df[["DAYS_LEFT"]])

df.head(1)


# Step 7
"""
Modelling
"""


y = df["PRICE"]

X = df.drop(["PRICE","FLIGHT" ,"TRAVEL_DATE"], axis = 1)
X.columns = X.columns.str.replace(".", "_")
X.columns = X.columns.str.replace(":", "_")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

"""
LightGBM
"""

lgbm_model = LGBMRegressor(verbose=-1)
lgbm_model.fit(X_train, y_train)


"""
Validation of Model Reliability
"""

cv_results = cross_validate(lgbm_model, X_train, y_train, cv=5, scoring=["r2"])

cv_results['test_r2'].mean()
# 0.9919887806442833

"""
Hyperparameter optimization
"""
lgbm_params = {
    "learning_rate": [0.09, 0.1],
    "n_estimators": [900,1000],
    "max_depth": [8,9],
    "subsample": [0.9, 1],
    "colsample_bytree": [0.9,1]
}


lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X_train, y_train)

cv_results = cross_validate(lgbm_final, X_train, y_train, cv=5, scoring=["r2"])

cv_results['test_r2'].mean()
# 0.9944007602794078

# Test verilerini tahmin etme
y_pred = lgbm_final.predict(X_test)

r2_test = r2_score(y_test, y_pred)
print("r2 Score on Test Data:", r2_test)
# r2 Score on Test Data: 0.9944582757456555


# Feature Importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:10])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X_train)

