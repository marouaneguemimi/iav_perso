import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc # Garbage Collector
import os
import sys
path = ""

# Chargement des données
# Check if files exist in 'data' directory, otherwise use kagglehub path
data_dir = 'data' if os.path.exists('data') else path

train = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=['date'])
test = pd.read_csv(os.path.join(data_dir, 'test.csv'), parse_dates=['date'])
stores = pd.read_csv(os.path.join(data_dir, 'stores.csv'))
oil = pd.read_csv(os.path.join(data_dir, 'oil.csv'), parse_dates=['date'])
transactions = pd.read_csv(os.path.join(data_dir, 'transactions.csv'), parse_dates=['date'])
holidays = pd.read_csv(os.path.join(data_dir, 'holidays_events.csv'), parse_dates=['date'])

# Préparation initiale et fusion
train['is_train'] = 1
test['is_train'] = 0
test['sales'] = np.nan
data = pd.concat([train, test], sort=False).reset_index(drop=True)


# Traitement du prix du pétrole (interpolation)
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
full_dates = pd.date_range(start=data['date'].min(), end=data['date'].max())
oil = oil.set_index('date').reindex(full_dates)
oil.index.name = 'date'
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
oil = oil.reset_index()

# Fusions de base
data = data.merge(stores, on='store_nbr', how='left')
data = data.merge(oil, on='date', how='left')

# Gestion des jours fériés
holidays_events = holidays[holidays['transferred'] == False]

# Séparation par portée (National, Regional, Local)
holidays_nat = holidays_events[holidays_events['locale'] == 'National'][['date', 'type']].rename(columns={'type': 'national_holiday_type'}).drop_duplicates(subset=['date'])
holidays_reg = holidays_events[holidays_events['locale'] == 'Regional'][['date', 'locale_name', 'type']].rename(columns={'locale_name': 'state', 'type': 'regional_holiday_type'})
holidays_loc = holidays_events[holidays_events['locale'] == 'Local'][['date', 'locale_name', 'type']].rename(columns={'locale_name': 'city', 'type': 'local_holiday_type'})

# Fusions des jours fériés
data = data.merge(holidays_nat, on='date', how='left')
data = data.merge(holidays_reg, on=['date', 'state'], how='left')
data = data.merge(holidays_loc, on=['date', 'city'], how='left')

# Création du flag is_holiday et nettoyage
data['is_holiday'] = (
    data['national_holiday_type'].notna() |
    data['regional_holiday_type'].notna() |
    data['local_holiday_type'].notna()
).astype(int)

data.drop(columns=['national_holiday_type', 'regional_holiday_type', 'local_holiday_type'], inplace=True)

# Vérification rapide
print(data[data['is_holiday'] == 1][['date', 'store_nbr', 'city', 'state', 'is_holiday']].head())


# Features temporelles basiques
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['dayofweek'] = data['date'].dt.dayofweek
data['weekofyear'] = data['date'].dt.isocalendar().week.astype(int)
data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
data['is_payday'] = ((data['date'].dt.is_month_end) | (data['date'].dt.day == 15)).astype(int)

# Intégration des transactions
data = data.merge(transactions, how="left", on=["date", "store_nbr"])

# Imputation des transactions manquantes
data.loc[data['transactions'].isnull() & (data['sales'] == 0), 'transactions'] = 0
data['transactions'] = data['transactions'].fillna(data.groupby(['store_nbr'])['transactions'].transform('mean'))

# Création des features avancées (Lags & Rolling)
print("Création des caractéristiques temporelles (ventes et transactions)...")

data.sort_values(by=['store_nbr', 'family', 'date'], inplace=True)
SHIFT_DAYS = 16

# Groupers
trans_grouper = data.groupby(['store_nbr', 'family'])['transactions']
sales_grouper = data.groupby(['store_nbr', 'family'])['sales']

# Séries décalées de base
shifted_transactions = trans_grouper.shift(SHIFT_DAYS)
base_shifted_sales = sales_grouper.shift(SHIFT_DAYS)

# --- Features Transactions ---
data[f'transactions_lag_{SHIFT_DAYS}'] = shifted_transactions
data['transactions_lag_28'] = trans_grouper.shift(28)

data['transactions_roll_mean_7'] = shifted_transactions.rolling(7).mean()
data['transactions_roll_mean_28'] = shifted_transactions.rolling(28).mean()
data['transactions_roll_std_7'] = shifted_transactions.rolling(7).std()


# --- Features Ventes ---
data[f'sales_lag_{SHIFT_DAYS}'] = base_shifted_sales
data['sales_lag_23'] = base_shifted_sales.shift(7)
data['sales_lag_30'] = base_shifted_sales.shift(14)
data['sales_lag_44'] = base_shifted_sales.shift(28)
data['sales_lag_380'] = base_shifted_sales.shift(364)

data['sales_roll_7'] = base_shifted_sales.rolling(window=7, min_periods=1).mean()
data['sales_roll_14'] = base_shifted_sales.rolling(window=14, min_periods=1).mean()
data['sales_roll_28'] = base_shifted_sales.rolling(window=28, min_periods=1).mean()

# Nettoyage final des NaN générés par les lags
data.fillna(0, inplace=True)
print("Traitement terminé.")

data.to_csv('data/processed_data.csv', index=False)
print("Data saved successfully!")

