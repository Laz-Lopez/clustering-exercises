import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
import env
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'




def zillow_data():
    filename = 'zillow.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        zillow_df = pd.read_sql('''SELECT 
    *
FROM
    properties_2017
        LEFT JOIN # maybe right join this one
    predictions_2017 AS pred USING (parcelid)
        LEFT JOIN
    airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN
    architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN
    buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN
    heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN
    typeconstructiontype USING (typeconstructiontypeid)
WHERE
    pred.transactiondate LIKE '2017%%'
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL;''', get_connection('zillow'))

        return zillow_df



def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)



def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['parcelid', 'num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)


def summarize(df):
    '''
    This function will take in a single argument (a pandas dataframe) and 
    output to console various statistices on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('----------------------')
    print('Dataframe head')
    print(df.head(3))
    print('----------------------')
    print('Dataframe Info ')
    print(df.info())
    print('----------------------')
    print('Dataframe Description')
    print(df.describe())
    print('----------------------')
    num_cols = [col for col in df.columns if df[col].dtypes != 'object']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('----------------------')
    print('Dataframe value counts ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('----------------------')
    print('nulls in df by column')
    print(nulls_by_col(df))
    print('----------------------')
    print('null in df by row')
    print(nulls_by_row(df))
    print('----------------------')


def keep_col(df):
    df['poolcnt'] = np.where((df['poolcnt'] == 1.0) , True , False)
    
    df['garagecarcnt'] = np.where((df['garagecarcnt'] >= 1.0) , df['garagecarcnt'] , 0)  


    df['poolcnt'] = df.poolcnt.map({True:1, False:0})
    return df 



def trim_bad_data_zillow(df):
    df = df[~(df.unitcnt > 1)]
    df = df[~(df.lotsizesquarefeet < df.calculatedfinishedsquarefeet)]
    df = df[~(df.calculatedfinishedsquarefeet < 500)]
    df = df[~(df.bathroomcnt < 1)]
    df = df[~(df.bedroomcnt < 1)]
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last')
    return df

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df


def series_upper_outliers(s, k=1.5):
  
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))




def df_upper_outliers(df, k, cols):
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        upper_bound = q3 + k * iqr
    return df.apply(lambda x: max([x - upper_bound, 0]))

def df_lower_outliers(df, k, cols):
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
    return df.apply(lambda x: min([x - lower_bound, 0]))    



def remove_outliers(df, k, cols):
  
    for col in cols:
        q1, q3 = df[col].quantile([.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr


        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
    return df    
    
def feature_engineering(df):

    df['pool_encoded'] = df.poolcnt.map({True:1, False:0})
    df['lat'] = df.latitude / 1_000_000
    df['long'] = df.longitude / 1_000_000
    
  
    df['bed_bath_ratio'] = df['bedroomcnt'] / df['bathroomcnt']
   
    df['house_age'] = 2017 - df['yearbuilt']
    
    
    
    return df


def add_upper_outlier_columns(df, k=1.5):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('float64'):
        df[col + '_outliers_upper'] = df_upper_outliers(df[col], k)
    return df


def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.70):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold) 
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold) 
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


def split_data(df):

    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                       random_state=123)
    return train, validate, test   




def plot_categorical_and_continuous_vars(df, cat_cols, cont_cols):
    for cont in cont_cols:
        for cat in cat_cols:
            fig = plt.figure(figsize= (20, 10))
            fig.suptitle(f'{cont} vs {cat}')
            
            plt.subplot(131)
            sns.violinplot(data=df, x = cat, y = cont)
           
            plt.subplot(1, 3, 3)
            sns.histplot(data = df, x = cont, bins = 50, hue = cat)
            
            plt.subplot(1, 3, 2)
            sns.barplot(data = df, x = cat, y = cont)