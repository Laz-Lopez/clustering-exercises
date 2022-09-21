import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from env import host, user, password





def get_db_url(db_name):
    """
    This function uses my env file to get the url to access the Codeup database.
    It takes in a string identifying the database I want to connect to.
    """
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"




def get_data_from_sql(str_db_name, query):
    """
    This function takes in a string for the name of the database I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    """
    df = pd.read_sql(query, get_db_url(str_db_name))
    return df




def get_new_zillow_data():
    """Returns a dataframe of all 2017 properties that are Single Family Residential"""

    sql = """
    select 
    bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    where propertylandusedesc = "Single Family Residential"
    """
    return pd.read_sql(sql, get_db_url("zillow"))