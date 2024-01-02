from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# import lib for datapipeline and training
import pandas as pd
from scipy.stats import yeojohnson
import sklearn.preprocessing as preproc
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# task_id/name
data_preparation = 'data_preparation'
bus_training = 'training_model_business'
eco_training = 'training_model_economy'


def data_pre(ti, path="/opt/airflow/data/raw_data/Clean_Dataset.csv"):
    '''
    This function will prepare our dataset for training model.

    Parameters
    ----------
    path : str, optional
        dataset directory, by default "/opt/airflow/data/Clear_Dataset.csv"
    '''
    df = pd.read_csv(path, index_col=0)
    # Encode the ordinal variables "stops" and "class".
    df["stops"] = df["stops"].replace({'zero': 0, 'one': 1,
                                       'two_or_more': 2}).astype(int)
    df["class"] = df["class"].replace({'Economy': 0,
                                       'Business': 1}).astype(int)
    dummies_variables = ["airline", "source_city",
                         "destination_city", "departure_time",
                         "arrival_time"]
    # one-hot encoding
    dummies = pd.get_dummies(df[dummies_variables])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(["flight", "airline", "source_city", "destination_city",
                  "departure_time", "arrival_time"], axis=1)
    print(df.head())
    # apply transformer for normal distribution
    col = "duration"
    y_value, _ = yeojohnson(df[col])
    df[col] = y_value
    # Standardization
    cols = ["duration", "days_left"]
    df[cols] = preproc.StandardScaler().fit_transform(df[cols])
    print(df.head())

    # outlier economy
    price = df[df['class'] == 0].price
    lower_limit = price.mean() - 3*price.std()
    upper_limit = price.mean() + 3*price.std()
    print("economy: ")
    print(lower_limit)
    print(upper_limit)
    # economy class data index
    cls_eco = df[(df['class'] == 0) &
                 (df['price'] >= lower_limit) &
                 (df['price'] <= upper_limit)].index
    # outlier business
    price = df[df['class'] == 1].price
    lower_limit = price.mean() - 3*price.std()
    upper_limit = price.mean() + 3*price.std()
    print("Business:")
    print(lower_limit)
    print(upper_limit)
    # business class data index
    cls_bsn = df[(df['class'] == 1) &
                 (df['price'] >= lower_limit) &
                 (df['price'] <= upper_limit)].index

    df.iloc[cls_eco].to_csv('/opt/airflow/data/feature/eco_features.csv', index=False)
    df.iloc[cls_bsn].to_csv('/opt/airflow/data/feature/bus_features.csv', index=False)

    feature_dic = {"business": "/opt/airflow/data/feature/bus_features.csv",
                   "economy": "/opt/airflow/data/feature/eco_features.csv"}
    ti.xcom_push(key='feature_dir', value=feature_dic)
    return True


def model_train_economy(ti):
    dir_dic = ti.xcom_pull(key='feature_dir')
    filename = dir_dic['economy']
    feature = pd.read_csv(filename)
    target = feature.pop("price")
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, random_state=1, test_size=0.3,
        shuffle=True)

    model = KNeighborsRegressor()
    trained_model = model.fit(x_train, y_train)
    y_pred = trained_model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"economy mean_absolute_error: {mae}")


def model_train_business(ti):
    dir_dic = ti.xcom_pull(key='feature_dir')
    filename = dir_dic['business']
    feature = pd.read_csv(filename)
    target = feature.pop("price")
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, random_state=1, test_size=0.3,
        shuffle=True)

    model = KNeighborsRegressor()
    trained_model = model.fit(x_train, y_train)
    y_pred = trained_model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"economy mean_absolute_error: {mae}")


# dag code
airline_dag = DAG(
    "Airline_ticket_price_prediction",
    schedule_interval=None,
    start_date=datetime(2024, 1, 2)
)


with airline_dag:
    data_preparation_task = PythonOperator(
        task_id=data_preparation,
        python_callable=data_pre,
        provide_context=True
    )

    bus_training_task = PythonOperator(
        task_id=bus_training,
        python_callable=model_train_business,
        provide_context=True
    )

    eco_training_task = PythonOperator(
        task_id=eco_training,
        python_callable=model_train_economy,
        provide_context=True
    )

    data_preparation_task >> [bus_training_task, eco_training_task]
