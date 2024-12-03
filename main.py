from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import statistics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import csv
from io import StringIO
from fastapi.responses import StreamingResponse
import io
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from contextlib import asynccontextmanager
import python_multipart

ml_models = {}

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def cost_of_car():
    data_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    data_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

    col = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque']
    col1 = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque']

    col_subset = list(data_train.columns)
    col_subset.remove('selling_price')

    data_train = data_train.drop_duplicates(keep = 'first', subset = col_subset)
    data_train = data_train.reset_index(drop = True)

    simple_imputer = SimpleImputer(strategy='median')
    std_scaler = StandardScaler()
    pipe_num = Pipeline([('imputer', simple_imputer), ('scaler', std_scaler)])

    y_train = data_train['selling_price']
    X_train = data_train.drop(columns = ['selling_price'])
    y_test = data_test['selling_price']
    X_test = data_test.drop(columns = ['selling_price'])



    x1_train = X_train.drop(col, axis=1)

    res_num = pipe_num.fit_transform(x1_train)

    res_df_num = pd.DataFrame(res_num, columns=pipe_num['scaler'].get_feature_names_out(x1_train.columns))

    s_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    pipe_cat = Pipeline([('imputer', s_imputer), ('encoder', ohe_encoder)])

    res_cat = pipe_cat.fit_transform(X_train[['fuel', 'seller_type', 'transmission', 'owner']])

    res_df_cat = pd.DataFrame(res_cat, columns=pipe_cat.get_feature_names_out())

    col_transformer = ColumnTransformer([('num_preproc', pipe_num, [x for x in X_train.columns if X_train[x].dtype!='object']), ('cat_preproc', pipe_cat, ['engine', 'max_power'])])

    res = col_transformer.fit_transform(X_train)

    res_df = pd.DataFrame(res, columns = [x.split('__')[-1] for x in col_transformer.get_feature_names_out()])

    model = Ridge()

    final_pipe = Pipeline([('preproc', col_transformer), ('model', model)])

    final_pipe.set_params().fit(X_train, y_train)

    return final_pipe

@asynccontextmanager
async def lifespan(app: FastAPI):
    clf = cost_of_car()

    ml_models["clf"] = clf

    yield

    ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {
        "Name": "Cost of car Prediction",
        "description": "This is a cost of car prediction model.",
    }

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    result = ml_models['clf'].predict(pd.DataFrame(item.__dict__, index=[0]))
    return result


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    res = []
    for i in items:
        res.append(ml_models['clf'].predict(pd.DataFrame(i.__dict__, index=[0])))

    return res

@app.post("/predict_items_csv")
def upload(file: UploadFile) -> List[float]:
    contents = file.file.read()
    buffer = StringIO(contents.decode('utf-8'))
    df = pd.read_csv(buffer, sep=';')
    buffer.close()
    file.file.close()

    res = []
    for index, row in df.iterrows():
        res.append(float(ml_models['clf'].predict(df[df.index == 0])))

    df['selling_price'] = res
    stream = io.StringIO()
    output = df.to_csv(stream, index=True)

    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    return response