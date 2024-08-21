import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

def extraer_tabla_oracle(name, IP, us, con, consulta, chunksize=100000):
    username = us
    password = con
    connection_str = f'oracle+cx_oracle://{username}:{password}@{name}:{IP}/?service_name=dawh'
    engine = create_engine(connection_str)
    return pd.read_sql(consulta, con=engine, chunksize=chunksize)

def preprocesar_datos(df, label_encoder_loc=None, label_encoder_sku=None):
    df['id_tie_fecha'] = pd.to_datetime(df['id_tie_fecha'])
    df['year'] = df['id_tie_fecha'].dt.year
    df['month'] = df['id_tie_fecha'].dt.month
    df['day'] = df['id_tie_fecha'].dt.day
    df['day_of_week'] = df['id_tie_fecha'].dt.dayofweek
    df['desc_cli_localidad'] = df['desc_cli_localidad'].str.strip()  # Eliminar espacios adicionales

    if label_encoder_loc is None:
        label_encoder_loc = LabelEncoder()
        df['localidad_encoded'] = label_encoder_loc.fit_transform(df['desc_cli_localidad'])
    else:
        loc_labels = df['desc_cli_localidad'].unique()
        new_labels = [label for label in loc_labels if label not in label_encoder_loc.classes_]
        if new_labels:
            label_encoder_loc.classes_ = np.append(label_encoder_loc.classes_, new_labels)
        df['localidad_encoded'] = label_encoder_loc.transform(df['desc_cli_localidad'])

    if label_encoder_sku is None:
        label_encoder_sku = LabelEncoder()
        df['sku_encoded'] = label_encoder_sku.fit_transform(df['id_sku'])
    else:
        sku_labels = df['id_sku'].unique()
        new_labels = [label for label in sku_labels if label not in label_encoder_sku.classes_]
        if new_labels:
            label_encoder_sku.classes_ = np.append(label_encoder_sku.classes_, new_labels)
        df['sku_encoded'] = label_encoder_sku.transform(df['id_sku'])

    return df, label_encoder_loc, label_encoder_sku

def entrenar_modelo_parcial(df, modelo):
    X = df[['year', 'month', 'day', 'day_of_week', 'localidad_encoded', 'sku_encoded']]
    y = df['vtas_neta_tns']
    modelo.fit(X, y)  # Entrenamiento del modelo con los datos del trozo actual
    return modelo

def guardar_modelo(modelo, nombre):
    joblib.dump(modelo, nombre)

def cargar_modelo(nombre):
    return joblib.load(nombre)

def validar_modelo(modelo, df_validacion):
    X_validacion = df_validacion[['year', 'month', 'day', 'day_of_week', 'localidad_encoded', 'sku_encoded']]
    y_validacion = df_validacion['vtas_neta_tns']
    
    df_validacion['prediccion'] = modelo.predict(X_validacion)
    df_validacion['error'] = df_validacion['vtas_neta_tns'] - df_validacion['prediccion']
    
    return df_validacion

def predecir_ventas(modelo, fecha, localidad, sku, label_encoder_loc, label_encoder_sku):
    localidad = localidad.strip()  # Eliminar espacios adicionales de la entrada

    if localidad not in label_encoder_loc.classes_:
        raise ValueError(f"Localidad desconocida: {localidad}")
    
    if sku not in label_encoder_sku.classes_:
        raise ValueError(f"SKU desconocido: {sku}")

    input_data = {
        'year': [fecha.year],
        'month': [fecha.month],
        'day': [fecha.day],
        'day_of_week': [fecha.dayofweek],
        'localidad_encoded': [label_encoder_loc.transform([localidad])[0]],
        'sku_encoded': [label_encoder_sku.transform([sku])[0]]
    }
    input_df = pd.DataFrame(input_data)
    
    prediccion = modelo.predict(input_df)
    return prediccion[0]
