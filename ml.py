import pandas as pd
from funcionesML import extraer_tabla_oracle, preprocesar_datos, entrenar_modelo_parcial, guardar_modelo, cargar_modelo
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parámetros de conexión
name = 'uargrob031'
IP = 1522
us = 'consultas'
con = 'test2020'
chunksize = 100000

consulta_entrenamiento = '''WITH
                                periodo_Y_enero as(
                                    SELECT  TO_DATE('01'||'01'||TO_CHAR(sysdate, 'YYYY') , 'dd/mm/yyyy') as FECHA from dual
                                )
                            
                                    SELECT a.id_tie_fecha as id_tie_fecha,
                                            CAST(a.id_pro_producto*10 || a.id_pro_sabor as INTEGER) as id_sku,
                                            r.DESC_CLI_LOCALIDAD as desc_cli_localidad,
                                            (a.f_vta_bruta_kilos - a.f_devolucion_kilos)  as vtas_neta_tns
                                    FROM  lcomer.bt_detalle_documento a
                                    JOIN  lcomer.bt_importes_documento b
                                            ON lcomer.a.id_nro_documento = b.id_nro_documento
                                            AND a.id_doc_centro_emisor = b.id_doc_centro_emisor
                                            AND a.id_doc_empresa_emisora = b.id_doc_empresa_emisora
                                            AND a.id_doc_letra = b.id_doc_letra
                                            AND a.id_doc_planta = b.id_doc_planta
                                            AND a.id_doc_tipo_documento = b.id_doc_tipo_documento
                                            AND a.id_nro_documento = b.id_nro_documento
                                            AND a.id_tie_fecha = b.id_tie_fecha     
                                    JOIN  (SELECT id_cli_cliente, id_cli_sucursal FROM lcomer.lk_cli_sucursal) c
                                            ON b.id_cli_cliente = c.id_cli_cliente 
                                            AND b.id_cli_sucursal = c.id_cli_sucursal
                                    JOIN lcomer.lk_cli_sucursal n ON n.id_cli_cliente*10 || n.id_cli_sucursal = CAST(c.id_cli_cliente*10 || c.id_cli_sucursal as INTEGER)
                                    LEFT JOIN lcomer.lk_cli_provincia q ON n.id_cli_provincia = q.id_cli_provincia
                                    LEFT JOIN lcomer.lk_cli_localidad r ON n.id_cli_localidad = r.id_cli_localidad
                                    WHERE  a.id_pro_empresa_productora IN (2, 9)
                                            AND a.id_tie_fecha >= (SELECT * FROM periodo_Y_enero)
                            '''

# Inicializar modelo y LabelEncoders
modelo = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)
label_encoder_loc = None
label_encoder_sku = None

# Extraer y procesar datos en paralelo
df_entrenamiento_iterator = extraer_tabla_oracle(name, IP, us, con, consulta_entrenamiento, chunksize)

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    n_estimators_increment = 10
    for chunk in df_entrenamiento_iterator:
        print("inicio")
        futures.append(executor.submit(preprocesar_datos, chunk, label_encoder_loc, label_encoder_sku))
        for future in as_completed(futures):
            # Cargar el modelo previamente guardado
            try:
                modelo = cargar_modelo('modelo_entrenado.pkl')
                modelo.warm_start = True  # Asegurar que warm_start esté activado
            except FileNotFoundError:
                modelo = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)

            # Obtener el chunk de datos preprocesado
            df_chunk, label_encoder_loc, label_encoder_sku = future.result()
            print(df_chunk)
            
            # Incrementar el número de árboles y reentrenar el modelo
            modelo.n_estimators += n_estimators_increment
            modelo = entrenar_modelo_parcial(df_chunk, modelo)

            # Guardar el modelo inmediatamente después de cada reentrenamiento
            guardar_modelo(modelo, 'modelo_entrenado.pkl')
            print(" se entreno y guardo el modelo")
            futures.remove(future)

guardar_modelo(modelo, 'modelo_entrenado.pkl')
print("Modelo guardado")

modelo_cargado = cargar_modelo('modelo_entrenado.pkl')
print("Modelo cargado")

# Validar el modelo
data_validacion = {
    'id_tie_fecha': pd.to_datetime(['2024-07-17'] * 5, format='%Y-%m-%d'),
    'id_sku': [138202, 609400, 138402, 138401, 138302],
    'desc_cli_localidad': ['ESQUEL'] * 5,
    'vtas_neta_tns': [-2.88, -3.48, -2.88, -2.88, -3.08]
}
df_validacion = pd.DataFrame(data_validacion)
print("Éxito carga de datos de validación")

df_validacion, label_encoder_loc, label_encoder_sku = preprocesar_datos(df_validacion, label_encoder_loc, label_encoder_sku)
df_validacion['prediccion'] = modelo_cargado.predict(df_validacion[['year', 'month', 'day', 'day_of_week', 'localidad_encoded', 'sku_encoded']])
df_validacion['error'] = df_validacion['vtas_neta_tns'] - df_validacion['prediccion']

print(df_validacion)
