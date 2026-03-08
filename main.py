import streamlit as st
import pandas as pd
import os
from src.model_controller import ModelController

st.title("Proyecto 2 machine learning no supervisado")

path_to_excel = os.path.join('data', 'textos.xlsx')
input_df = pd.read_excel(path_to_excel)
controller = ModelController()

st.caption("▶️ puedes probar con textos pre escritos")


if input_df is not None:
    st.caption("✅ Estos son tus datos")
    event = st.dataframe(
        input_df.drop(input_df.columns[1], axis=1, inplace=False),
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
    )
    st.caption("▶️ Selecciona una fila")
    

    if event is not None and event.selection.rows:
        current_row_index = event.selection.rows[0]
        current_row = input_df.iloc[[current_row_index]]
                
        X = current_row['textos'].values[0]
        Y = current_row['ODS'].values[0]

        st.caption("Texto Seleccionado")
        st.text(X)
        Y_pred = controller.predict([X])


        class_name = controller.name_mapper(Y_pred[0])

        col1, col2 = st.columns([1, 2])

        with col1:
            st.caption("🎯 Your predicción")
            st.text(class_name)

        with col2:
            st.caption("🎯 Tus resultados")
            #TO-DO
            st.metric("Real", Y)
            st.metric("Predicción", Y_pred[0])


st.caption("▶️ o puedes probar con tu propio texto")
text = st.text_area("escribe el texto que quieras clasificar")


if text is not None:
    ods_number_predicted = controller.predict([text])
    label_number_predicted = controller.name_mapper(ods_number_predicted[0])

    st.subheader("Resultado de la predicción")
    st.success(f"ODS predicho: {ods_number_predicted[0]}")
    st.info(f"Etiqueta: {label_number_predicted}")