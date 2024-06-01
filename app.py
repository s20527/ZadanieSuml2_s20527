import streamlit as st
import pickle
import numpy as np
from datetime import datetime

startTime = datetime.now()

filename = "model_s20527.h5"
model = pickle.load(open(filename, 'rb'))

sex_d = {0: "Kobieta", 1: "Mężczyzna"}
pclass_d = {1: "Pierwsza", 2: "Druga", 3: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}

def main():
    st.set_page_config(page_title="Przeżyłbyś katastrofę tytanika?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with overview:
        st.title("Przeżyłbyś katastrofę tytanika?")
        st.write("Przygotowane przez Norbert Isański s20527")
        st.image("titanic-sinking.jpg", caption='Tonący tytanic', use_column_width=True)

    with left:
        age_slider = st.slider("Wiek:", min_value=1, max_value=100)
        fare_slider = st.slider("Cena biletu:", min_value=0, max_value=500, step=10)
        sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera:", min_value=0, max_value=8)
        parch_slider = st.slider("Liczba rodziców i/lub dzieci:", min_value=0, max_value=6)

    with right:
        sex_radio = st.radio("Płeć:", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio("Klasa:", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embarked_radio = st.radio("Port zaokrętowania:", list(embarked_d.keys()), index=2, format_func=lambda x: embarked_d[x])

    if st.button('Przewiduj'):
        data = np.array([[pclass_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio, sex_radio]])

        st.header("Dodatkowe informacje:")
        st.write("Dane wejściowe:", data)

        try:
            survival = model.predict(data)
            s_confidence = model.predict_proba(data)
            
            st.write("Wynik predykcji:", survival)
            st.write("Prawdopodobieństwa:", s_confidence)

            with prediction:
                if survival[0] == 1:
                    st.subheader("Tak")
                    st.image("plywanie.jpg", use_column_width=True)
                    st.balloons()
                else:
                    st.subheader("Nie")
                    st.image("drown.jpg", use_column_width=True)
                    st.snow()
                st.subheader("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival[0]] * 100))
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {e}")

if __name__ == "__main__":
    main()

## Źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic), zastosowane przez Norbert Isański