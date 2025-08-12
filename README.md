# Clasificador de Opiniones

---

## Índice

- Descripción del Proyecto 
- Estado del Proyecto
- Demostración de Funciones y Aplicaciones  
- Acceso al Proyecto]
- Tecnologías Utilizadas

---

## Descripción del Proyecto

Este proyecto implementa un **clasificador de opiniones en tweets** mediante técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático. Utiliza limpieza y tokenización de texto, vectorización TF-IDF, balanceo de datos con SMOTE y un modelo de Regresión Logística para predecir si un tweet tiene una opinión positiva o negativa.

---

## Estado del Proyecto

- Código funcional para entrenamiento y evaluación del modelo.
- Función integrada para predecir etiquetas de nuevos tweets.
- Modelo y vectorizador guardados para uso futuro.
- No se han implementado interfaces gráficas ni despliegue en producción.

---

## Demostración de Funciones y Aplicaciones

- Limpieza y normalización de texto de tweets (eliminación de menciones, URLs y caracteres especiales).
- Tokenización y filtrado de palabras vacías (stopwords).
- Vectorización con TF-IDF para convertir texto en datos numéricos.
- Balanceo de clases usando SMOTE para mejorar el entrenamiento.
- Entrenamiento de modelo de Regresión Logística con evaluación detallada.
- Predicción en tiempo real de la polaridad de nuevos textos mediante función `predecir_tweet(texto, modelo, vectorizer)`.

Ejemplo de uso para predicción:

from joblib import load

modelo = load("modelo_logistico_tweets.pkl")
vectorizer = load("vectorizador_tfidf.pkl")

texto = "I love this product! It works great."
etiqueta = predecir_tweet(texto, modelo, vectorizer)
print("Predicción:", etiqueta)
