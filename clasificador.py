import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Descargar recursos nltk (solo la primera vez)
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Cargar datos
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(url)


# Función para limpiar tweets
def limpiar_tweet(texto):
    texto = texto.lower()
    texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)  # eliminar menciones
    texto = re.sub(r"http\S+", "", texto)  # eliminar URLs
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)  # conservar solo letras y espacios
    texto = re.sub(r"\s+", " ", texto).strip()  # eliminar espacios extra
    return texto


# Aplicar limpieza
df["tweet_limpio"] = df["tweet"].apply(limpiar_tweet)


# Función para tokenizar y filtrar stopwords (tokenizador simple)
def tokenizar_y_filtrar(texto):
    palabras = re.findall(r"\b[a-záéíóúüñ]+\b", texto.lower())
    palabras_filtradas = [w for w in palabras if w not in stop_words]
    return " ".join(palabras_filtradas)


# Aplicar tokenización y filtrado
df["tokens"] = df["tweet_limpio"].apply(tokenizar_y_filtrar)

# Unir tokens en strings para vectorización
df["tokens_joined"] = df["tokens"]  # ya es string
df = df[df["tokens_joined"].str.strip() != ""]  # eliminar filas vacías si hay

# Vectorización TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["tokens_joined"])

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, df["label"], test_size=0.2, random_state=42
)

# Balanceo con SMOTE solo en conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Entrenamiento de Regresión Logística
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train_smote, y_train_smote)

# Evaluación
y_pred = modelo.predict(X_test)
print("Exactitud:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))


# Función para predecir la etiqueta de nuevos tweets
def predecir_tweet(texto, modelo, vectorizer):
    texto_limpio = limpiar_tweet(texto)
    texto_tokenizado = tokenizar_y_filtrar(texto_limpio)
    texto_vectorizado = vectorizer.transform([texto_tokenizado])
    prediccion = modelo.predict(texto_vectorizado)
    return prediccion[0]


# Guardar modelo y vectorizador para uso futuro
joblib.dump(modelo, "modelo_logistico_tweets.pkl")
joblib.dump(vectorizer, "vectorizador_tfidf.pkl")
