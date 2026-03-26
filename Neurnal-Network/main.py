# =============================================================
#  ACTIVIDAD - Clasificación de Imágenes con MLP
#  Dataset : MNIST (dígitos del 0 al 9)
#  Pasos   : 1, 2 y 3
# =============================================================
#
#  ESTRUCTURA DE ARCHIVOS QUE CREA ESTE SCRIPT:
#
#    proyecto/
#    ├── main.py                        ← este archivo
#    ├── graficas/
#    │   ├── mlp_size_vs_accuracy.png   ← gráfica tamaño vs accuracy
#    │   └── mlp_curvas_entrenamiento.png
#    └── train/
#        └── mnist_train.npz            ← dataset de entrenamiento
#
#  INSTALACIÓN (solo la primera vez, en tu terminal):
#    pip install tensorflow matplotlib numpy
#
#  DATASET:
#    La primera vez, Keras descarga MNIST automáticamente desde internet.
#    El script lo guarda en la carpeta train/ como mnist_train.npz.
#    Las siguientes veces lo carga directo de ese archivo (sin internet).
#
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ── Semilla para resultados reproducibles ─────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Crear las carpetas si no existen ──────────────────────────
# os.makedirs crea la carpeta (y subcarpetas) de forma segura.
# exist_ok=True evita error si la carpeta ya existe.
os.makedirs("graficas", exist_ok=True)
os.makedirs("train",    exist_ok=True)

print("Carpetas listas: graficas/ y train/")


# ==============================================================
# PASO 1 — Cargar el dataset y dividirlo 70% train / 30% test
# ==============================================================
print("\n" + "="*55)
print("PASO 1: Cargando MNIST y dividiendo el dataset")
print("="*55)

RUTA_NPZ = os.path.join("train", "mnist_train.npz")

if os.path.exists(RUTA_NPZ):
    # ── Si el archivo ya existe, lo cargamos directamente ─────
    print("  Cargando desde archivo local:", RUTA_NPZ)
    datos = np.load(RUTA_NPZ)
    x_todo = datos["imagenes"]
    y_todo  = datos["etiquetas"]
else:
    # ── Si no existe, lo descargamos con Keras ─────────────────
    print("  Descargando MNIST por primera vez (requiere internet)...")
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.mnist.load_data()

    # Unimos los 70,000 ejemplos (60k train + 10k test de Keras)
    x_todo = np.concatenate([x_train_orig, x_test_orig], axis=0)
    y_todo  = np.concatenate([y_train_orig, y_test_orig], axis=0)

    # Guardamos en train/mnist_train.npz para usos futuros
    np.savez(RUTA_NPZ, imagenes=x_todo, etiquetas=y_todo)
    print("  Dataset guardado en:", RUTA_NPZ)

print(f"\n  Total de imágenes : {len(x_todo)}")
print(f"  Tamaño de cada img: {x_todo[0].shape}  (28×28 píxeles, escala de grises)")
print(f"  Clases            : 10  (dígitos del 0 al 9)")

# ── Preprocesamiento ──────────────────────────────────────────
# 1. Normalizar píxeles de [0, 255] → [0.0, 1.0]
#    Esto hace que la red aprenda más rápido y de forma más estable.
x_todo = x_todo.astype("float32") / 255.0

# 2. Aplanar cada imagen de 28×28 a un vector de 784 valores.
#    Los MLP necesitan entrada 1D, no una matriz 2D.
x_todo_flat = x_todo.reshape(len(x_todo), -1)   # forma: (70000, 784)

# ── Split 70 / 30 ─────────────────────────────────────────────
split   = int(len(x_todo_flat) * 0.70)   # 49,000 para entrenar
x_train = x_todo_flat[:split]
y_train = y_todo[:split]
x_test  = x_todo_flat[split:]
y_test  = y_todo[split:]

print(f"\n  Split 70/30 aplicado:")
print(f"    Entrenamiento : {len(x_train)} imágenes")
print(f"    Prueba        : {len(x_test)}  imágenes")


# ==============================================================
# FUNCIÓN — Construir un MLP
# ==============================================================
def crear_mlp(neuronas_por_capa, nombre="MLP"):
    """
    Construye un MLP (red neuronal densa) con las capas indicadas.

    neuronas_por_capa : lista con neuronas por capa oculta.
                        Ejemplo: [512, 256, 128]
    nombre            : nombre del modelo

    Cada capa oculta usa:
      - Dense     : la capa de neuronas con activación ReLU
      - Dropout   : apaga el 30% de neuronas al azar durante el
                    entrenamiento para evitar memorizar los datos
    La capa de salida tiene 10 neuronas (una por dígito) con
    softmax, que convierte la salida en probabilidades (suman 1).
    """
    model = keras.Sequential(name=nombre)
    model.add(keras.layers.Input(shape=(784,)))

    for n in neuronas_por_capa:
        model.add(keras.layers.Dense(n, activation="relu"))
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


# ==============================================================
# FUNCIÓN — Compilar y entrenar un modelo
# ==============================================================
def entrenar(model, x_tr, y_tr, x_te, y_te, epocas=30, batch=128):
    """
    Compila, entrena y evalúa el modelo.

    epocas : vueltas completas sobre el dataset de entrenamiento
    batch  : imágenes que se procesan juntas en cada paso interno

    EarlyStopping para el entrenamiento si la accuracy de validación
    no mejora en 5 épocas consecutivas, y restaura el mejor modelo.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    parar_temprano = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    historial = model.fit(
        x_tr, y_tr,
        epochs=epocas,
        batch_size=batch,
        validation_split=0.1,
        callbacks=[parar_temprano],
        verbose=1
    )

    _, accuracy = model.evaluate(x_te, y_te, verbose=0)
    parametros  = model.count_params()
    return historial, accuracy, parametros


# ==============================================================
# PASO 2 — MLP Completo (modelo base al 100%)
# ==============================================================
print("\n" + "="*55)
print("PASO 2: Entrenando MLP — Modelo Completo (100%)")
print("="*55)

ARQUITECTURA_BASE = [512, 256, 128]

modelo_100 = crear_mlp(ARQUITECTURA_BASE, nombre="MLP_100pct")
modelo_100.summary()

hist_100, acc_100, params_100 = entrenar(
    modelo_100, x_train, y_train, x_test, y_test
)
print(f"\n  Accuracy modelo 100% : {acc_100*100:.2f}%")
print(f"  Parámetros totales   : {params_100:,}")


# ==============================================================
# PASO 3a — Reducir el modelo un 25% (nos quedamos con el 75%)
# ==============================================================
print("\n" + "="*55)
print("PASO 3a: MLP con 75% del tamaño  (-25% de neuronas)")
print("="*55)

# 512→384 · 256→192 · 128→96
arq_75 = [int(n * 0.75) for n in ARQUITECTURA_BASE]
print(f"  Neuronas por capa: {arq_75}")

modelo_75 = crear_mlp(arq_75, nombre="MLP_75pct")
hist_75, acc_75, params_75 = entrenar(
    modelo_75, x_train, y_train, x_test, y_test
)
print(f"\n  Accuracy modelo 75% : {acc_75*100:.2f}%")
print(f"  Parámetros totales  : {params_75:,}")


# ==============================================================
# PASO 3b — Reducir el modelo un 50% (nos quedamos con el 50%)
# ==============================================================
print("\n" + "="*55)
print("PASO 3b: MLP con 50% del tamaño  (-50% de neuronas)")
print("="*55)

# 512→256 · 256→128 · 128→64
arq_50 = [int(n * 0.50) for n in ARQUITECTURA_BASE]
print(f"  Neuronas por capa: {arq_50}")

modelo_50 = crear_mlp(arq_50, nombre="MLP_50pct")
hist_50, acc_50, params_50 = entrenar(
    modelo_50, x_train, y_train, x_test, y_test
)
print(f"\n  Accuracy modelo 50% : {acc_50*100:.2f}%")
print(f"  Parámetros totales  : {params_50:,}")


# ==============================================================
# RESUMEN EN CONSOLA
# ==============================================================
print("\n" + "="*55)
print("RESUMEN — Tamaño del modelo vs Accuracy (split 70/30)")
print("="*55)
print(f"  {'Modelo':<20} {'Parámetros':>12}  {'Accuracy':>10}")
print(f"  {'-'*46}")
print(f"  {'MLP 100% (base)':<20} {params_100:>12,}  {acc_100*100:>9.2f}%")
print(f"  {'MLP 75%  (-25%)':<20} {params_75:>12,}  {acc_75*100:>9.2f}%")
print(f"  {'MLP 50%  (-50%)':<20} {params_50:>12,}  {acc_50*100:>9.2f}%")


# ==============================================================
# GRÁFICAS — se guardan en la carpeta graficas/
# ==============================================================
etiquetas  = ["100%\n(base)", "75%\n(-25%)", "50%\n(-50%)"]
accuracies = [acc_100 * 100, acc_75 * 100, acc_50 * 100]
num_params = [params_100, params_75, params_50]

# ── Gráfica 1: Tamaño del modelo vs Accuracy ──────────────────
fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("MLP en MNIST  —  Split 70% Train / 30% Test",
             fontsize=13, fontweight="bold")

ejes[0].plot(etiquetas, accuracies, marker="o", color="steelblue",
             linewidth=2.5, markersize=9)
ejes[0].axhline(y=95, color="red", linestyle="--", linewidth=1.5,
                label="Meta 95%")
for lbl, acc in zip(etiquetas, accuracies):
    ejes[0].annotate(f"{acc:.2f}%", (lbl, acc),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=11, fontweight="bold")
ejes[0].set_title("Tamaño relativo vs Accuracy")
ejes[0].set_xlabel("Tamaño del modelo")
ejes[0].set_ylabel("Accuracy en prueba (%)")
ejes[0].set_ylim(85, 102)
ejes[0].legend()
ejes[0].grid(True, alpha=0.3)

ejes[1].bar(etiquetas, num_params, color=["steelblue", "darkorange", "green"],
            width=0.5, edgecolor="black", linewidth=0.8)
for i, p in enumerate(num_params):
    ejes[1].text(i, p + 1000, f"{p:,}", ha="center", fontsize=10)
ejes[1].set_title("# Parámetros por Modelo")
ejes[1].set_xlabel("Tamaño del modelo")
ejes[1].set_ylabel("Número de parámetros")
ejes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
ruta1 = os.path.join("graficas", "mlp_size_vs_accuracy.png")
plt.savefig(ruta1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Gráfica guardada en: {ruta1}")

# ── Gráfica 2: Curvas de entrenamiento por época ──────────────
fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(hist_100.history["val_accuracy"], label="MLP 100%",
        color="steelblue",   linewidth=2)
ax.plot(hist_75.history["val_accuracy"],  label="MLP 75%",
        color="darkorange",  linewidth=2)
ax.plot(hist_50.history["val_accuracy"],  label="MLP 50%",
        color="green",       linewidth=2)
ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1.5,
           label="Meta 95%")
ax.set_title("Accuracy de Validación por Época")
ax.set_xlabel("Época")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
ruta2 = os.path.join("graficas", "mlp_curvas_entrenamiento.png")
plt.savefig(ruta2, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Gráfica guardada en: {ruta2}")

print("\nPasos 1, 2 y 3 completados.")
print(f"    Dataset en  : train/mnist_train.npz")
print(f"    Gráficas en : graficas/")