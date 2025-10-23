# 🧠 Tarea 2: Perceptrón Multicapa (MLP) 🚀

¡Bienvenido! Este proyecto es una implementación completa de una **Red Neuronal de Perceptrón Multicapa (MLP)**, construida desde cero en Python, cumpliendo con todos los requisitos de la "Tarea 2".

El proyecto no utiliza librerías de Deep Learning como PyTorch o TensorFlow, sino que implementa la lógica de `feedforward` y `backpropagation` manualmente usando **Numpy**.

Además, incluye una **interfaz gráfica de usuario (GUI)** 🖥️ desarrollada con `CustomTkinter` que te permite crear, entrenar, guardar, cargar y probar tus redes neuronales de forma interactiva.

---

## ✨ Características Principales

Este proyecto cumple con todos los requisitos funcionales de la tarea:

* **Creación y Carga:** El programa inicia permitiéndote **crear un nuevo MLP** o **cargar un modelo entrenado** desde un archivo `.json`.
* **Arquitectura Flexible:** 🏗️ Al crear una red, puedes especificar su arquitectura completa:
    * Neuronas de entrada
    * Neuronas de salida
    * Número de capas ocultas
    * Número de neuronas por cada capa oculta
* **Entrenamiento con Backpropagation:** 🚂 El núcleo del proyecto. La red se entrena usando el algoritmo de retropropagación del error.
* **Función de Activación ReLU:** Utiliza la función **Rectified Linear Unit (ReLU)** en las capas ocultas para un entrenamiento más rápido y eficiente.
* **Monitorización en Vivo:** 📈 Durante el entrenamiento, el programa **imprime la precisión promedio** tanto con los datos de *entrenamiento* como con los de *prueba* al final de **cada época**.
* **Gestión del Modelo:** Una vez que la red está entrenada o cargada, tienes un panel de control con 3 opciones:
    1.  **Ejecutar Feedforward:** ⚡ Prueba la red cargando un archivo `.csv` de prueba.
    2.  **Seguir Entrenando:** 🔄 Continúa el entrenamiento con más épocas o nuevos archivos.
    3.  **Guardar la Red:** 💾 Almacena la arquitectura, pesos y sesgos de tu red en un archivo `.json`.
* **Lector de CSV:** Incluye un lector (`utils.py`) que procesa los archivos `.csv` donde las primeras `N-1` columnas son las entradas y la última columna es la salida esperada.

---

## 🔬 Curiosidades del Proyecto y Conceptos Clave

Este proyecto es una excelente forma de entender los pilares del Deep Learning.

### 1. ¿Por qué ReLU (Rectified Linear Unit)? ⚡
Como se implementa en `perceptron.py`, ReLU (`f(x) = max(0, x)`) es la función de activación preferida en las capas ocultas sobre opciones más antiguas (como la sigmoide).
* **Evita el "Desvanecimiento del Gradiente":** No se "satura" para valores positivos, permitiendo que el gradiente fluya mejor durante el backpropagation, lo que acelera radicalmente el aprendizaje.
* **Eficiencia Computacional:** Es una operación increíblemente simple y rápida (una simple comparación).

### 2. El Motor del Aprendizaje: Backpropagation 🧠⚙️
Implementado en `perceptron.py`, este es el algoritmo que permite a la red "aprender". Funciona calculando el error en la capa de salida y luego propagando ese error "hacia atrás", capa por capa, para determinar cuánto debe ajustarse cada peso y sesgo en la red para minimizar el error.

### 3. ¡Hecho desde Cero!
Cumpliendo con un requisito técnico clave, este proyecto **no usa PyTorch ni TensorFlow**. Toda la lógica de matrices, derivadas, propagación y actualización de pesos está hecha "a mano" con **Numpy**, lo que te obliga a entender *realmente* cómo funciona una red neuronal por dentro.

---

## 🛠️ Tecnologías y Dependencias

Este proyecto está construido 100% en Python y utiliza las siguientes librerías:

* **Numpy:** Para todos los cálculos matriciales y operaciones de la red neuronal.
* **CustomTkinter:** Para la interfaz gráfica de usuario moderna y atractiva.
* **Matplotlib:** Para la visualización de la arquitectura de la red.

---

## 🚀 Flujo de Trabajo Básico:

1.  Al iniciar, elige **"Crear un nuevo perceptrón"**.
2.  Rellena el formulario con la arquitectura (ej: 2 entradas, 1 salida, 2 capas ocultas con 4 y 3 neuronas).
3.  Serás llevado al panel principal. Haz clic en **"Entrenar Red"**.
4.  Se te pedirá que definas las épocas, la tasa de aprendizaje y la tolerancia.
5.  Carga tu archivo `.csv` de **entrenamiento** (ej: `2-d_2-class_train.csv`).
6.  Carga tu archivo `.csv` de **prueba** (ej: `2-d_2-class_test.csv`).
7.  Observa la consola: verás la precisión de entrenamiento y prueba mejorar con cada época 📈.
8.  Una vez entrenada, puedes usar **"Feedforward"** con un archivo de prueba o **"Guardar red"** para obtener tu archivo `.json`.

---

## 📂 Estructura de Archivos
├── main.py # 🚀 Punto de entrada. Lanza la GUI inicial. 

├── perceptron.py # 🧠 Clase principal del MLP (lógica de feedforward, backpropagation, ReLU). 

├── gui_controller.py # 🎮 Lógica de validación de los formularios de la GUI. 

├── gui_functions.py # ⚙️ Funciones para los botones (train, feedforward, save_mlp). 

├── utils.py # 🛠️ Funciones auxiliares (leer CSV, seleccionar archivos). 

│ ├── config/ 
│   └── config_mlp.json # 💾 Ejemplo de una red guardada (pesos y sesgos). 

│ └── CSV/ 
    ├── 2-d_2-class_train.csv # 📊 Datos de entrenamiento (2D, 2 clases). 
    ├── 2-d_2-class_test.csv # 🧪 Datos de prueba (2D, 2 clases). 
    ├── 3-d_4-class_train.csv # 📊 Datos de entrenamiento (3D, 4 clases). 
    └── 3-d_4-class_test.csv # 🧪 Datos de prueba (3D, 4 clases).
