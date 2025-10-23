import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, neuronas_entrada: int, neuronas_salida: int, capas_ocultas: int, neuronas_por_capa: list):
        # Parámetros de la Red
        self.neuronas_entrada = neuronas_entrada
        self.neuronas_salida = neuronas_salida
        self.capas_ocultas = capas_ocultas
        self.neuronas_por_capa = neuronas_por_capa
        self.neuronas_en_total = neuronas_entrada + sum(neuronas_por_capa) + neuronas_salida
        
        # Arreglo de activaciones por capa (para uso en feedforward y backpropagation)
        self.activaciones = []
        
        # --- Inicialización de Pesos y Sesgos ---
        # Creamos una lista con la arquitectura completa de la red
        # Ejemplo: [2, 3, 3, 1] para 2 entradas, 2 capas ocultas de 3 neuronas y 1 salida
        self.arquitectura = [self.neuronas_entrada] + self.neuronas_por_capa + [self.neuronas_salida]
        
        self.pesos = []
        self.sesgos = []
        
        # Iteramos a través de las capas para crear las matrices de pesos y vectores de sesgos
        for i in range(len(self.arquitectura) - 1):
            # Pesos: Matriz con dimensiones (neuronas_capa_actual, neuronas_capa_siguiente)
            w = np.random.randn(self.arquitectura[i], self.arquitectura[i+1]) * np.sqrt(2.0 / self.arquitectura[i])
            self.pesos.append(w)
            
            # Sesgos: Se inicia con cero | Vector con dimensión (neuronas_capa_siguiente,)
            b = np.full(self.arquitectura[i+1], 0.01)
            self.sesgos.append(b)

    # Función de Activación ReLU
    def relu(self, x):
        return np.maximum(0, x)

    # Derivada de ReLU para backpropagation
    def relu_backpropagation(self, a):
        return (a > 0).astype(float)

    # Propagación hacia adelante
    def feedforward(self, input_data):
        """
        Calcula la salida de la red para una entrada dada,
        procesándola capa por capa.
        """
        # Comprobar que la dimension sea la correcta
        if input_data.shape[0] != self.neuronas_entrada:
            messagebox.showerror("Error de Entrada", "Asegurate de que los datos de entrada tengan la dimensión correcta.")
            return -1
        
        # Inicialmente, la activación son los datos de entrada
        self.activaciones = [input_data]
        
        # Iteramos a través de cada capa de la red | zip() nos permite recorrer los pesos y sesgos en paralelo
        i = 0
        for w, b in zip(self.pesos, self.sesgos):
            # 1. Calcular la suma ponderada (producto punto) y añade el sesgo
            z = np.dot(self.activaciones[-1], w) + b
            
            # Si es la última capa (capa de salida), usamos activación lineal (z)
            # Si no, usamos ReLU
            if i == len(self.pesos) - 1:
                self.activaciones.append(z) # Activación lineal para la salida
            else:
                self.activaciones.append(self.relu(z)) # ReLU para capas ocultas
            
            # Incrementamos el índice de capa
            i += 1
        
        # Retornamos la salida de la red
        return self.activaciones[-1]

    # Propagación hacia atrás: Ajuste de pesos y sesgos | Aprendizaje de la red
    def backpropagation(self, y, tasa_aprendizaje):
        """
        Realiza la retropropagación del error y actualiza los pesos y sesgos
        para una ÚNICA muestra de entrenamiento (x, y).
        """
        # Obtener la SALIDA de la ÚLTIMA capa en el FEEDFORWARD
        predict_output = self.activaciones[-1]
        
        # Calculamos el ERROR en la CAPA de SALIDA
        output_error = y - predict_output
        
        # Calculamos el DELTA para la CAPA de SALIDA
        delta = output_error # Derivada de activación lineal es 1
        
        # Lista para almacenar los gradientes de todas las capas
        delta_pesos = []
        delta_sesgos = []
        
        # --- Retropropagación del error a través de las capas ---
        for L in range( len(self.arquitectura) - 1, 0, -1):
            # Índice para la listas de los pesos/sesgos
            index_L = L - 1

            # Activación anterior (la que entró a la capa L)
            activacion_anterior = self.activaciones[L - 1]
            
            # --- Calcular gradientes para la capa actual (L) ---
            gradiente_w = np.outer(activacion_anterior, delta)
            gradiente_b = delta
            
            # Almacenamos los gradientes al principio, ya que vamos de atrás hacia adelante
            delta_pesos.insert(0, gradiente_w)
            delta_sesgos.insert(0, gradiente_b)
            
            # --- Calcular Delta para la siguiente iteración (capa L-1) ---
            if L > 1:
                error_anterior = np.dot(delta, self.pesos[index_L].T)
                
                # Delta para la capa anterior: s * (1 - s), donde 's' es self.activaciones[L - 1]
                s_anterior = self.activaciones[L - 1]
                delta = error_anterior * self.relu_backpropagation(s_anterior)
        
        # --- Paso 3: Actualizar Pesos y Sesgos ---
        for i in range(len(self.pesos)):
            self.pesos[i] += delta_pesos[i] * tasa_aprendizaje
            self.sesgos[i] += delta_sesgos[i] * tasa_aprendizaje
        
    # Entrenamiento de la Red Neuronal | Recibe 'y' como la salida esperada
    def train(self, x_train, y_train, x_test, y_test, epocas, tasa_aprendizaje, tolerancia):
        historial_loss = []
        historial_precision_train = []
        historial_precision_test = []
        
        print("-------------- Iniciando Entrenamiento -----------")        
        # Bucle de épocas de entrenamiento
        for epoch in range(epocas):
            # 1. Para calcular la pérdida (loss) en esta época
            epoch_loss = []
            
            # 2. Iteramos a través de cada muestra en el dataset
            for x, y in zip(x_train, y_train):
                
                # 1. Propagación hacia adelante
                output = self.feedforward(x)
                
                # Verificación de error
                if output == -1:
                    messagebox.showerror("Error en dimensiones", "Las dimensiones de los datos de entrada no coinciden con las dimensiones esperadas por la red.")
                    return  # Salir si hubo un error de dimensión
                
                # 2. Retropropagación del error
                self.backpropagation(y, tasa_aprendizaje)
                
                # 3. Guardamos el error cuadrático medio (MSE) para esta muestra
                loss = np.mean(np.square(y - output))
                epoch_loss.append(loss)
            
            # --------------- Calculo de la pérdida promedio de la época ---------------
            # 3. Calculamos la pérdida promedio para esta época {promedio([loss_muestra1, loss_muestra2, ..., loss_muestraN])}
            avg_epoch_loss = np.mean(epoch_loss)
            
            # 3. Guardamos la pérdida en el historial
            historial_loss.append(avg_epoch_loss)
            # --------------------------------------------------------------------------
            
            # ----------- Calculo de la precisión sobre un umbral de la época ----------
            # 4. Calcular Precisión (Accuracy) en DATOS DE ENTRENAMIENTO
            correct_train = 0
            for x, y in zip(x_train, y_train):
                # Predecimos la salida
                output = self.feedforward(x)
                
                # Comprobamos la salida con un umbral de tolerancia (tolerancia)
                if y - tolerancia < output < y + tolerancia:
                    correct_train += 1
            
            # 5. Calculamos el porcentaje de aciertos
            precision_train = (correct_train / len(x_train)) * 100
            historial_precision_train.append(precision_train)
            
            # 6. Calcular Precisión (Accuracy) en DATOS DE PRUEBA
            correct_test = 0
            for x, y in zip(x_test, y_test):
                # Solo predecir
                output = self.feedforward(x) 
                
                # Comprobamos la salida con un umbral de tolerancia (tolerancia)
                if y - tolerancia < output < y + tolerancia:
                    correct_test += 1
            
            # 7. Calculamos el porcentaje de aciertos
            precision_test = (correct_test / len(x_test)) * 100
            historial_precision_test.append(precision_test)
            # --------------------------------------------------------------------------
            
            # X. Se imprime el progreso de la epoca cada 2000 épocas y la última época
            if epoch % 100 == 0 or epoch == epocas - 1:
                print(
                    f"Época {epoch}/{epocas} - Pérdida (pérdida baja = predicción buena): {avg_epoch_loss:.6f} - "
                    f"Precisión (Train): {precision_train:.2f}% - "
                    f"Precisión (Test): {precision_test:.2f}%"
                    )
        print("-------------- Entrenamiento Completo -----------")
        
        # Gráfico 1: Pérdida (Loss)
        fig, ax = plt.subplots()
        dias = range(epocas)
        ax.plot(dias, historial_loss, label='Pérdida (Loss)', color='red')
        ax.set_xlabel('Épocas')
        ax.set_ylabel('Pérdida (Loss)')
        ax.set_title('Pérdida durante el Entrenamiento')
        ax.legend()
        plt.show()

        # Gráfico 2: Precisión (Accuracy)
        fig, ax = plt.subplots()
        ax.plot(range(epocas), historial_precision_train, label='Precisión (Entrenamiento)', color='blue')
        ax.plot(range(epocas), historial_precision_test, label='Precisión (Prueba)', color='green')
        ax.set_xlabel('Épocas')
        ax.set_ylabel(f'Precisión (%) (Tolerancia: {tolerancia})')
        ax.set_title('Precisión durante el Entrenamiento')
        ax.legend()
        plt.show()
    
    # Visualización de la Red Neuronal para mayor entendimiento
    def visualizar_red(self):
        """
        Genera una representación gráfica de la arquitectura de la red neuronal,
        incluyendo neuronas, conexiones, pesos y sesgos.
        """
        # Configuramos el gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Desactivamos los ejes para una apariencia más limpia
        ax.axis('off')

        # Calculamos las posiciones de las neuronas
        posiciones_capas = {}
        espacio_horizontal = 1 / len(self.arquitectura)
        
        for i, num_neuronas in enumerate(self.arquitectura):
            # Calculamos la posición x de la capa actual
            x = (i + 0.5) * espacio_horizontal
            # Calculamos las posiciones y de las neuronas en esta capa
            espacio_vertical = 1 / (num_neuronas + 1)
            posiciones_capas[i] = [(x, (j + 1) * espacio_vertical) for j in range(num_neuronas)]

        # --- Dibujamos las Conexiones (Pesos) ---
        for i in range(len(self.arquitectura) - 1):
            for j in range(self.arquitectura[i]):
                for k in range(self.arquitectura[i+1]):
                    # Coordenadas de inicio y fin de la línea
                    x1, y1 = posiciones_capas[i][j]
                    x2, y2 = posiciones_capas[i+1][k]
                    
                    # Dibujamos la línea que representa la conexión
                    ax.plot([x1, x2], [y1, y2], 'gray', zorder=1)
                    
                    # Obtenemos el peso y lo mostramos en la conexión
                    peso = self.pesos[i][j, k]
                    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f'{peso:.2f}',
                            ha='center', va='center', fontsize=8, color='blue',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

        # --- Dibujamos las Neuronas y los Sesgos ---
        for i, num_neuronas in enumerate(self.arquitectura):
            x, _ = posiciones_capas[i][0] # Todas las neuronas en una capa tienen la misma x
            
            # Etiqueta de la capa
            if i == 0:
                ax.text(x, 1.05, 'Capa de Entrada', ha='center', fontsize=12, weight='bold')
            elif i == len(self.arquitectura) - 1:
                ax.text(x, 1.05, 'Capa de Salida', ha='center', fontsize=12, weight='bold')
            else:
                ax.text(x, 1.05, f'Capa Oculta {i}', ha='center', fontsize=12, weight='bold')

            for j in range(num_neuronas):
                # Obtenemos la posición de la neurona
                x_neurona, y_neurona = posiciones_capas[i][j]
                
                # Dibujamos el círculo que representa la neurona
                circulo = plt.Circle((x_neurona, y_neurona), 0.03, color='skyblue', ec='black', zorder=2)
                ax.add_patch(circulo)
                
                # Añadimos el sesgo (bias) a las neuronas de capas ocultas y de salida
                if i > 0:
                    sesgo = self.sesgos[i-1][j]
                    ax.text(x_neurona, y_neurona, f'b={sesgo:.2f}', ha='center', va='center', fontsize=7, color='black')

        plt.show()