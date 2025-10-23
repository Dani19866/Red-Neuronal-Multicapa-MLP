from perceptron import MLP
from utils import read_csv, seleccionar_archivo, seleccionar_ruta_guardado
import numpy as np
from tkinter import messagebox
import customtkinter as ctk
import json

def feedforward(mlp: MLP):
    print("Feedforward Invocado")
    messagebox.showinfo("Información", "Selecciona el archivo que usará para realizar las pruebas.")
    
    # Solicitamos los datos de prueba (test)
    path = seleccionar_archivo("Selecciona el archivo de datos de prueba (test)", [("CSV Files", "*.csv")])
    if path is None:
        messagebox.showerror("Error de explorador de archivos", "Es indispensable seleccionar un archivo de datos de prueba.")
        return
    
    # Leemos los datos
    vectores, salidas_esperadas = read_csv(path)
    
    # Verificación de error en la lectura
    if vectores == -1 and salidas_esperadas == -1:
        return
    
    # Usamos la funcion feedforward del MLP
    resultados = []
    for vector in vectores:
        # Ejecutamos feedforward
        resultado = mlp.feedforward(np.array(vector))
        
        # Verificación de error
        if resultado == -1:
            return
        
        # Añadimos los resultados a la lista
        resultados.append(resultado)
    
    # Mostramos los resultados
    print("Resultados de la red para los datos de prueba:")
    for i, res in enumerate(resultados):
        print(f"Entrada: {str(vectores[i]):<30} | Salida Esperada: {salidas_esperadas[i]} | Salida Red: {str(res):<20}")
    
def train(mlp: MLP, parent: ctk.CTk):
    print("Comenzando entrenamiento de la red...")
    
    # Solicitamos las epocas y el learning rate
    epochs, learning_rate, tolerancia = request_train_parameters(parent)
    if epochs is None or learning_rate is None:
        messagebox.showerror("Error de selección", "No se introducieron valores o se canceló la solicitud de valores.")
        return
    
    # Solicitamos los datos de entrenamiento (train)
    messagebox.showinfo("Información", "Por favor, cargue los datos de entrenamiento.")
    path_train = seleccionar_archivo("Selecciona el archivo de datos de entrenamiento (train)", [("CSV Files", "*.csv")])
    if path_train is None:
        messagebox.showerror("Error de explorador de archivos", "Es indispensable seleccionar un archivo de datos de prueba.")
        return
    
    # Solicitamos los datos de prueba (test)
    messagebox.showinfo("Información", "Por favor, cargue los datos de prueba.")
    path_test = seleccionar_archivo("Selecciona el archivo de datos de prueba (test)", [("CSV Files", "*.csv")])
    if path_test is None:
        messagebox.showerror("Error de explorador de archivos", "Es indispensable seleccionar un archivo de datos de prueba.")
        return
    
    # Leemos los datos
    x_train, y_train = read_csv(path_train)
    x_test, y_test = read_csv(path_test)
    
    # Convertir las listas a arrays de numpy
    x_train = [np.array(v) for v in x_train]
    y_train = [np.array(s) for s in y_train]
    x_test = [np.array(v) for v in x_test]
    y_test = [np.array(s) for s in y_test]
    
    # Llamamos a la función MLP.train
    mlp.train(x_train, y_train, x_test, y_test, epochs, learning_rate, tolerancia)

def save_mlp(mlp: MLP):
    print("Guardando el modelo del MLP...")
    
    # Extraer la configuración actual del MLP
    config = {}
    config["neuronas_entrada"] = mlp.neuronas_entrada
    config["neuronas_salida"] = mlp.neuronas_salida
    config["capas_ocultas"] = mlp.capas_ocultas
    config["neuronas_por_capa"] = mlp.neuronas_por_capa
    config["neuronas_en_total"] = mlp.neuronas_en_total
    config["pesos"] = [w.tolist() for w in mlp.pesos]
    config["sesgos"] = [b.tolist() for b in mlp.sesgos]
    
    print(config["pesos"])
    print(config["sesgos"])
    
    # Solicitar la ubicación para guardar el archivo
    path = seleccionar_ruta_guardado("Guardar configuración del MLP", [("JSON Files", "*.json")])
    if path is None:
        messagebox.showerror("Error de explorador de archivos", "Es indispensable seleccionar una ruta para guardar el archivo.")
        return
    
    # Guardar el modelo
    with open(path, "w") as f:
        json.dump(config, f, indent=4)

# Retorna los valores de (epochs, lr) | int, float si no hay error | None, None si se cancela
def request_train_parameters(parent: ctk.CTk):
    # Configuración de la ventana
    ventana_params = ctk.CTkToplevel(parent)
    ventana_params.title("Parámetros de Entrenamiento")
    ventana_params.geometry("400x325")
    ventana_params.resizable(False, False)
    ventana_params.transient(parent)
    ventana_params.grab_set()
    
    frame = ctk.CTkFrame(ventana_params)
    frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    # Variable para almacenar los valores de retorno: epochs, learning_rate, tolerancia
    return_values = [None, None, None]
    
    # Épocas
    ctk.CTkLabel(frame, text="Número de Épocas:").pack(pady=(10, 0), anchor="w", padx=10)
    entry_epocas = ctk.CTkEntry(frame, placeholder_text="Ej: 1000")
    entry_epocas.pack(fill="x", padx=10)
    entry_epocas.insert(0, str(10000)) # Valor por defecto

    # Tasa de Aprendizaje
    ctk.CTkLabel(frame, text="Tasa de Aprendizaje:").pack(pady=(10, 0), anchor="w", padx=10)
    entry_lr = ctk.CTkEntry(frame, placeholder_text="Ej: 0.01")
    entry_lr.pack(fill="x", padx=10)
    entry_lr.insert(0, str(0.0001)) # Valor por defecto
    
    # Tolerancia ± para testear valores predichos en train y test
    ctk.CTkLabel(frame, text="Tolerancia ± para testear valores predichos en train y test:").pack(pady=(10, 0), anchor="w", padx=10)
    entry_tolerancia = ctk.CTkEntry(frame, placeholder_text="Ej: 0.5")
    entry_tolerancia.pack(fill="x", padx=10)
    entry_tolerancia.insert(0, str(0.5)) # Valor por defecto
    
    # --- Validación ---
    def _on_confirm():
        try:
            e = int(entry_epocas.get())
            l = float(entry_lr.get())
            t = float(entry_tolerancia.get())

            if e <= 0:
                messagebox.showerror("Error de Validación", "El número de épocas debe ser un entero positivo.", parent=ventana_params)
                return
            
            if l <= 0.0:
                messagebox.showerror("Error de Validación", "La tasa de aprendizaje debe ser un número positivo.", parent=ventana_params)
                return
            
            if t < 0.0:
                messagebox.showerror("Error de Validación", "La tolerancia debe ser un número no negativo.", parent=ventana_params)
                return
            
            # --- Guardar resultados ---
            return_values[0] = e
            return_values[1] = l
            return_values[2] = t
            
            ventana_params.destroy()

        except ValueError:
            messagebox.showerror("Error de Tipo", "Por favor, ingrese valores numéricos válidos.\n""Épocas debe ser un entero (Ej: 100).\n""Tasa de aprendizaje debe ser un decimal (Ej: 0.1).", parent=ventana_params)
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}", parent=ventana_params)

    def _on_cancel():
        ventana_params.destroy()
        
    # Botones
    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.pack(pady=20, fill="x", side="bottom")
    btn_frame.grid_columnconfigure((0,1), weight=1)
    
    btn_cancel = ctk.CTkButton(btn_frame, text="Cancelar", command=_on_cancel, fg_color="gray", hover_color="#505050")
    btn_cancel.grid(row=0, column=0, padx=(0, 5), sticky="ew")
    
    btn_ok = ctk.CTkButton(btn_frame, text="Confirmar", command=_on_confirm)
    btn_ok.grid(row=0, column=1, padx=(5, 0), sticky="ew")

    # Manejar el cierre de la ventana (X)
    ventana_params.protocol("WM_DELETE_WINDOW", _on_cancel)
    
    # Esperar a que la ventana se cierre
    parent.wait_window(ventana_params)
    
    # Devolver los resultados directamente
    return return_values[0], return_values[1], return_values[2]
