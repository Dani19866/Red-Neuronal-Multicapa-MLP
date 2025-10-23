import customtkinter as ctk
from tkinter import messagebox
from utils import seleccionar_archivo
from perceptron import MLP
from gui_controller import crear_config
from gui_functions import train, save_mlp, feedforward
import numpy as np
import json

# NOTE: Inicializar la GUI
def GUI():
    """Configura y muestra la ventana principal de la GUI."""
    # --- Configuración de la Apariencia ---
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # --- Creación de la Ventana Principal ---
    app = ctk.CTk()
    app.title("Tarea 1 - Perceptrón multicapa")
    app.geometry("400x250")
    app.resizable(False, False)

    # --- Creación y Posicionamiento de los Widgets ---
    # Título de la interfaz
    titulo_label = ctk.CTkLabel(app, text="Eliga una de las opciones", font=ctk.CTkFont(size=20, weight="bold"))
    titulo_label.pack(pady=(30, 20))

    # Botón para crear un nuevo perceptrón
    boton_crear = ctk.CTkButton(app, text="Crear un nuevo perceptrón", command= lambda: datos_para_crear_MLP(app), width=250, height=40, font=ctk.CTkFont(size=14))
    boton_crear.pack(pady=10)

    # Botón para cargar un perceptrón existente
    boton_cargar = ctk.CTkButton(app, text="Cargar un perceptrón", command= lambda: cargar_perceptron(app), width=250, height=40, font=ctk.CTkFont(size=14))
    boton_cargar.pack(pady=10)

    # --- Iniciar el Bucle Principal de la Aplicación ---
    app.mainloop()
    
# NOTE: Obtener los datos para la creación del MLP mediante una GUI
def datos_para_crear_MLP(parent_window):
    # --- Ventana del Formulario (Toplevel) ---
    ventana_creacion = ctk.CTkToplevel()
    ventana_creacion.title("Configurar Nuevo Perceptrón")
    ventana_creacion.geometry("550x375") 
    ventana_creacion.resizable(False, False)
    ventana_creacion.transient()
    
    # Ocultar la ventana principal mientras se muestra el formulario
    parent_window.withdraw()

    # --- Variables para almacenar las rutas de los archivos ---
    file_paths = {
        "train_data": ctk.StringVar(),
        "train_expected": ctk.StringVar(),
        "test_data": ctk.StringVar(),
        "test_expected": ctk.StringVar()
    }

    # --- Diseño del Formulario ---
    frame = ctk.CTkFrame(ventana_creacion)
    frame.pack(padx=20, pady=20, fill="both", expand=True)

    # Título dentro del formulario
    titulo_formulario = ctk.CTkLabel(frame, text="Configuración del Perceptrón Multicapa", font=ctk.CTkFont(size=18, weight="bold"))
    titulo_formulario.grid(row=0, column=0, columnspan=2, padx=3, pady=(0, 10), sticky="w")

    # --- Entradas numéricas ---
    ctk.CTkLabel(frame, text="Neuronas de Entrada (x1, x2, ..., xn):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    entry_neuronas_entrada = ctk.CTkEntry(frame, placeholder_text="Ej: 2")
    entry_neuronas_entrada.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
    entry_neuronas_entrada.insert(0, "2")

    ctk.CTkLabel(frame, text="Neuronas de Salida (y1, y2, ..., yn):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    entry_neuronas_salida = ctk.CTkEntry(frame, placeholder_text="Ej: 1")
    entry_neuronas_salida.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
    entry_neuronas_salida.insert(0, "1")

    ctk.CTkLabel(frame, text="Número de Capas Ocultas:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
    entry_capas_ocultas = ctk.CTkEntry(frame, placeholder_text="Ej: 2")
    entry_capas_ocultas.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

    ctk.CTkLabel(frame, text="Neuronas por Capa Oculta:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
    entry_neuronas_por_capa = ctk.CTkEntry(frame, placeholder_text="Separadas por comas. Ej: 4,3")
    entry_neuronas_por_capa.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
    
    nota_pesos = ctk.CTkLabel(frame, 
        text="Los pesos y sesgo serán obtenidos aleatoriamente y ajustados mediante el método de la retropropagación", 
        wraplength=550, 
        text_color="#FACC15", 
        justify="left", 
        font=ctk.CTkFont(size=12, slant="italic"))
    nota_pesos.grid(row=8, column=0, columnspan=2, padx=10, pady=(15, 5), sticky="w")
    
    # Diccionario para pasar los widgets de entrada a la función externa
    entries = {
        "neuronas_entrada": entry_neuronas_entrada,
        "neuronas_salida": entry_neuronas_salida,
        "capas_ocultas": entry_capas_ocultas,
        "neuronas_por_capa": entry_neuronas_por_capa,
    }

    # Botón de confirmación
    boton_confirmar = ctk.CTkButton(frame, text="Crear Perceptrón", command=lambda: cargar_configuracion(entries, file_paths, ventana_creacion), height=40)
    boton_confirmar.grid(row=14, column=0, columnspan=2, padx=10, pady=(20, 10), sticky="ew")
    
# NOTE: Cargar la configuración y mostrar el resumen del MLP
def cargar_configuracion(entries, file_paths, parent_window):
    CONFIG = crear_config(entries, file_paths)
    resumen_MLP(CONFIG, parent_window)

# NOTE: Carga un perceptrón desde un archivo JSON
def cargar_perceptron(parent_window):
    # Buscar archivo
    path = seleccionar_archivo( "Abrir configuración del MLP", [("JSON Files", "*.json")])
    if path is None:
        messagebox.showerror("Error de explorador de archivos", "Es indispensable seleccionar el archivo de configuración de MLP.")
        return

    # Extraer datos de configuración y mostrar resumen
    try:
        with open(path, "r") as f:
            config = json.load(f)
            
        # Crear la instancia de MLP
        mlp = MLP(
            neuronas_entrada=config["neuronas_entrada"],
            neuronas_salida=config["neuronas_salida"],
            capas_ocultas=config["capas_ocultas"],
            neuronas_por_capa=config["neuronas_por_capa"]
        )
        
        # Sobrescribir pesos y sesgos convirtiendo listas a np.array
        mlp.pesos = [np.array(w) for w in config["pesos"]]
        mlp.sesgos = [np.array(b) for b in config["sesgos"]]
        mlp.neuronas_en_total = config["neuronas_en_total"]
        
        messagebox.showinfo("Carga Exitosa", f"Modelo MLP cargado correctamente desde:\n{path}")
        resumen_MLP(config, parent_window, mlp)
        
    except FileNotFoundError:
        messagebox.showerror("Error de Archivo", f"No se encontró el archivo en la ruta:\n{path}")
        return None
    
    except KeyError as e:
        messagebox.showerror("Error de Configuración", f"El archivo JSON no tiene la clave esperada: {e}.")
        return None
    
    except Exception as e:
        messagebox.showerror("Error Inesperado", f"Ocurrió un error al cargar el modelo:\n{e}")
        return None
    
# NOTE: Resume la información del MLP y ofrece acciones
def resumen_MLP(CONFIG, parent_window, mlp: MLP = None):
    # 1. Crear la instancia de la lógica del MLP
    if mlp is None:
        mlp = MLP(
            neuronas_entrada=CONFIG["neuronas_entrada"],
            neuronas_salida=CONFIG["neuronas_salida"],
            capas_ocultas=CONFIG["capas_ocultas"],
            neuronas_por_capa=CONFIG["neuronas_por_capa"]
        )        
    
    # Ocultar el formulario mientras se muestra el resumen del MLP
    parent_window.withdraw()
    
    # 3. Crear la nueva ventana de resumen con mayor anchura
    ventana_resumen = ctk.CTk()
    ventana_resumen.title("Panel de Control del Perceptrón Multicapa (MLP)")
    ventana_resumen.geometry("700x550")
    ventana_resumen.resizable(False, False)

    # --- Frame Principal (ahora usará un grid de 2 columnas) ---
    main_frame = ctk.CTkFrame(ventana_resumen, corner_radius=15)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    # Configurar el grid para que las dos columnas se expandan por igual
    main_frame.grid_columnconfigure((0, 1), weight=1)
    main_frame.grid_rowconfigure(1, weight=1)

    # Título General (abarca las dos columnas)
    titulo_general = ctk.CTkLabel(main_frame, text="Configuración y Control del Perceptrón", font=ctk.CTkFont(size=22, weight="bold"))
    titulo_general.grid(row=0, column=0, columnspan=2, padx=20, pady=(10, 20))

    # --- COLUMNA IZQUIERDA: Resumen de Configuración ---
    frame_izquierda = ctk.CTkFrame(main_frame)
    frame_izquierda.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")

    titulo_config = ctk.CTkLabel(frame_izquierda, text="Resumen de Configuración", font=ctk.CTkFont(size=16, weight="bold"))
    titulo_config.pack(pady=(10, 15))

    config_frame = ctk.CTkFrame(frame_izquierda, fg_color="transparent")
    config_frame.pack(fill="x", expand=True, padx=15)
    config_frame.grid_columnconfigure(1, weight=1)

    label_map = {
        "neuronas_entrada": "Neuronas de Entrada:", 
        "neuronas_salida": "Neuronas de Salida:",
        "capas_ocultas": "Nº Capas Ocultas:",
        "neuronas_por_capa": "Neuronas por Capa:",
    }
    
    for i, (key, label_text) in enumerate(label_map.items()):
        if key in CONFIG:
            value = CONFIG[key]
            label = ctk.CTkLabel(config_frame, text=label_text, font=ctk.CTkFont(weight="bold"), anchor="w")
            label.grid(row=i, column=0, sticky="w", padx=5, pady=8)
            
            display_value = ', '.join(map(str, value)) if isinstance(value, list) else value
            value_label = ctk.CTkLabel(config_frame, text=display_value, anchor="e", wraplength=200, justify="right")
            value_label.grid(row=i, column=1, sticky="e", padx=5, pady=8)

    # --- COLUMNA DERECHA: Acciones ---
    frame_derecha = ctk.CTkFrame(main_frame)
    frame_derecha.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")

    titulo_acciones = ctk.CTkLabel(frame_derecha, text="Acciones", font=ctk.CTkFont(size=16, weight="bold"))
    titulo_acciones.pack(pady=(10, 15))

    acciones_frame = ctk.CTkFrame(frame_derecha, fg_color="transparent")
    acciones_frame.pack(fill="x", expand=True, padx=30)

    button_definitions = [
        {"text": "Visualizar Red", "command": mlp.visualizar_red, "colors": {"fg_color": "#1F6AA5", "hover_color": "#144E7A"}},
        {"text": "Entrenar Red", "command": lambda: train(mlp, parent_window), "colors": {"fg_color": "#D35400", "hover_color": "#A04000"}},
        {"text": "Feedforward", "command": lambda: feedforward(mlp), "colors": {"fg_color": "#8E44AD", "hover_color": "#6C3483"}},
        {"text": "Guardar red", "command": lambda: save_mlp(mlp), "colors": {"fg_color": "#917701", "hover_color": "#8B7C37"}}
    ]

    for btn_info in button_definitions:
        button = ctk.CTkButton(
            acciones_frame, text=btn_info["text"], command=btn_info["command"], height=45,
            font=ctk.CTkFont(size=14, weight="bold"), fg_color=btn_info["colors"]["fg_color"],
            hover_color=btn_info["colors"]["hover_color"], cursor="hand2"
        )
        button.pack(pady=10, fill="x", expand=True)

    # 4. Iniciar el bucle de la ventana
    ventana_resumen.mainloop()

if __name__ == "__main__":
    GUI()