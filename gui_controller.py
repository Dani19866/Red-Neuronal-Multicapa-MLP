from tkinter import messagebox

# LISTO: Verificar los datos antes de crear el MLP
# NOTE: EDITAR FUNCIÓN PARA QUITAR LOS ARCHIVOS DE PRUEBA Y ENTRENAMIENTO POR DEFECTO
def crear_config(entries, file_paths):
    try:
        # --- RECOPILACIÓN DE DATOS ---
        CONFIG = {
            "neuronas_entrada": int(entries["neuronas_entrada"].get()),
            "neuronas_salida": int(entries["neuronas_salida"].get()),
            "capas_ocultas": int(entries["capas_ocultas"].get()),
            "neuronas_por_capa": [int(n.strip()) for n in entries["neuronas_por_capa"].get().split(',')],
        }
        
        # Comprueba si el número de capas ocultas coincide con la cantidad de neuronas especificadas.
        if CONFIG["capas_ocultas"] != len(CONFIG["neuronas_por_capa"]):
            messagebox.showerror(
                "Error de Configuración",
                "El número de capas ocultas no coincide con la cantidad de neuronas por capa.\n\n"
                f"Número de capas esperado: {CONFIG['capas_ocultas']}\n"
                f"Cantidad de neuronas proveídas: {len(CONFIG['neuronas_por_capa'])}\n\n"
                "Ejemplo: Si elige 3 capas ocultas, debe ingresar 3 números separados por comas (ej: 4,3,2)."
            )
            return
        
        return CONFIG
        
    except ValueError:
        messagebox.showerror("Error de Entrada", "Por favor, asegúrate de que todos los campos numéricos contengan números válidos (ej. '4,3,2' para neuronas y '0.01' para el error).")
    
    except Exception as e:
        messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
