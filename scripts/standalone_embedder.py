

import os
import json
import argparse
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generador de Embeddings con límite opcional.")
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de imágenes a procesar")
    args = parser.parse_args()
    
    #  Configuración Inicial y Rutas

    # Obtenemos la ruta principal del proyecto y definimos las carpetas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, "data", "images")
    output_dir = os.path.join(base_dir, "data", "embeddings_numpy")
    
    # Creamos la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
   
    # 2. Carga del Modelo CLIP y Preprocesamiento
    
    # Utilizamos la GPU si está disponible, de lo contrario usamos el CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # open_clip nos devuelve 3 cosas: el modelo, _, y la función de preprocesamiento para imágenes
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    # Movemos el modelo a la memoria de la GPU (o CPU) y lo ponemos en modo evaluación
    model = model.to(device)
    model.eval() 
    
    
    # 3. Escaneo de la Carpeta de Imágenes
    
    print(f"\nBuscando imágenes en: {images_dir}")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = []
    
    # Recorremos la carpeta buscando archivos con extensión de imagen válida
    for root, _, files in os.walk(images_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                # Guardamos solo la ruta relativa (ej: n02084071/perro.jpg)
                image_paths.append(os.path.relpath(os.path.join(root, file), images_dir))
                
    if not image_paths:
        print("No se encontraron imágenes en el directorio especificado.")
        return
        
    if args.limit:
        # Aquí cortamos la lista para quedarnos solo con la cantidad indicada
        image_paths = image_paths[:args.limit]
        print(f"Límite aplicado: Se procesarán como máximo {len(image_paths)} imágenes.")
    else:
        print(f"Se encontraron {len(image_paths)} imágenes listas para procesar.")
    
    
    # 4. Procesamiento: De Imagen a Vector (Embedding)
    embeddings_list = []
    kept_paths = []
    
    print("\nGenerando embeddings...")
    # tqdm nos muestra una barra de progreso bonita en la terminal
    for rel_path in tqdm(image_paths):
        abs_path = os.path.join(images_dir, rel_path)
        try:
            #  Abrir la imagen usando PIL (Python Imaging Library)
            image = Image.open(abs_path).convert("RGB")
            
            # Preprocesar la imagen (recortar, redimensionar y normalizar colores para CLIP)
            # unsqueeze(0) añade una dimensión extra requerida por el modelo (batch size de 1)
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Inferencia de la Inteligencia Artificial
            # Usamos torch.no_grad() para que no consuma memoria extra calculando gradientes de entrenamiento
            with torch.no_grad():
                # CLIP extrae las características (el significado de la imagen en números)
                features = model.encode_image(image_input)
                
                # Normalización L2 (Crucial)
                # Escala el vector para que la búsqueda por similitud de coseno funcione correctamente
                features = features / features.norm(dim=-1, keepdim=True)
                
            # Conversión a Numpy
            # Sacamos el vector de la GPU a la CPU, lo convertimos a un array clásico y a float32
            vector = features.cpu().numpy().astype("float32")[0]
            
            # Guardamos el vector y su respectiva ruta
            embeddings_list.append(vector)
            kept_paths.append(rel_path)
            
        except Exception as e:
            print(f"Error procesando la imagen {rel_path}: {e}")
            
    
    # Agrupamiento y Guardado en Matriz (Numpy Array)
    
    if not embeddings_list:
        print("No se generó ningún embedding.")
        return
        
    # Apilamos la lista de vectores individuales (1D) en una sola Matriz Gigante Bidimensional (2D)
    matrix = np.vstack(embeddings_list)
    
    # Definimos dónde guardar los resultados
    matrix_path = os.path.join(output_dir, "embeddings.npy")
    paths_path = os.path.join(output_dir, "paths.json")
    
    # Guardamos la matriz matemáticamente estructurada usando NumPy puro
    np.save(matrix_path, matrix)
    
    # Guardamos la lista de rutas en texto plano (JSON) para saber a qué imagen pertenece cada fila
    with open(paths_path, "w", encoding="utf-8") as f:
        json.dump(kept_paths, f, indent=4)
        
    print("\n=======================================================")
    print("¡Proceso Finalizado con Éxito!")
    print(f"- Matriz NumPy guardada en: {matrix_path}")
    print(f"  (La matriz tiene {matrix.shape[0]} filas/imágenes y {matrix.shape[1]} columnas/dimensiones)")
    print(f"- Diccionario de rutas guardado en: {paths_path}")

if __name__ == "__main__":
    main()
