import os
import json
import torch
import open_clip
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data", "embeddings_numpy")
    
    matrix_path = os.path.join(output_dir, "embeddings.npy")
    paths_path = os.path.join(output_dir, "paths.json")
    
    if not os.path.exists(matrix_path) or not os.path.exists(paths_path):
        print("❌ No se encontraron los embeddings. Ejecuta standalone_embedder.py primero.")
        return

    # 1. Cargar embeddings e info de rutas
    print("Cargando matriz de embeddings de prueba...")
    embeddings = np.load(matrix_path)
    with open(paths_path, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
        
    print(f"✅ Cargados {len(image_paths)} embeddings de imágenes.")
    
    # Cargar metadatos si existen
    metadata_path = os.path.join(base_dir, "data", "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("✅ Metadatos de categorías cargados.")

    # 2. Cargar modelo CLIP (Solo para el texto esta vez)
    print("Cargando modelo CLIP para texto...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Solo necesitamos el modelo y el tokenizador (para convertir palabras a tokens)
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    model = model.to(device)
    model.eval()

    # 3. Bucle interactivo de prueba
    print("\n=============================================")
    print(" 🧪 MODO DE PRUEBA DE SIMILITUD DE TEXTO 🧪")
    print("=============================================")
    print("Escribe 'salir' para terminar el programa.")
    
    while True:
        query = input("\n🔍 Ingresa tu búsqueda (ej. 'a dog', 'un gato'): ")
        if query.lower() == 'salir':
            print("Saliendo de la prueba...")
            break
            
        if not query.strip():
            continue
            
        # Convertir texto a vector (Embedding de texto)
        text_tokens = tokenizer([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            # Normalización L2 del vector de texto
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        text_vector = text_features.cpu().numpy().astype("float32")[0]
        
        # 4. Calcular Similitud del Coseno
        # Como los vectores de imagen y de texto ya están normalizados (L2), 
        # la similitud del coseno se calcula simplemente multiplicándolos (producto punto)
        similarities = np.dot(embeddings, text_vector)
        
        # Obtener los 5 índices con la puntuación de similitud más alta
        # argsort ordena de menor a mayor, [::-1] invierte para que sea de mayor a menor, [:5] toma los 5 primeros
        top_5_indices = np.argsort(similarities)[::-1][:5]
        
        print(f"\n🏆 Top 5 imágenes más similares para: '{query}'")
        print("-" * 50)
        for i, idx in enumerate(top_5_indices):
            # Formateamos la salida para que sea fácil de leer
            score = similarities[idx]
            path = image_paths[idx]
            filename = os.path.basename(path)
            
            # Buscar categorías en el metadato
            categorias = metadata.get(filename, ["Desconocida"])
            cat_str = ", ".join(categorias)
            
            print(f"{i+1}. [Similitud: {score:.4f}] -> Categorías: [{cat_str}] | Archivo: {filename}")

if __name__ == "__main__":
    main()
