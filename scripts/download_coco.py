"""
download_coco.py
================
Smart downloader for the COCO 2017 dataset. Instead of downloading 18GB of images,
this script fetches the lightweight annotation JSON, selects N images per category
(default 100), and downloads only those specific images directly from COCO servers
via multi-threading.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import zipfile
import io
import requests

from collections import defaultdict
from tqdm import tqdm

# Make backend importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import utils

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_annotations():
    # Descarga el archivo ZIP de anotaciones y lo extrae directamente en la memoria RAM.
    # Al procesarlo "al vuelo", evitamos guardar archivos temporales pesados en el disco duro.
    print(f"Downloading COCO annotations from {ANNOTATIONS_URL}...")
    response = requests.get(ANNOTATIONS_URL, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    zip_buffer = io.BytesIO()
    
    with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Annotations ZIP") as pbar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            zip_buffer.write(data)
            pbar.update(len(data))
            
    print("Extracting JSON in memory...")
    with zipfile.ZipFile(zip_buffer) as z:
        with z.open('annotations/instances_train2017.json') as f:
            print("Parsing JSON...")
            return json.load(f)

def download_image(url: str, filepath: str):
    # Función que descarga y guarda una sola imagen.
    # Si detecta que la imagen ya fue descargada antes, la omite. Esto permite reanudar el script si se interrumpe.
    if os.path.exists(filepath):
        return True
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(r.content)
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Download specific COCO images per category.")
    parser.add_argument("--images-per-cat", type=int, default=200, help="Target images per category.")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent download threads.")
    args = parser.parse_args()

    utils.ensure_dirs()
    
    coco_data = download_annotations()
    
    # Map categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Found {len(categories)} categories.")
    
    # Map images
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Map category to images y también image to categories para el metadata
    cat_to_images = defaultdict(set)
    image_to_cats = defaultdict(list)
    
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        cat_name = categories[cat_id]
        
        cat_to_images[cat_id].add(img_id)
        if cat_name not in image_to_cats[img_id]:
            image_to_cats[img_id].append(cat_name)
        
    print(f"Selecting up to {args.images_per_cat} images per category...")
    selected_images = set()
    
    for cat_id, name in categories.items():
        img_ids = list(cat_to_images[cat_id])
        
        count = 0
        # FILTRO CRUCIAL: Aquí limitamos la cantidad de imágenes por clase (por defecto 100).
        # Gracias a esto, no descargamos los 18 GB del dataset completo, sino una versión ligera y balanceada.
        for img_id in img_ids:
            if count >= args.images_per_cat:
                break
            selected_images.add(img_id)
            count += 1
            
    print(f"Total unique images selected: {len(selected_images)}")
    
    download_tasks = []
    for img_id in selected_images:
        info = images_info.get(img_id)
        if not info or 'coco_url' not in info:
            continue
        
        out_path = os.path.join(utils.IMAGES_DIR, f"coco_{img_id:012d}.jpg")
        download_tasks.append((info['coco_url'], out_path))
        
    print(f"Starting parallel download with {args.workers} workers...")
    success_count = 0
    # MULTIHILO: Ejecuta descargas simultáneas usando 'workers' (hilos).
    # Esto reduce el tiempo de descarga de horas a tan solo unos minutos.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_image, url, path): path for url, path in download_tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading Images"):
            if future.result():
                success_count += 1
                
    print(f"\n Successfully downloaded {success_count} / {len(download_tasks)} images to {utils.IMAGES_DIR}")
    
    # Generar y guardar metadatos
    print("Generando archivo de metadatos (metadata.json)...")
    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata.json")
    metadata = {}
    for img_id in selected_images:
        filename = f"coco_{img_id:012d}.jpg"
        metadata[filename] = image_to_cats[img_id]
        
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"✅ Metadata guardada exitosamente en: {metadata_path}")
    
    print("Next step: python scripts/build_index.py")

if __name__ == "__main__":
    main()
