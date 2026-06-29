# 🔍 Sistema Multimodal de Búsqueda de Imágenes

Búsqueda de imágenes **texto → imagen** e **imagen → imagen** usando **CLIP**
para los embeddings y **FAISS** para la búsqueda vectorial eficiente por
similitud coseno. Backend en **FastAPI**, frontend en HTML/CSS/JS vanilla
servido directamente desde el mismo servidor.

| Modo | Ruta | Entrada | Salida |
|------|------|---------|--------|
| Texto → Imagen | `/` | Una descripción (`"a sleeping cat"`) | Imágenes más parecidas |
| Imagen → Imagen | `/image-search` | Una imagen subida | Imágenes visualmente similares |

---

## 🏗️ Arquitectura

```
                   ┌─────────────────────────────────────────────┐
   Texto  ───────► │ CLIP text encoder  ──┐                       │
                   │                       ├─► embedding (512-d)   │
   Imagen ───────► │ CLIP image encoder ──┘   (normalizado L2)    │
                   └───────────────────────────────┬─────────────┘
                                                    │
                                                    ▼
                                   ┌─────────────────────────────┐
                                   │  FAISS IndexFlatIP           │
                                   │  (inner product = coseno)    │
                                   │  top-k vecinos más cercanos  │
                                   └──────────────┬──────────────┘
                                                  │
                                                  ▼
                                   JSON: [{ image_url, score }, ...]
```

Como CLIP proyecta texto e imágenes al **mismo espacio vectorial**, ambos tipos
de consulta usan exactamente el mismo índice. Los embeddings se **normalizan
L2**, por lo que el producto interno (`IndexFlatIP`) equivale a la **similitud
coseno**.

### Estructura del proyecto

```
image_search_system/
├── backend/
│   ├── main.py            # App FastAPI: lifespan, CORS, rutas, estáticos
│   ├── embedder.py        # CLIP: get_text_embedding / get_image_embedding
│   ├── indexer.py         # FAISS: build_index / load_index / search
│   ├── utils.py           # Config (.env) + helpers de imágenes y respuestas
│   └── routes/
│       ├── text_search.py   # POST /search/text
│       └── image_search.py  # POST /search/image
├── frontend/
│   ├── templates/         # index.html, image_search.html (Jinja2)
│   └── static/            # css/styles.css, js/app.js
├── data/
│   ├── images/            # Imágenes del dataset (pobladas por script)
│   └── embeddings/        # index.faiss + image_paths.json (generados)
├── scripts/
│   ├── download_dataset.py  # Descarga el dataset desde HuggingFace
│   └── build_index.py       # Extrae embeddings y construye el índice
├── requirements.txt
├── .env.example
└── README.md
```

---

## ✅ Requisitos previos

- **Python 3.10+**
- **~4 GB de espacio en disco** (modelo CLIP + dataset)
- **Conexión a internet** en la primera ejecución (descarga del modelo y del
  dataset desde HuggingFace)

> El modelo por defecto es `ViT-B-32` (pesos `openai`), que corre cómodamente en
> CPU. No se requiere GPU.

---

## 🚀 Instalación y ejecución

```bash
# 1. Crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el dataset (por defecto Oxford Flowers 102, ~2000 imágenes)
python scripts/download_dataset.py

# 4. Construir el índice FAISS (5–15 min según el hardware)
python scripts/build_index.py

# 5. Iniciar el servidor
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Luego abrir en el navegador:

- Búsqueda por texto:  <http://localhost:8000/>
- Búsqueda por imagen: <http://localhost:8000/image-search>

> ⚠️ Ejecuta `uvicorn` **desde la raíz del proyecto** (la carpeta
> `image_search_system/`), de modo que el paquete `backend` sea importable.

---

## 🗂️ Elección del dataset

El script `download_dataset.py` usa la librería `datasets` de HuggingFace y
acepta varias opciones:

```bash
# Oxford Flowers 102 (por defecto, ligero y fiable)
python scripts/download_dataset.py

# Imagenette: 10 clases de fotos reales y diversas (mejor demo de texto)
python scripts/download_dataset.py --dataset imagenette

# Limitar la cantidad de imágenes
python scripts/download_dataset.py --max-images 1000

# Cualquier dataset de imágenes del Hub por su repo id
python scripts/download_dataset.py --dataset <usuario/dataset> --split train
```

> 💡 El ejemplo `"a sleeping cat"` luce mejor con un dataset **diverso**. Flowers
> es homogéneo (solo flores); para una demo más vistosa de texto→imagen usa
> `--dataset imagenette` o un mirror de COCO. Con Flowers, prueba consultas como
> `"a field of yellow flowers"` o `"a pink rose"`.

Tras descargar, **vuelve a ejecutar** `python scripts/build_index.py` para
reconstruir el índice con las nuevas imágenes.

---

## 🔌 API

### `POST /search/text`
```jsonc
// Request
{ "query": "a sleeping cat", "top_k": 6 }

// Response
{
  "results": [
    { "image_url": "/images/img000123.jpg", "score": 0.8731 },
    ...
  ],
  "count": 6,
  "took_ms": 41.2
}
```

### `POST /search/image`
`multipart/form-data` con el campo `file` (JPG/PNG/WebP) y, opcionalmente,
`top_k`. La respuesta tiene la misma forma que la de texto.

```bash
curl -X POST http://localhost:8000/search/image \
  -F "file=@/ruta/a/mi_imagen.jpg" \
  -F "top_k=6"
```

### Otros endpoints
| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Página de búsqueda por texto |
| `GET` | `/image-search` | Página de búsqueda por imagen |
| `GET` | `/images/{nombre}` | Sirve las imágenes del dataset (estáticas) |
| `GET` | `/health` | Estado del modelo y del índice |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

### Manejo de errores
- **400** — el archivo subido no es una imagen válida.
- **503** — el índice FAISS aún no está cargado (ejecuta `build_index.py` y
  reinicia el servidor).

---

## 📈 Notas sobre escalabilidad

La arquitectura está pensada para crecer:

- **Índice aproximado para datasets grandes.** `IndexFlatIP` es exacto pero hace
  búsqueda por fuerza bruta (O(N) por consulta). Para **> 100K imágenes**,
  cámbialo por `IndexIVFFlat`. En `backend/indexer.py` ya está documentado el
  reemplazo:

  ```python
  nlist = 100
  quantizer = faiss.IndexFlatIP(dim)
  index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
  index.train(vectors)   # IVF requiere una pasada de entrenamiento
  index.add(vectors)
  index.nprobe = 10      # compromiso velocidad/exactitud
  ```

- **Memoria controlada.** Si la RAM es limitada, los embeddings pueden
  calcularse y persistirse **por bloques (chunks)** en lugar de mantener toda la
  matriz en memoria.

- **Escalado horizontal.** La API es **stateless**: el índice se carga al inicio
  y solo se lee. Puede replicarse tras un balanceador o servirse con varios
  workers:

  ```bash
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
  ```

- **Aceleración del indexado.** `build_index.py` embebe imagen por imagen
  (siguiendo el flujo del enunciado). Para corpus grandes conviene **procesar
  por lotes (batches)** en la GPU, lo que reduce drásticamente el tiempo de
  construcción.

---

## ⚙️ Configuración (`.env`)

Copia `.env.example` a `.env` para ajustar parámetros (todos tienen valores por
defecto):

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `DATA_DIR` | `<root>/data` | Carpeta de imágenes e índice |
| `TOP_K_DEFAULT` | `6` | Número de resultados por consulta |
| `MODEL_NAME` | `ViT-B-32` | Arquitectura CLIP (open-clip) |
| `PRETRAINED` | `openai` | Pesos preentrenados |

---

## 🧰 Solución de problemas

| Problema | Causa probable | Solución |
|----------|----------------|----------|
| `503 ... index is not loaded` | Falta el índice | Ejecuta `python scripts/build_index.py` y reinicia |
| `No images found in ...` | Dataset sin descargar | Ejecuta `python scripts/download_dataset.py` |
| `ModuleNotFoundError: backend` | Ruta de ejecución incorrecta | Lanza `uvicorn` desde la raíz del proyecto |
| Descarga del dataset falla | Dataset no disponible | Prueba `--dataset imagenette` u otro repo id |
| Instalación de `torch` lenta | Wheel grande | Es normal en la primera instalación |

---

## 🧪 Stack tecnológico

- **Backend:** FastAPI · Uvicorn
- **ML / Embeddings:** open-clip-torch (CLIP `ViT-B-32`) · PyTorch
- **Búsqueda vectorial:** FAISS (`faiss-cpu`, `IndexFlatIP`)
- **Imágenes:** Pillow · NumPy
- **Datos:** HuggingFace `datasets`
- **Frontend:** HTML + CSS + JavaScript vanilla (servido con `StaticFiles` +
  `Jinja2Templates`)
