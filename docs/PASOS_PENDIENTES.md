# Pasos pendientes para publicar el proyecto

Guía detallada de todo lo que falta para pasar de la rama `modernizacion` a un demo público y un repositorio listo para portafolio. Cada paso tiene sus comandos exactos, qué hacer en cada plataforma, cómo comprobar que funcionó y los errores más comunes.

**Tiempo total estimado:** 2 a 3 horas, la mayor parte esperando descargas y builds.

| # | Paso | Dónde | Tiempo | Depende de |
|---|------|-------|--------|------------|
| 0 | Cuentas y herramientas | Local | 10 min | — |
| 1 | Base de datos en Supabase | supabase.com | 10 min | 0 |
| 2 | Dataset e indexación | Local (o Docker) | 30 a 60 min | 1 |
| 3 | Backend en Hugging Face Spaces | huggingface.co | 20 min + 10 min de build | 1, 2 |
| 4 | Frontend en Render | render.com | 10 min | 3 |
| 5 | Cerrar CORS y probar de punta a punta | HF + navegador | 10 min | 3, 4 |
| 6 | README final: enlaces, capturas, GIF | Local | 20 min | 5 |
| 7 | Fusionar `modernizacion` en `main` | GitHub | 5 min | 6 |
| 8 | Mantenimiento y extras | — | opcional | 7 |

---

## 0 · Cuentas y herramientas

Necesitas cuentas gratuitas en:

- [supabase.com](https://supabase.com) (puede ser con GitHub).
- [huggingface.co](https://huggingface.co/join).
- [render.com](https://dashboard.render.com/register) (puede ser con GitHub).

En tu máquina ya tienes Python 3.13, Node 22, Docker Desktop y Git. Verifica que estás en la rama correcta y al día:

```bash
cd C:\Users\gcdav\Documents\PERSONAL\PROYECTOS\Project_imagen_search_system
git checkout modernizacion
git pull
git status          # debe decir "nothing to commit, working tree clean"
```

---

## 1 · Base de datos en Supabase

### 1.1 Crear el proyecto

1. Entra a [supabase.com/dashboard](https://supabase.com/dashboard) → **New project**.
2. Organización: la tuya. Nombre: `image-search`. Región: la más cercana a ti (por ejemplo `South America (São Paulo)` o `East US`).
3. **Database password:** genera una y **guárdala**. La vas a necesitar dos veces. Evita caracteres como `@`, `:` o `/` para no tener que codificarla en la URL; si ya la tiene, ver 1.3.
4. Plan **Free**. Espera 1 a 2 minutos a que el proyecto quede en estado *Active*.

### 1.2 Obtener las cadenas de conexión

En el proyecto: **Project Settings** (engranaje, abajo a la izquierda) → **Database** → sección **Connection string** → pestaña **URI**. Hay tres modos; vas a usar dos:

| Modo | Puerto | Para qué | Notas |
|------|--------|----------|-------|
| **Session pooler** | 5432 | Paso 2 (indexar desde tu PC) | Funciona por IPv4. La conexión "Direct" solo tiene IPv6 en el plan gratuito y muchas redes domésticas no la soportan. |
| **Transaction pooler** | 6543 | Paso 3 (el backend en producción) | Ideal para servidores: muchas conexiones cortas. |

Copia ambas URIs. Tienen esta forma:

```
postgresql://postgres.abcdefghijklmnop:[YOUR-PASSWORD]@aws-0-sa-east-1.pooler.supabase.com:5432/postgres
postgresql://postgres.abcdefghijklmnop:[YOUR-PASSWORD]@aws-0-sa-east-1.pooler.supabase.com:6543/postgres
```

Reemplaza `[YOUR-PASSWORD]` por la contraseña del paso 1.1.

### 1.3 Si la contraseña tiene caracteres especiales

Codifícala para URL. En Python:

```bash
python -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "tu-contraseña"
```

Usa el resultado en lugar de la contraseña dentro de la URI.

### 1.4 Habilitar pgvector (opcional, el backend lo hace solo)

El backend ejecuta `CREATE EXTENSION IF NOT EXISTS vector` al arrancar. Si prefieres hacerlo a mano: **SQL Editor** → **New query**:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Verificación:** en **Database → Extensions** busca `vector` y debe aparecer habilitada.

---

## 2 · Dataset e indexación

Esto genera los embeddings de las imágenes y los sube a Supabase. Solo hay que hacerlo una vez (y cada vez que cambies el dataset o el modelo).

### 2.1 Configurar `.env`

```bash
cp .env.example .env
```

Edita `.env`:

```ini
INDEX_BACKEND=pgvector
DATABASE_URL=postgresql://postgres.abcdefghijklmnop:TU_PASSWORD@aws-0-sa-east-1.pooler.supabase.com:5432/postgres
```

(la URI del **Session pooler**, puerto 5432).

### 2.2 Descargar el subset de COCO

Elige un tamaño. Cuanto más grande, mejores resultados, pero más tiempo de descarga e indexación:

| Comando | Imágenes aprox. | Descarga | Indexación en CPU |
|---------|-----------------|----------|-------------------|
| `python scripts/download_coco.py --split val --images-per-cat 40` | ~2 500 | 5 min | 10 min |
| `python scripts/download_coco.py` (train, 100/categoría) | ~6 500 | 15 min | 30 a 45 min |

Recomendación para el portafolio: el segundo. Necesitas `requests` y `tqdm` instalados (`pip install requests tqdm`). El script parsea un JSON de 450 MB en memoria; cierra programas pesados si tienes menos de 8 GB de RAM.

**Verificación:** `data/images/` tiene miles de `.jpg` y existe `data/metadata.json`. Ábrelo: cada entrada debe tener `categories` y `url` empezando por `https://images.cocodataset.org/`.

### 2.3 Generar embeddings y subirlos

Tienes dos formas. La **opción B** evita instalar PyTorch en tu máquina.

**Opción A · Python local**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # descarga ~800 MB de PyTorch, una sola vez
python scripts/build_index.py          # la primera vez descarga el modelo (~1 GB)
```

**Opción B · Con la imagen Docker que ya está construida**

```powershell
docker run --rm `
  -v "${PWD}\data:/app/data" `
  --env-file .env `
  image-search-backend python scripts/build_index.py
```

Si borraste la imagen, reconstrúyela con `docker build -t image-search-backend .` (10 min).

Verás una barra de progreso "Generando embeddings" y al final:

```
Embeddings generados: 6513 imágenes × 512 dimensiones.
Índice construido en 'pgvector' con 6513 imágenes.
```

**Verificación en Supabase:** **Table Editor** → tabla `images` → debe tener tantas filas como imágenes, con `image_url` lleno y `categories` con listas. En **SQL Editor**:

```sql
SELECT count(*) FROM images;
SELECT indexname FROM pg_indexes WHERE tablename = 'images';   -- debe listar images_embedding_hnsw_idx
```

**Errores comunes**

| Error | Causa | Solución |
|-------|-------|----------|
| `password authentication failed` | Contraseña mal copiada o sin codificar | Revisa 1.1 y 1.3 |
| `could not translate host name` | URI del modo Direct (IPv6) | Usa el Session pooler (1.2) |
| `SSL SYSCALL error` a mitad de la inserción | Red inestable | Vuelve a ejecutar; el script hace `TRUNCATE` y reinserta todo |
| Se cae por memoria en `download_coco.py` | JSON de anotaciones de 450 MB | Usa `--split val` |

### 2.4 Probar el índice sin servidor

```bash
python scripts/search_cli.py
Consulta > un perro corriendo en la playa
```

Los primeros resultados deben tener categorías coherentes (`dog`, `person`, `frisbee`...). Si salen aleatorios, el índice se construyó con un modelo distinto al de `.env`: reconstruye.

---

## 3 · Backend en Hugging Face Spaces

### 3.1 Crear el Space

1. Entra a [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Owner:** tu usuario. **Space name:** `image-search` (la URL será `https://<usuario>-image-search.hf.space`).
3. **License:** MIT. **SDK:** **Docker** → plantilla **Blank**.
4. **Space hardware:** `CPU basic · 2 vCPU · 16 GB · FREE`.
5. **Visibility:** Public. → **Create Space**.

### 3.2 Token de escritura

**Settings** (avatar) → **Access Tokens** → **Create new token** → tipo *Write* → nombre `image-search-deploy`. Cópialo; es la contraseña para hacer `git push` al Space.

### 3.3 Añadir los metadatos del Space al README

Hugging Face lee la configuración del Space de un bloque YAML al inicio de `README.md`. Añade esto en la **primera línea** del archivo, antes del `<div align="center">`:

```yaml
---
title: Buscador Multimodal de Imágenes
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Búsqueda texto→imagen e imagen→imagen con CLIP + pgvector
---
```

GitHub muestra ese bloque como una tablita al inicio del README; es lo habitual en proyectos alojados en Spaces y no afecta al resto.

```bash
git add README.md
git commit -m "Metadatos del Space de Hugging Face"
git push
```

### 3.4 Subir el código al Space

```bash
git remote add hf https://huggingface.co/spaces/<usuario>/image-search
git push hf modernizacion:main
```

Cuando pida credenciales: usuario = tu usuario de HF, contraseña = el token del paso 3.2. Si el Space se creó con un README propio y el push es rechazado, usa `git push --force hf modernizacion:main` (el Space está vacío, no pierdes nada).

### 3.5 Variables y secretos

En el Space: **Settings** → **Variables and secrets**:

| Tipo | Nombre | Valor |
|------|--------|-------|
| **Secret** | `DATABASE_URL` | URI del **Transaction pooler** (puerto 6543), con la contraseña |
| Variable | `INDEX_BACKEND` | `pgvector` |
| Variable | `CORS_ORIGINS` | `*` por ahora; se ajusta en el paso 5 |

Guardar reinicia el Space.

### 3.6 Esperar el build y verificar

Pestaña **Logs** del Space. El build tarda unos 10 minutos (instala PyTorch y descarga el modelo). Luego, en los logs de *Container*, deberías ver:

```
Arrancando: modelo CLIP 'xlm-roberta-base-ViT-B-32' + backend 'pgvector' ...
Modelo CLIP cargado. Dimensión de incrustación: 512
PostgreSQL listo: 6513 vectores en la tabla images.
Application startup complete.
```

**Verificación:**

- `https://<usuario>-image-search.hf.space/health` → `"model_loaded": true, "index_loaded": true, "index_size": 6513`.
- `https://<usuario>-image-search.hf.space/docs` → Swagger. Prueba `POST /search/text` con `{"query": "un gato", "top_k": 3}`.

**Errores comunes**

| Síntoma | Causa | Solución |
|---------|-------|----------|
| Build falla en `pip install` | Caída puntual de PyPI | **Factory reboot** en Settings |
| `index_loaded: false` y en logs `Base de datos no configurada` | Falta `DATABASE_URL` | Revisa 3.5 (es *Secret*, no *Variable*, pero ambos funcionan) |
| `password authentication failed` | Contraseña sin codificar | Ver 1.3 |
| `/health` tarda o da 502 los primeros 2 min | Carga del modelo | Espera; el frontend ya lo contempla |
| El Space "duerme" | 48 h sin visitas | Normal en el plan gratuito; despierta solo al primer acceso |

---

## 4 · Frontend en Render

### 4.1 Crear el static site desde el blueprint

1. [dashboard.render.com](https://dashboard.render.com) → **New +** → **Blueprint**.
2. Conecta tu cuenta de GitHub si no lo está y elige el repositorio `Project_imagen_search_system`.
3. **Branch:** `modernizacion` (cámbiala a `main` después del paso 7). Render detecta `render.yaml`.
4. Te pedirá el valor de `VITE_API_URL` (está marcado `sync: false`): pon la URL del Space **sin barra final**, por ejemplo `https://<usuario>-image-search.hf.space`.
5. **Apply**. El build tarda 1 a 2 minutos.

### 4.2 Verificar

Render asigna una URL como `https://image-search-frontend.onrender.com`. Ábrela:

- La barra superior debe pasar de "Conectando con el servidor…" a **"En línea · 6 513 imágenes · pgvector"** (si el Space está dormido, primero "Despertando el servidor…" durante 1 a 2 min).
- Busca "un gato durmiendo". Deben aparecer 6 tarjetas con imágenes de COCO y chips de categorías.
- Pestaña **Por imagen**: arrastra una foto y busca similares.

**Errores comunes**

| Síntoma | Causa | Solución |
|---------|-------|----------|
| Estado en rojo y consola con `CORS policy` | `CORS_ORIGINS` del Space no incluye tu dominio de Render | Paso 5 |
| Consola con `Mixed Content` | `VITE_API_URL` con `http://` | Debe ser `https://` |
| "No se pudo conectar" siempre | `VITE_API_URL` mal escrita o con barra final | Corrige en **Environment** y **Manual Deploy → Clear build cache & deploy** |
| Las imágenes no cargan pero la API responde | Bloqueador de contenido o red corporativa filtrando `images.cocodataset.org` | Prueba en otra red |

---

## 5 · Cerrar CORS y prueba de punta a punta

1. En el Space → **Settings → Variables** → edita `CORS_ORIGINS` con el dominio exacto de Render:
   ```
   https://image-search-frontend.onrender.com
   ```
   Si además quieres seguir probando en local, sepáralos por coma:
   ```
   https://image-search-frontend.onrender.com,http://localhost:5173
   ```
2. El Space se reinicia (1 a 2 min).
3. Vuelve a abrir el frontend en Render y repite las dos búsquedas. Abre las herramientas de desarrollador (F12) → pestaña **Network**: las peticiones a `/search/text` deben responder `200`.

Checklist de la prueba final:

- [ ] `/health` responde `index_loaded: true` con el número correcto de imágenes.
- [ ] Búsqueda por texto en español devuelve resultados coherentes.
- [ ] Búsqueda por texto en inglés también.
- [ ] Búsqueda por imagen con una foto tuya devuelve imágenes del mismo tipo.
- [ ] Selector de resultados (6 / 12 / 24 / 48) cambia la cantidad de tarjetas.
- [ ] Un archivo que no es imagen muestra el mensaje de error, no rompe la página.
- [ ] Funciona en el móvil (la interfaz es responsive).

---

## 6 · README final: enlaces, capturas y GIF

### 6.1 Enlaces del demo

En `README.md`, sección **🎬 Demo**, reemplaza:

```markdown
> **Demo en vivo:** _pendiente de desplegar_ · **API:** _pendiente_
```

por:

```markdown
> **Demo en vivo:** https://image-search-frontend.onrender.com · **API (Swagger):** https://<usuario>-image-search.hf.space/docs
```

Y en la cabecera, el enlace `[🚀 Demo en vivo](#-demo)` puede apuntar directo a la URL de Render.

### 6.2 Capturas

Toma dos capturas del demo desplegado a 1400 px de ancho aproximadamente (Windows: `Win + Shift + S`):

- `docs/screenshot-text.png`: resultados de una búsqueda por texto.
- `docs/screenshot-image.png`: pestaña imagen con la vista previa y los resultados.

### 6.3 GIF (opcional pero vende mucho)

1. Instala [ScreenToGif](https://www.screentogif.com/) (gratis).
2. Graba 10 a 15 segundos: escribir una consulta → resultados → cambiar a "Por imagen" → soltar una foto → resultados.
3. Exporta a `docs/demo.gif` con ancho 900 px y calidad media (objetivo: menos de 5 MB para que GitHub lo muestre fluido).
4. En el README, descomenta el bloque de la sección Demo:

```markdown
<p align="center"><img src="docs/demo.gif" width="800" alt="Demo del buscador" /></p>
```

### 6.4 Badge de Hugging Face (opcional)

Junto a los badges de la cabecera:

```markdown
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-image--search-yellow)](https://huggingface.co/spaces/<usuario>/image-search)
```

### 6.5 Subir

```bash
git add README.md docs/
git commit -m "README: enlaces del demo, capturas y GIF"
git push
git push hf modernizacion:main     # para que el Space también tenga el README final
```

---

## 7 · Fusionar `modernizacion` en `main`

**Opción A · Pull request (recomendado, queda en el historial)**

1. Abre https://github.com/gcdavidq/Project_imagen_search_system/pull/new/modernizacion
2. Título: `Modernización: backend unificado, FAISS offline, frontend de una página, despliegue`.
3. Espera a que el check **CI** esté en verde (lint + pruebas + build del frontend, ~2 min).
4. **Merge pull request** → **Confirm**. Puedes borrar la rama después.

**Opción B · Desde la terminal**

```bash
git checkout main
git pull
git merge --no-ff modernizacion -m "Merge modernizacion: proyecto listo para portafolio"
git push origin main
```

**Después del merge**

- En Render: **Settings → Build & Deploy → Branch** → cámbiala a `main`.
- En Hugging Face: a partir de ahora publica con `git push hf main:main`.
- Ajusta en `.github/workflows/ci.yml` la lista de ramas si quieres quitar `modernizacion`.
- En GitHub, en la página del repo → engranaje junto a **About** → pega la URL del demo en **Website** y añade *topics*: `clip`, `pgvector`, `fastapi`, `vector-search`, `multimodal`, `image-search`, `supabase`.

---

## 8 · Mantenimiento y extras

### Cosas que conviene saber de los planes gratuitos

| Servicio | Comportamiento | Qué hacer |
|----------|----------------|-----------|
| **Supabase** | Pausa el proyecto tras **7 días sin actividad**. El backend devolverá 503. | Entra al dashboard y pulsa **Restore project** (1 min). Para evitarlo, visita el demo cada semana o programa un ping semanal al endpoint `/health` (p. ej. con [cron-job.org](https://cron-job.org)). |
| **Hugging Face Space** | Se duerme tras 48 h sin visitas; despierta en 1 a 2 min. | El frontend ya lo indica. Un ping diario a `/health` lo mantiene despierto. |
| **Render static** | Sin límites relevantes para un sitio estático. | — |

### Antes de una entrevista o de compartir el enlace

1. Abre el demo 5 minutos antes para despertar el Space.
2. Comprueba `/health`. Si `index_loaded` es `false`, es Supabase pausado: restáuralo.

### Si cambias el dataset o el modelo

```bash
python scripts/download_coco.py ...     # solo si cambia el dataset
python scripts/build_index.py           # reconstruye todo el índice (TRUNCATE + reinserción)
```

Si cambias `MODEL_NAME`/`PRETRAINED`, actualiza también las variables del Space y haz **Factory reboot** para que el Dockerfile pre-descargue el nuevo modelo (`ARG MODEL_NAME` y `ARG PRETRAINED` en el `Dockerfile`).

### Roadmap (pendiente de decidir)

Quedaron fuera de esta fase por alcance; están en el README como roadmap:

- Lightbox al hacer clic en un resultado y "buscar similares" desde la propia tarjeta (reutiliza `POST /search/image` descargando la imagen del CDN).
- Sección "cómo funciona" animada dentro del frontend con los `took_ms` reales.
- Filtro por categoría (la API ya expone `/stats`; faltaría un parámetro `category` en la búsqueda y un `WHERE categories ? %s` en SQL).

### Limpieza local (opcional)

```bash
docker rmi image-search-backend      # libera ~3 GB
rmdir /s /q data                     # si ya no necesitas las imágenes en local (se pueden volver a descargar)
```
