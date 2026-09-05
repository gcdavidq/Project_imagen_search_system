# Pasos pendientes para publicar el proyecto

Guía detallada de todo lo que falta para pasar de la rama `modernizacion` a un demo público. Cada paso tiene comandos exactos, qué tocar en cada plataforma, cómo comprobar que funcionó y los errores más comunes.

**Tiempo total estimado:** 2 a 3 horas, la mayor parte esperando descargas y builds.

| # | Paso | Dónde | Tiempo | Depende de |
|---|------|-------|--------|------------|
| — | [Cómo encajan las piezas](#cómo-encajan-las-piezas) | Leer primero | 5 min | — |
| 0 | [Cuentas y herramientas](#0--cuentas-y-herramientas) | Local | 10 min | — |
| 1 | [Base de datos en Neon](#1--base-de-datos-en-neon) | neon.tech | 5 min | 0 |
| 2 | [Dataset e indexación](#2--dataset-e-indexación) | Local | 30 a 60 min | 1 |
| 3 | [Backend en Hugging Face Spaces](#3--backend-en-hugging-face-spaces) | huggingface.co | 20 min + build | 1, 2 |
| 4 | [Frontend en Render](#4--frontend-en-render) | render.com | 10 min | 3 |
| 5 | [CORS y prueba final](#5--cerrar-cors-y-prueba-de-punta-a-punta) | HF + navegador | 10 min | 3, 4 |
| 6 | [README final](#6--readme-final-enlaces-capturas-y-gif) | Local | 20 min | 5 |
| 7 | [Fusionar en `main`](#7--fusionar-modernizacion-en-main) | GitHub | 5 min | 6 |
| 8 | [Mantenimiento](#8--mantenimiento-y-extras) | — | opcional | 7 |

---

## Cómo encajan las piezas

Antes de tocar nada, el modelo mental. **Hay dos momentos distintos** y confundirlos es la principal fuente de dudas.

### Momento A · Preparación (una sola vez, en tu PC)

Esto lo ejecutas tú a mano y **no forma parte del despliegue**:

```mermaid
graph LR
    A["scripts/download_coco.py<br/><i>en tu PC</i>"] -->|"descarga ~6 500 JPG"| B["data/images/<br/><i>carpeta local</i>"]
    A -->|"guarda categorías + URL del CDN"| C["data/metadata.json"]
    B --> D["scripts/build_index.py<br/><i>en tu PC</i>"]
    C --> D
    D -->|"CLIP: cada imagen → vector 512-d"| E[("Neon<br/>tabla images")]
```

El resultado es que **Neon queda con una fila por imagen**: su vector, sus categorías y **la URL pública de la imagen**. Las imágenes en sí se quedan en tu disco y ya no hacen falta.

### Momento B · Producción (cada vez que alguien busca)

```mermaid
graph LR
    U["Navegador<br/><i>frontend en Render</i>"] -->|"1 · 'un gato durmiendo'"| S["Backend en HF Spaces<br/><i>modelo CLIP</i>"]
    S -->|"2 · texto → vector 512-d"| S
    S -->|"3 · vecinos más cercanos"| N[("Neon<br/>pgvector + HNSW")]
    N -->|"4 · URLs + score + categorías"| S
    S -->|"5 · JSON"| U
    U -->|"6 · descarga las fotos"| C["images.cocodataset.org<br/><i>CDN de MS COCO</i>"]
```

Fíjate en el paso 6: **las imágenes viajan del CDN de COCO al navegador del usuario sin pasar por tu servidor**.

### Qué guarda cada servicio

| Servicio | Qué almacena | Tamaño | ¿Le importan las imágenes? |
|----------|--------------|--------|----------------------------|
| **Neon** | Vectores (512 floats), categorías y el **texto de la URL** de cada imagen | ~13 MB de vectores; con índice y metadatos, unas decenas de MB | No. Solo guarda la cadena `https://images.cocodataset.org/...` |
| **HF Spaces** (backend) | Código + modelo CLIP (~1 GB) | ~3 GB de imagen Docker | No. Su `data/` está vacío en producción |
| **Render** (frontend) | HTML, CSS y JS compilados | ~30 KB | No. Solo pide las URLs que le da la API |
| **CDN de MS COCO** | Las fotos reales | 18 GB (no tuyos) | Es quien las sirve, gratis y sin registro |

---

## ❓ Tus tres dudas, respondidas

### 1. ¿Ejecuto `python scripts/download_coco.py` manualmente?

**Sí, una sola vez, en tu PC**, junto con `build_index.py`. Es el "Momento A" de arriba.

No se ejecuta en el despliegue por dos razones: descargar 6 500 imágenes y calcular sus embeddings tarda entre 30 y 60 minutos en CPU (un contenedor de HF Spaces se reiniciaría antes de terminar), y el resultado es siempre el mismo. Es trabajo de preparación, no de tiempo de ejecución.

Lo repites solo si cambias el dataset (más imágenes, otras categorías) o el modelo CLIP. Detalle en el [paso 2](#2--dataset-e-indexación).

### 2. ¿Para qué sirve Hugging Face Spaces?

Es donde vive **el cerebro**: el proceso que convierte texto e imágenes en vectores.

Ese proceso necesita PyTorch y el modelo CLIP cargado en memoria, lo que ocupa **unos 2 GB de RAM**. Esa es toda la razón por la que no está en Render:

| Plataforma | RAM en plan gratuito | ¿Sirve para este backend? |
|------------|----------------------|----------------------------|
| Render (web service) | 512 MB | ❌ El proceso muere al cargar el modelo |
| Railway (trial) | 512 MB – 1 GB | ❌ Justo o insuficiente |
| Fly.io | 256 MB por VM gratis | ❌ |
| **Hugging Face Spaces** | **16 GB, 2 vCPU** | ✅ Y está pensado para demos de ML |

Spaces te da un contenedor Docker gratis con hardware suficiente, una URL pública `https://<usuario>-<space>.hf.space`, y logs. A cambio, el contenedor se duerme tras 48 h sin visitas y tarda 1 a 2 minutos en despertar (el frontend ya lo maneja mostrando "Despertando el servidor…").

En resumen: **Render sirve el frontend** (archivos estáticos, no necesita RAM), **Spaces sirve el backend** (necesita RAM para el modelo) y **Neon guarda los vectores**.

### 3. ¿Dónde se almacenan las imágenes? ¿Necesito S3?

**No necesitas S3 ni ningún almacenamiento de objetos.** Las imágenes de MS COCO **ya están alojadas públicamente** por el propio proyecto COCO, en `https://images.cocodataset.org/`, y el sistema aprovecha eso.

Cómo funciona, concretamente:

1. `download_coco.py` guarda en `metadata.json`, junto a las categorías, la URL pública de cada imagen:
   ```json
   "coco_000000000009.jpg": {
     "categories": ["bowl", "broccoli", "orange"],
     "url": "https://images.cocodataset.org/train2017/000000000009.jpg"
   }
   ```
2. `build_index.py` escribe esa URL en la columna `image_url` de la tabla `images` en Neon.
3. La API la devuelve tal cual en cada resultado.
4. El navegador del usuario descarga la foto **directamente del CDN de COCO**.

Consecuencias prácticas:

- Tu backend **nunca sirve bytes de imagen** en producción: ni almacenamiento, ni ancho de banda, ni coste.
- La carpeta `data/images/` **solo existe en tu PC** y solo se usa para calcular los embeddings. Después de `build_index.py` puedes borrarla (o conservarla por si quieres reindexar sin volver a descargar).
- Añadir S3 significaría **pagar por duplicar archivos que ya son públicos y gratuitos**.

Y la imagen que el usuario sube para buscar por similitud tampoco se guarda: se lee en memoria, se convierte en un vector y se descarta. Por eso el README puede decir "sin registro, sin tracking" con honestidad. Lo puedes comprobar en [backend/routes/image_search.py](../backend/routes/image_search.py): los bytes van a `load_image_from_bytes()` y nunca a disco.

**El único riesgo real** es depender de un CDN ajeno: si `images.cocodataset.org` se cae o bloquea el hotlinking, las tarjetas se ven vacías (la búsqueda sigue funcionando, porque los vectores están en Neon). Si eso te preocupa o quieres ser autónomo, hay una alternativa gratuita en el [anexo](#anexo--si-algún-día-quieres-alojar-tú-las-imágenes).

**Cuándo sí necesitarías almacenamiento de objetos:** si el catálogo fueran fotos tuyas o de usuarios, o un dataset privado. Ahí sí harían falta S3, Cloudflare R2 o similar. No es este caso.

---

## 0 · Cuentas y herramientas

Cuentas gratuitas necesarias:

- [neon.tech](https://neon.tech) (puedes entrar con GitHub) — ✅ ya la tienes.
- [huggingface.co/join](https://huggingface.co/join).
- [dashboard.render.com/register](https://dashboard.render.com/register).

En tu máquina ya tienes Python 3.13, Node 22, Docker Desktop y Git. Verifica que estás en la rama correcta:

```bash
cd C:\Users\gcdav\Documents\PERSONAL\PROYECTOS\Project_imagen_search_system
git checkout modernizacion
git pull
git status          # debe decir "nothing to commit, working tree clean"
```

---

## 1 · Base de datos en Neon

> ✅ **Ya hecho.** Tu proyecto de Neon existe y está verificado: PostgreSQL 18.6, pgvector 0.8.6, índice HNSW soportado, y el archivo `.env` local ya tiene la cadena de conexión. Si alguna vez tienes que rehacerlo, aquí están los pasos.

### 1.1 Crear el proyecto

1. [console.neon.tech](https://console.neon.tech) → **New project**.
2. Nombre: `image-search`. Región: la más cercana (tú usas `sa-east-1`, São Paulo). Postgres 17 o 18.
3. Plan **Free**.

### 1.2 Obtener la cadena de conexión

En el dashboard, botón **Connect** → copia la URI. Neon ofrece dos endpoints:

| Endpoint | Cuándo usarlo |
|----------|---------------|
| **`-pooler`** (recomendado, el que ya usas) | Todo: indexar desde tu PC y el backend en producción. Usa PgBouncer, aguanta muchas conexiones cortas. |
| Sin `-pooler` (direct) | Migraciones largas o herramientas que necesiten sesiones persistentes. No hace falta aquí. |

Tu cadena tiene esta forma:

```
postgresql://neondb_owner:CONTRASEÑA@ep-lingering-art-acbsnj3m-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

**A diferencia de Supabase, la misma cadena vale para los dos usos**, así que no tienes que manejar dos puertos distintos.

### 1.3 Verificar

La extensión `vector` y la tabla `images` las crea el propio backend al arrancar. Si quieres comprobarlo a mano, en el **SQL Editor** de Neon:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname = 'vector';   -- 0.8.6
```

### 1.4 Diferencia importante con Supabase (para bien)

| | Supabase Free | Neon Free |
|---|---|---|
| Inactividad | **Pausa el proyecto tras 7 días** y hay que restaurarlo a mano desde el dashboard | **Suspende solo el cómputo** tras unos minutos y **despierta automáticamente** en la siguiente consulta, en menos de un segundo |
| Efecto en tu demo | El backend devolvía 503 hasta que entraras a restaurarlo | La primera búsqueda tras la inactividad tarda un poco más y ya |

Es decir: con Neon te quitas el mantenimiento semanal que tenía la guía anterior. Revisa los límites vigentes del plan en [neon.tech/pricing](https://neon.tech/pricing); tus datos ocupan unas decenas de MB, así que entran de sobra.

---

## 2 · Dataset e indexación

Este es el "Momento A". Se hace **una vez**.

### 2.1 El `.env` ya está configurado

Tu `.env` local (que **no se sube al repositorio**) ya contiene:

```ini
INDEX_BACKEND=pgvector
DATABASE_URL=postgresql://neondb_owner:...@ep-lingering-art-acbsnj3m-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
MODEL_NAME=xlm-roberta-base-ViT-B-32
PRETRAINED=laion5b_s13b_b90k
```

### 2.2 Descargar el subset de COCO

Elige tamaño según cuánto quieras esperar:

| Comando | Imágenes | Descarga | Indexación en CPU | Espacio en disco |
|---------|----------|----------|-------------------|------------------|
| `python scripts/download_coco.py --split val --images-per-cat 40` | ~2 500 | 5 min | ~10 min | ~400 MB |
| `python scripts/download_coco.py` (train, 100/categoría) | ~6 500 | 15 min | 30 a 45 min | ~1 GB |

Para el portafolio recomiendo el segundo: más variedad, resultados más convincentes.

```bash
pip install requests tqdm          # si aún no los tienes
python scripts/download_coco.py
```

El script descarga el ZIP de anotaciones (~250 MB) y parsea un JSON de 450 MB **en memoria**; cierra programas pesados si tienes menos de 8 GB de RAM, o usa `--split val`.

**Verificación:** `data/images/` tiene miles de `.jpg` y existe `data/metadata.json`. Ábrelo y comprueba que cada entrada tiene `categories` y `url` empezando por `https://images.cocodataset.org/`. **Esa URL es la clave de todo el tema del almacenamiento**: sin ella, el backend tendría que servir las imágenes él mismo.

### 2.3 Generar embeddings y subirlos a Neon

Dos formas. La **opción B** evita instalar PyTorch en tu máquina.

**Opción A · Python local**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # ~800 MB de PyTorch, una sola vez
python scripts/build_index.py          # la primera vez descarga el modelo (~1 GB)
```

**Opción B · Con la imagen Docker ya construida** (PowerShell)

```powershell
docker run --rm `
  -v "${PWD}\data:/app/data" `
  --env-file .env `
  image-search-backend python scripts/build_index.py
```

Si borraste la imagen: `docker build -t image-search-backend .` (unos 10 min).

Salida esperada al final:

```
Embeddings generados: 6513 imágenes × 512 dimensiones.
Insertando 6513 vectores en PostgreSQL ...
Vectores insertados e índice HNSW verificado.
Índice construido en 'pgvector' con 6513 imágenes.
```

**Verificación en Neon** (SQL Editor):

```sql
SELECT count(*) FROM images;
SELECT image_path, image_url, categories FROM images LIMIT 3;   -- image_url NO debe ser NULL
SELECT indexname FROM pg_indexes WHERE tablename = 'images';    -- images_embedding_hnsw_idx
```

Si `image_url` sale `NULL`, el `metadata.json` es de una versión antigua: vuelve a ejecutar `download_coco.py` y luego `build_index.py`.

**Errores comunes**

| Error | Causa | Solución |
|-------|-------|----------|
| `password authentication failed` | Contraseña mal copiada | Vuelve a copiar la URI desde **Connect** en Neon |
| `SSL SYSCALL error` a mitad de la inserción | Red inestable o cómputo suspendido | Vuelve a ejecutar; el script hace `TRUNCATE` y reinserta todo |
| `channel_binding` no reconocido | libpq antiguo | Quita `&channel_binding=require` de la URL; `sslmode=require` es suficiente |
| Se cae por memoria en `download_coco.py` | El JSON de anotaciones ocupa 450 MB | Usa `--split val` |

### 2.4 Probar el índice sin servidor

```bash
python scripts/search_cli.py
Consulta > un perro corriendo en la playa
```

Los primeros resultados deben tener categorías coherentes (`dog`, `person`, `frisbee`). Si salen aleatorios, el índice se construyó con un modelo distinto al de `.env`: reconstrúyelo.

### 2.5 ¿Puedo borrar `data/images/` después?

Sí. Una vez que Neon tiene los vectores y las URLs, el demo funciona sin esa carpeta. Consérvala solo si piensas reindexar pronto (te ahorra volver a descargar). El repositorio ya la ignora en `.gitignore`, así que nunca se sube.

---

## 3 · Backend en Hugging Face Spaces

### 3.1 Crear el Space

1. [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Space name:** `image-search` → la URL será `https://<usuario>-image-search.hf.space`.
3. **License:** MIT. **SDK:** **Docker** → plantilla **Blank**.
4. **Hardware:** `CPU basic · 2 vCPU · 16 GB · FREE`.
5. **Visibility:** Public → **Create Space**.

### 3.2 Token de escritura

**Settings** (tu avatar) → **Access Tokens** → **Create new token** → tipo *Write* → nombre `image-search-deploy`. Cópialo: es la contraseña para el `git push` al Space.

### 3.3 Metadatos del Space en el README

Hugging Face lee la configuración del Space de un bloque YAML al **inicio** de `README.md`. Añádelo como primera línea del archivo, antes del `<div align="center">`:

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

GitHub lo muestra como una tabla pequeña arriba del banner; es lo normal en proyectos alojados en Spaces.

```bash
git add README.md && git commit -m "Metadatos del Space de Hugging Face" && git push
```

### 3.4 Subir el código al Space

```bash
git remote add hf https://huggingface.co/spaces/<usuario>/image-search
git push hf modernizacion:main
```

Usuario = tu usuario de HF; contraseña = el token del paso 3.2. Si el push es rechazado porque el Space se creó con su propio README, usa `git push --force hf modernizacion:main` (el Space está vacío, no pierdes nada).

### 3.5 Variables y secretos

En el Space: **Settings → Variables and secrets**:

| Tipo | Nombre | Valor |
|------|--------|-------|
| **Secret** | `DATABASE_URL` | Tu cadena de Neon completa, la misma del `.env` |
| Variable | `INDEX_BACKEND` | `pgvector` |
| Variable | `CORS_ORIGINS` | `*` por ahora; se ajusta en el [paso 5](#5--cerrar-cors-y-prueba-de-punta-a-punta) |

> ⚠️ `DATABASE_URL` debe ir como **Secret**, no como Variable: las Variables son visibles públicamente en un Space público.

### 3.6 Esperar el build y verificar

Pestaña **Logs**. El build tarda ~10 minutos (instala PyTorch y pre-descarga el modelo). Luego, en los logs del contenedor:

```
Arrancando: modelo CLIP 'xlm-roberta-base-ViT-B-32' + backend 'pgvector' ...
Modelo CLIP cargado. Dimensión de incrustación: 512
PostgreSQL listo: 6513 vectores en la tabla images.
Application startup complete.
```

**Verificación:**

- `https://<usuario>-image-search.hf.space/health` → `"model_loaded": true, "index_loaded": true, "index_size": 6513`.
- `https://<usuario>-image-search.hf.space/docs` → Swagger. Prueba `POST /search/text` con `{"query": "un gato", "top_k": 3}`. En la respuesta, cada `image_url` debe ser una URL de `images.cocodataset.org`.

**Errores comunes**

| Síntoma | Causa | Solución |
|---------|-------|----------|
| Build falla en `pip install` | Caída puntual de PyPI | **Factory reboot** en Settings |
| `index_loaded: false` y en logs `Base de datos no configurada` | Falta `DATABASE_URL` | Revisa 3.5 |
| `/health` da 502 los primeros 2 min | Carga del modelo | Espera; el frontend ya lo contempla |
| Las búsquedas tardan 3 s la primera vez | Cómputo de Neon despertando + calentamiento de PyTorch | Normal; las siguientes bajan a cientos de ms |

---

## 4 · Frontend en Render

### 4.1 Crear el static site

1. [dashboard.render.com](https://dashboard.render.com) → **New +** → **Blueprint**.
2. Conecta GitHub y elige `Project_imagen_search_system`.
3. **Branch:** `modernizacion` (cámbiala a `main` tras el paso 7). Render lee `render.yaml`.
4. Pedirá `VITE_API_URL`: la URL del Space **sin barra final**, p. ej. `https://<usuario>-image-search.hf.space`.
5. **Apply**. Build de 1 a 2 minutos.

### 4.2 Verificar

Render asigna algo como `https://image-search-frontend.onrender.com`:

- La barra superior debe pasar de "Conectando…" a **"En línea · 6 513 imágenes · pgvector"**.
- Busca "un gato durmiendo": 6 tarjetas con fotos y chips de categorías.
- Pestaña **Por imagen**: arrastra una foto y busca similares.

**Errores comunes**

| Síntoma | Causa | Solución |
|---------|-------|----------|
| Estado en rojo, consola con `CORS policy` | `CORS_ORIGINS` no incluye tu dominio | [Paso 5](#5--cerrar-cors-y-prueba-de-punta-a-punta) |
| Consola con `Mixed Content` | `VITE_API_URL` con `http://` | Debe ser `https://` |
| "No se pudo conectar" siempre | `VITE_API_URL` mal escrita o con barra final | Corrige en **Environment** → **Manual Deploy → Clear build cache & deploy** |
| La API responde pero las tarjetas salen vacías | El navegador no llega al CDN de COCO (bloqueador, red corporativa) | Prueba en otra red; ver el [anexo](#anexo--si-algún-día-quieres-alojar-tú-las-imágenes) |

---

## 5 · Cerrar CORS y prueba de punta a punta

1. En el Space → **Settings → Variables** → `CORS_ORIGINS` con el dominio exacto de Render:
   ```
   https://image-search-frontend.onrender.com
   ```
   Para seguir probando en local, sepáralos por coma:
   ```
   https://image-search-frontend.onrender.com,http://localhost:5173
   ```
2. El Space se reinicia (1 a 2 min).
3. Recarga el frontend y repite las búsquedas. Con F12 → **Network**, las peticiones a `/search/text` deben responder `200`.

Checklist final:

- [ ] `/health` responde `index_loaded: true` con el número correcto de imágenes.
- [ ] Búsqueda por texto en español devuelve resultados coherentes.
- [ ] Búsqueda por texto en inglés también.
- [ ] Búsqueda por imagen con una foto tuya devuelve imágenes del mismo tipo.
- [ ] El selector 6 / 12 / 24 / 48 cambia la cantidad de tarjetas.
- [ ] Un archivo que no es imagen muestra el error sin romper la página.
- [ ] Funciona en el móvil.

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

### 6.2 Capturas

Dos capturas del demo desplegado, ~1400 px de ancho (`Win + Shift + S`):

- `docs/screenshot-text.png`: resultados de una búsqueda por texto.
- `docs/screenshot-image.png`: pestaña imagen con vista previa y resultados.

### 6.3 GIF (opcional, pero vende mucho)

1. Instala [ScreenToGif](https://www.screentogif.com/).
2. Graba 10 a 15 s: escribir consulta → resultados → cambiar a "Por imagen" → soltar foto → resultados.
3. Exporta a `docs/demo.gif`, ancho 900 px, menos de 5 MB.
4. Descomenta en el README el bloque de la sección Demo:
   ```markdown
   <p align="center"><img src="docs/demo.gif" width="800" alt="Demo del buscador" /></p>
   ```

### 6.4 Badges opcionales

```markdown
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-image--search-yellow)](https://huggingface.co/spaces/<usuario>/image-search)
[![Neon](https://img.shields.io/badge/Neon-Postgres-00E599?logo=postgresql&logoColor=white)](https://neon.tech)
```

### 6.5 Subir

```bash
git add README.md docs/
git commit -m "README: enlaces del demo, capturas y GIF"
git push
git push hf modernizacion:main     # para que el Space tenga el README final
```

---

## 7 · Fusionar `modernizacion` en `main`

**Opción A · Pull request (recomendado)**

1. Abre https://github.com/gcdavidq/Project_imagen_search_system/pull/new/modernizacion
2. Título: `Modernización: backend unificado, FAISS offline, frontend de una página, despliegue`.
3. Espera el check **CI** en verde (~2 min).
4. **Merge pull request** → **Confirm**.

**Opción B · Terminal**

```bash
git checkout main
git pull
git merge --no-ff modernizacion -m "Merge modernizacion: proyecto listo para portafolio"
git push origin main
```

**Después del merge**

- Render: **Settings → Build & Deploy → Branch** → cámbiala a `main`.
- Hugging Face: a partir de ahora `git push hf main:main`.
- En `.github/workflows/ci.yml` puedes quitar `modernizacion` de la lista de ramas.
- GitHub → engranaje junto a **About** → pega la URL del demo en **Website** y añade *topics*: `clip`, `pgvector`, `fastapi`, `vector-search`, `multimodal`, `image-search`, `neon`.

---

## 8 · Mantenimiento y extras

### Comportamiento de los planes gratuitos

| Servicio | Comportamiento | Qué hacer |
|----------|----------------|-----------|
| **Neon** | Suspende el cómputo tras unos minutos sin consultas y **despierta solo** en menos de un segundo | Nada |
| **HF Space** | Se duerme tras 48 h sin visitas; despierta en 1 a 2 min | Nada; el frontend lo indica. Un ping diario a `/health` (p. ej. con [cron-job.org](https://cron-job.org)) lo mantiene despierto |
| **Render static** | Sin límites relevantes | Nada |

### Antes de una entrevista o de compartir el enlace

Abre el demo 5 minutos antes para despertar el Space y comprueba `/health`.

### Si cambias el dataset o el modelo

```bash
python scripts/download_coco.py ...     # solo si cambia el dataset
python scripts/build_index.py           # reconstruye el índice (TRUNCATE + reinserción)
```

Si cambias `MODEL_NAME` / `PRETRAINED`, actualiza también las variables del Space y haz **Factory reboot**, para que el Dockerfile pre-descargue el nuevo modelo (`ARG MODEL_NAME` y `ARG PRETRAINED` en el [Dockerfile](../Dockerfile)).

### Seguridad de las credenciales

- `.env` está en `.gitignore` y nunca se sube. Verifícalo con `git check-ignore -v .env`.
- En Neon puedes rotar la contraseña cuando quieras: **Dashboard → Roles → Reset password**. Hazlo si la cadena se te escapó a un chat, una captura o un log. Después actualiza `.env` y el secret del Space.

### Roadmap pendiente

- Lightbox al hacer clic en un resultado y "buscar similares" desde la propia tarjeta.
- Sección "cómo funciona" animada dentro del frontend con los `took_ms` reales.
- Filtro por categoría (la API ya expone `/stats`; faltaría un parámetro `category` y un `WHERE categories ? %s` en SQL).

### Limpieza local (opcional)

```bash
docker rmi image-search-backend      # libera ~3 GB
rmdir /s /q data                     # las imágenes se pueden volver a descargar
```

---

## Anexo · Si algún día quieres alojar tú las imágenes

Solo tiene sentido si el CDN de COCO te falla, o si cambias a un dataset propio. La opción **gratuita** que mejor encaja con este proyecto es un **Dataset repo de Hugging Face**, no S3:

| Opción | Coste | Notas |
|--------|-------|-------|
| **HF Dataset repo** | Gratis | Repo git con LFS. Sirve archivos en `https://huggingface.co/datasets/<usuario>/<nombre>/resolve/main/<archivo>`. Mismo ecosistema que tu Space |
| **Cloudflare R2** | Gratis hasta 10 GB, **sin coste de egreso** | La mejor opción de tipo S3 para servir imágenes |
| **AWS S3** | Almacenamiento barato pero **se paga el egreso** | El tráfico de un demo público puede sorprender |

El cambio en el código sería mínimo: `download_coco.py` escribiría en `metadata.json` la URL de tu almacenamiento en lugar de la de COCO, y se reindexa. Nada más: el resto del sistema ya trabaja con la columna `image_url`. Si llegas a necesitarlo, dilo y lo implemento.
