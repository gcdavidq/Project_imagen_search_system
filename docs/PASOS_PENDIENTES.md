# Pasos pendientes para publicar el proyecto

Guía detallada de todo lo que falta para pasar de la rama `modernizacion` a un demo público. Cada paso tiene comandos exactos, cómo comprobar que funcionó y los errores más comunes.

**Tiempo total estimado:** 2 a 3 horas, la mayor parte esperando descargas y builds.

| # | Paso | Dónde | Tiempo | Depende de |
|---|------|-------|--------|------------|
| — | [Cómo encajan las piezas](#cómo-encajan-las-piezas) | Leer primero | 5 min | — |
| 0 | [Cuentas y herramientas](#0--cuentas-y-herramientas) | Local | 10 min | — |
| 1 | [Base de datos en Neon](#1--base-de-datos-en-neon) | neon.tech | 5 min | 0 |
| 2 | [Dataset e indexación](#2--dataset-e-indexación) | Local | 30 a 60 min | 1 |
| 3 | [Backend en Google Cloud Run](#3--backend-en-google-cloud-run) | Google Cloud | 30 min + build | 1, 2 |
| 4 | [Frontend en Render](#4--frontend-en-render) | render.com | 10 min | 3 |
| 5 | [CORS y prueba final](#5--cerrar-cors-y-prueba-de-punta-a-punta) | Cloud Run + navegador | 10 min | 3, 4 |
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
    U["Navegador<br/><i>frontend en Render</i>"] -->|"1 · 'un gato durmiendo'"| S["Backend en Cloud Run<br/><i>modelo CLIP</i>"]
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
| **Cloud Run** (backend) | Código + modelo CLIP (~1,6 GB) dentro de la imagen Docker | ~3 GB de imagen | No. Su `data/` está vacío en producción |
| **Render** (frontend) | HTML, CSS y JS compilados | ~30 KB | No. Solo pide las URLs que le da la API |
| **CDN de MS COCO** | Las fotos reales | 18 GB (no tuyos) | Es quien las sirve, gratis y sin registro |

---

## ❓ Dudas frecuentes, respondidas

### 1. ¿Ejecuto `python scripts/download_coco.py` manualmente?

**Sí, una sola vez, en tu PC**, junto con `build_index.py`. Es el "Momento A" de arriba.

No se ejecuta en el despliegue por dos razones: descargar 6 500 imágenes y calcular sus embeddings tarda entre 30 y 60 minutos (un contenedor se reiniciaría antes de terminar), y el resultado es siempre el mismo. Es trabajo de preparación, no de tiempo de ejecución.

Lo repites solo si cambias el dataset o el modelo CLIP. Detalle en el [paso 2](#2--dataset-e-indexación).

### 2. ¿Por qué el backend va en Cloud Run y no junto al frontend en Render?

Por **memoria**. El backend carga PyTorch y el modelo CLIP multilingüe. Estas cifras están medidas en un contenedor con 2 vCPU, no estimadas:

| Métrica | Valor medido |
|---------|--------------|
| Tiempo de carga del modelo | 55 s |
| Memoria en régimen | 2 160 MB |
| **Pico al deserializar los pesos** | **3 196 MB** |
| Latencia por consulta | ~0,9 s |

Los planes gratuitos de Render, Koyeb o Fly.io dan **512 MB**. El proceso muere antes de terminar de cargar. Por eso el backend necesita una plataforma que ofrezca varios GB.

**Sobre Hugging Face Spaces:** era la opción prevista en la versión anterior de esta guía, pero **el 8 de julio de 2026 Hugging Face movió los Spaces de Docker y Gradio detrás de un plan de pago**, sin anuncio previo. La documentación oficial dice ahora: *"Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts"*. La única excepción gratuita son 2 Spaces de Gradio sobre ZeroGPU, que no admiten Docker y cuya cuota diaria la consume cada visitante (2 minutos sin autenticar). Para una API pública llamada desde un frontend externo, no es viable.

Cloud Run cubre lo mismo: **Docker nativo, 4 GiB de RAM, escala a cero y un free tier mensual holgado** para un demo.

En resumen: **Render sirve el frontend** (archivos estáticos, sin necesidad de RAM), **Cloud Run sirve el backend** (necesita RAM para el modelo) y **Neon guarda los vectores**.

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

- Tu backend **nunca sirve bytes de imagen** en producción: ni almacenamiento, ni ancho de banda, ni coste. Esto importa especialmente en Cloud Run, donde el tráfico de salida sí se factura.
- La carpeta `data/images/` **solo existe en tu PC** y solo se usa para calcular los embeddings. Después de `build_index.py` puedes borrarla.
- Añadir S3 significaría **pagar por duplicar archivos que ya son públicos y gratuitos**.

Y la imagen que el usuario sube para buscar por similitud tampoco se guarda: se lee en memoria, se convierte en un vector y se descarta. Puedes comprobarlo en [backend/routes/image_search.py](../backend/routes/image_search.py): los bytes van a `load_image_from_bytes()` y nunca a disco.

**El único riesgo real** es depender de un CDN ajeno: si `images.cocodataset.org` se cae, las tarjetas se ven vacías (la búsqueda sigue funcionando, porque los vectores están en Neon). Si quieres ser autónomo, hay alternativas gratuitas en el [anexo](#anexo--si-algún-día-quieres-alojar-tú-las-imágenes).

**Cuándo sí necesitarías almacenamiento de objetos:** si el catálogo fueran fotos tuyas o de usuarios, o un dataset privado. No es este caso.

---

## 0 · Cuentas y herramientas

Cuentas necesarias:

- [neon.tech](https://neon.tech) — ✅ ya la tienes.
- **Google Cloud** ([console.cloud.google.com](https://console.cloud.google.com)) con facturación activada. Requiere tarjeta, aunque el uso de este demo cabe en el free tier permanente. Ver [3.1](#31--cuenta-proyecto-y-control-de-gasto).
- [dashboard.render.com/register](https://dashboard.render.com/register).

Verifica que estás en la rama correcta:

```bash
cd C:\Users\gcdav\Documents\PERSONAL\PROYECTOS\Project_imagen_search_system
git checkout modernizacion
git pull
git status          # "nothing to commit, working tree clean"
```

---

## 1 · Base de datos en Neon

> ✅ **Ya hecho y verificado:** PostgreSQL 18.6, pgvector 0.8.6, índice HNSW soportado, y tu `.env` local ya tiene la cadena de conexión. Se probó una inserción y una búsqueda reales. Si alguna vez tienes que rehacerlo, aquí están los pasos.

### 1.1 Crear el proyecto

1. [console.neon.tech](https://console.neon.tech) → **New project**.
2. Nombre `image-search`, región `sa-east-1` (São Paulo), plan **Free**.

### 1.2 Cadena de conexión

Botón **Connect** → copia la URI del endpoint **`-pooler`**. Sirve tanto para indexar desde tu PC como para el backend en producción:

```
postgresql://neondb_owner:CONTRASEÑA@ep-lingering-art-acbsnj3m-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

### 1.3 Verificar

La extensión `vector` y la tabla `images` las crea el backend al arrancar. Para comprobarlo a mano, en el **SQL Editor** de Neon:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname = 'vector';   -- 0.8.6
```

### 1.4 Ventaja frente a Supabase

| | Supabase Free | Neon Free |
|---|---|---|
| Inactividad | **Pausa el proyecto tras 7 días** y hay que restaurarlo a mano | **Suspende solo el cómputo** y **despierta automáticamente** en la siguiente consulta |
| Efecto en el demo | 503 hasta que entraras a restaurarlo | La primera búsqueda tarda un poco más y ya |

Te quitas el mantenimiento semanal. Límites vigentes en [neon.tech/pricing](https://neon.tech/pricing); tus datos ocupan decenas de MB.

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

| Comando | Imágenes | Descarga | Indexación | Disco |
|---------|----------|----------|------------|-------|
| `python scripts/download_coco.py --split val --images-per-cat 40` | ~2 500 | 5 min | ~10 min | ~400 MB |
| `python scripts/download_coco.py` (train, 100/categoría) | ~6 500 | 15 min | 30 a 45 min | ~1 GB |

Para el portafolio recomiendo el segundo: más variedad, resultados más convincentes.

```bash
pip install requests tqdm          # si aún no los tienes
python scripts/download_coco.py
```

El script parsea un JSON de 450 MB **en memoria**; cierra programas pesados si tienes menos de 8 GB de RAM, o usa `--split val`.

**Verificación:** `data/images/` tiene miles de `.jpg` y existe `data/metadata.json`. Cada entrada debe tener `categories` y `url` empezando por `https://images.cocodataset.org/`. **Esa URL es la clave del tema del almacenamiento.**

### 2.3 Generar embeddings y subirlos a Neon

**Opción A · Python local**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # ~800 MB de PyTorch, una sola vez
python scripts/build_index.py          # la primera vez descarga el modelo (~1,6 GB)
```

**Opción B · Con la imagen Docker ya construida** (PowerShell, evita instalar PyTorch)

```powershell
docker run --rm `
  -v "${PWD}\data:/app/data" `
  --env-file .env `
  image-search-backend python scripts/build_index.py
```

Si borraste la imagen: `docker build -t image-search-backend .` (~10 min).

Salida esperada:

```
Embeddings generados: 6513 imágenes × 512 dimensiones.
Insertando 6513 vectores en PostgreSQL ...
Vectores insertados e índice HNSW verificado.
```

**Verificación en Neon** (SQL Editor):

```sql
SELECT count(*) FROM images;
SELECT image_path, image_url, categories FROM images LIMIT 3;   -- image_url NO debe ser NULL
SELECT indexname FROM pg_indexes WHERE tablename = 'images';    -- images_embedding_hnsw_idx
```

**Errores comunes**

| Error | Causa | Solución |
|-------|-------|----------|
| `password authentication failed` | Contraseña mal copiada | Vuelve a copiar la URI desde **Connect** en Neon |
| `SSL SYSCALL error` a mitad | Red inestable | Vuelve a ejecutar; el script hace `TRUNCATE` y reinserta todo |
| `channel_binding` no reconocido | libpq antiguo | Quita `&channel_binding=require`; `sslmode=require` basta |
| Se cae por memoria al descargar | El JSON de anotaciones ocupa 450 MB | Usa `--split val` |

### 2.4 Probar el índice sin servidor

```bash
python scripts/search_cli.py
Consulta > un perro corriendo en la playa
```

Los primeros resultados deben tener categorías coherentes. Si salen aleatorios, el índice se construyó con un modelo distinto al de `.env`: reconstrúyelo.

### 2.5 ¿Puedo borrar `data/images/` después?

Sí. Una vez que Neon tiene los vectores y las URLs, el demo funciona sin esa carpeta.

---

## 3 · Backend en Google Cloud Run

Cloud Run ejecuta el `Dockerfile` del repositorio tal cual. No hay que cambiar código.

### 3.1 · Cuenta, proyecto y control de gasto

1. Entra a [console.cloud.google.com](https://console.cloud.google.com) con tu cuenta de Google y acepta los términos.
2. **Crea un proyecto** llamado `image-search` (menú desplegable arriba a la izquierda → **New project**).
3. **Activa la facturación**: menú **Billing** → vincula una tarjeta. Google exige un método de pago aunque no vaya a cobrarte dentro del free tier. Las cuentas nuevas reciben además crédito de prueba, y el free tier permanente sigue después.
4. **Pon una alerta de presupuesto antes de desplegar nada.** Es la red de seguridad real:
   **Billing → Budgets & alerts → Create budget** → importe **1 USD** → alertas al 50 %, 90 % y 100 %. Recibirás un correo si algo se sale de lo previsto.

> Una alerta de presupuesto **avisa, no corta el servicio**. Con `--max-instances 2` (paso 3.5) el gasto máximo posible ya queda acotado.

### 3.2 · Cloud Shell (recomendado: no instalas nada)

En la consola, arriba a la derecha, pulsa el icono **Activate Cloud Shell** (`>_`). Se abre una terminal Linux en el navegador con `gcloud`, `git` y Docker ya instalados.

```bash
git clone https://github.com/gcdavidq/Project_imagen_search_system.git
cd Project_imagen_search_system
git checkout modernizacion

gcloud config set project image-search      # o el ID exacto que te asignó Google
```

<details>
<summary>Alternativa: instalar gcloud CLI en Windows</summary>

Descarga el instalador desde [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install), luego:

```powershell
gcloud init
gcloud auth login
gcloud config set project TU_PROJECT_ID
```

Y ejecuta el resto de comandos desde la carpeta del repositorio en tu PC.

</details>

### 3.3 · Habilitar las APIs necesarias

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

Tarda un minuto.

### 3.4 · Guardar la cadena de Neon como secreto

No pongas la contraseña en un comando: quedaría en el historial del shell. Este comando la lee por entrada estándar.

```bash
gcloud secrets create image-search-db --data-file=-
# Pega aquí la URL completa de Neon, pulsa Enter y luego Ctrl+D
```

Da permiso a la cuenta de servicio que ejecuta Cloud Run para leerlo:

```bash
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')

gcloud secrets add-iam-policy-binding image-search-db \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### 3.5 · Desplegar

```bash
gcloud run deploy image-search \
  --source . \
  --region southamerica-east1 \
  --memory 4Gi \
  --cpu 2 \
  --cpu-boost \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 120 \
  --allow-unauthenticated \
  --set-env-vars INDEX_BACKEND=pgvector,CORS_ORIGINS=* \
  --set-secrets DATABASE_URL=image-search-db:latest
```

La primera vez preguntará si quiere crear un repositorio en Artifact Registry: responde **Y**. El build tarda unos 10 minutos (instala PyTorch y pre-descarga el modelo). Al terminar imprime la URL del servicio:

```
Service URL: https://image-search-xxxxxxxxxx.southamerica-east1.run.app
```

**Por qué cada opción** (útil también para explicarlo en una entrevista):

| Opción | Motivo |
|--------|--------|
| `--memory 4Gi` | El pico medido al deserializar los pesos es 3,2 GB. Con 2 GiB el contenedor muere |
| `--cpu 2` | El modelo carga en 55 s, cómodamente dentro de los 240 s que da Cloud Run para empezar a escuchar |
| `--cpu-boost` | CPU adicional durante el arranque, para reducir el arranque en frío |
| `--concurrency 4` | Cada petición reserva memoria para tensores. El valor por defecto (80) agotaría la RAM |
| `--min-instances 0` | Escala a cero: sin tráfico no hay contenedor ni consumo |
| `--max-instances 2` | Techo de gasto: como mucho dos contenedores a la vez |
| `--timeout 120` | Las búsquedas tardan ~1 s; 120 s es margen de sobra |
| `--allow-unauthenticated` | Es una API pública que llama tu frontend |
| `--region southamerica-east1` | São Paulo: misma región que tu base de datos en Neon, así las consultas no cruzan continentes |

### 3.6 · Verificar

```bash
SERVICE_URL=$(gcloud run services describe image-search --region southamerica-east1 --format='value(status.url)')
curl "$SERVICE_URL/health"
```

Debe responder:

```json
{"status":"ok","version":"2.0.0","backend":"pgvector","model_loaded":true,"index_loaded":true,"index_size":6513}
```

Prueba también una búsqueda real:

```bash
curl -X POST "$SERVICE_URL/search/text" \
  -H "Content-Type: application/json" \
  -d '{"query":"un gato durmiendo","top_k":3}'
```

Cada `image_url` de la respuesta debe apuntar a `images.cocodataset.org`. Y `$SERVICE_URL/docs` abre la documentación interactiva.

Los logs están en **Cloud Run → image-search → Logs**, o con `gcloud run services logs read image-search --region southamerica-east1`.

### 3.7 · Cuánto cuesta realmente

El free tier mensual de Cloud Run es de **180 000 vCPU-segundos, 360 000 GiB-segundos y 2 millones de peticiones**. Con la configuración de arriba (2 vCPU y 4 GiB):

| Recurso | Free tier | Equivale a |
|---------|-----------|------------|
| CPU | 180 000 vCPU-s ÷ 2 vCPU | 25 h de contenedor activo al mes |
| Memoria | 360 000 GiB-s ÷ 4 GiB | 25 h de contenedor activo al mes |

Con la facturación por petición (la de por defecto), **solo se cobra mientras el contenedor atiende peticiones, más el arranque**. Un contenedor inactivo no consume. Un arranque en frío cuesta unos 60 s, es decir ~120 vCPU-s y ~240 GiB-s: caben más de mil arranques en frío al mes dentro del free tier.

**El único gasto que puede pasar de cero** es el almacenamiento de la imagen Docker en Artifact Registry: el free tier son 0,5 GB y nuestra imagen es mayor por culpa de PyTorch y el modelo. Son céntimos al mes. Para mantenerlo al mínimo, borra las versiones antiguas cuando redespliegues:

```bash
gcloud artifacts docker images list \
  southamerica-east1-docker.pkg.dev/$(gcloud config get-value project)/cloud-run-source-deploy \
  --include-tags
# y elimina las que ya no uses:
gcloud artifacts docker images delete IMAGEN@sha256:DIGEST --delete-tags
```

Confirma los precios vigentes en [cloud.google.com/run/pricing](https://cloud.google.com/run/pricing) y [artifact-registry/pricing](https://cloud.google.com/artifact-registry/pricing); la alerta de presupuesto del paso 3.1 es tu aviso si algo cambia.

### 3.8 · Errores comunes

| Síntoma | Causa | Solución |
|---------|-------|----------|
| `The user-provided container failed to start and listen on the port` | Memoria insuficiente: el contenedor murió cargando el modelo | Comprueba que pusiste `--memory 4Gi`. Mira los logs para ver si hubo OOM |
| Arranque agotado a los 240 s | Poca CPU | Añade `--cpu 4` (con 4 GiB está permitido) y `--cpu-boost` |
| `PERMISSION_DENIED` al leer el secreto | Falta el permiso de la cuenta de servicio | Repite el `add-iam-policy-binding` del paso 3.4 |
| `index_loaded: false` | `DATABASE_URL` no llega al contenedor | `gcloud run services describe image-search --region ... ` y revisa que el secreto esté montado |
| El build falla al subir el contexto | Se están subiendo las imágenes de `data/` | Verifica que existe `.gcloudignore` en la raíz (ya está en el repo) |
| Primera búsqueda tarda 3 s | Arranque en frío + Neon despertando | Normal; las siguientes bajan a ~1 s |

---

## 4 · Frontend en Render

### 4.1 Crear el static site

1. [dashboard.render.com](https://dashboard.render.com) → **New +** → **Blueprint**.
2. Conecta GitHub y elige `Project_imagen_search_system`.
3. **Branch:** `modernizacion` (cámbiala a `main` tras el paso 7). Render lee `render.yaml`.
4. Pedirá `VITE_API_URL`: la URL de Cloud Run **sin barra final**, p. ej. `https://image-search-xxxxxxxxxx.southamerica-east1.run.app`.
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
| La API responde pero las tarjetas salen vacías | El navegador no llega al CDN de COCO | Prueba en otra red; ver el [anexo](#anexo--si-algún-día-quieres-alojar-tú-las-imágenes) |

---

## 5 · Cerrar CORS y prueba de punta a punta

Con el dominio real de Render ya conocido, restringe el acceso a la API:

```bash
gcloud run services update image-search \
  --region southamerica-east1 \
  --update-env-vars CORS_ORIGINS=https://image-search-frontend.onrender.com
```

Para seguir probando también en local, separa por coma **sin espacios** y escapa la coma con `^,^` (gcloud usa la coma como separador de variables):

```bash
gcloud run services update image-search \
  --region southamerica-east1 \
  --update-env-vars "^|^CORS_ORIGINS=https://image-search-frontend.onrender.com,http://localhost:5173"
```

Cloud Run despliega una revisión nueva en segundos. Recarga el frontend y repite las búsquedas. Con F12 → **Network**, las peticiones a `/search/text` deben responder `200`.

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
> **Demo en vivo:** https://image-search-frontend.onrender.com · **API (Swagger):** https://image-search-xxxxxxxxxx.southamerica-east1.run.app/docs
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
[![Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-desplegado-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![Neon](https://img.shields.io/badge/Neon-Postgres-00E599?logo=postgresql&logoColor=white)](https://neon.tech)
```

### 6.5 Subir

```bash
git add README.md docs/
git commit -m "README: enlaces del demo, capturas y GIF"
git push
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
- Cloud Run: para redesplegar, `git pull` en Cloud Shell y repite el `gcloud run deploy` del paso 3.5.
- En `.github/workflows/ci.yml` puedes quitar `modernizacion` de la lista de ramas.
- GitHub → engranaje junto a **About** → pega la URL del demo en **Website** y añade *topics*: `clip`, `pgvector`, `fastapi`, `vector-search`, `multimodal`, `image-search`, `neon`, `cloud-run`.

---

## 8 · Mantenimiento y extras

### Comportamiento de los planes gratuitos

| Servicio | Comportamiento | Qué hacer |
|----------|----------------|-----------|
| **Neon** | Suspende el cómputo tras unos minutos sin consultas y **despierta solo** en menos de un segundo | Nada |
| **Cloud Run** | **Escala a cero** sin tráfico. El primer acceso arranca un contenedor y tarda ~60 s | Nada; el frontend lo indica |
| **Render static** | Sin límites relevantes | Nada |

Ninguno de los tres exige mantenimiento periódico. Si quieres que el demo responda al instante durante una entrevista, ábrelo 2 minutos antes para que el contenedor esté caliente.

<details>
<summary>Si prefieres eliminar el arranque en frío (tiene coste)</summary>

`--min-instances 1` mantiene un contenedor siempre encendido. Deja de escalar a cero, así que se factura de forma continua y **supera el free tier**: 4 GiB durante un mes son unos 10 000 000 GiB-s frente a los 360 000 gratuitos. Para un portafolio no compensa.

</details>

### Si cambias el dataset o el modelo

```bash
python scripts/download_coco.py ...     # solo si cambia el dataset
python scripts/build_index.py           # reconstruye el índice (TRUNCATE + reinserción)
```

Si cambias `MODEL_NAME` / `PRETRAINED`, actualiza también las variables del servicio y vuelve a desplegar, para que el `Dockerfile` pre-descargue el nuevo modelo (`ARG MODEL_NAME` y `ARG PRETRAINED` en el [Dockerfile](../Dockerfile)). Comprueba de nuevo la memoria: un modelo mayor puede necesitar más de 4 GiB.

### Seguridad de las credenciales

- `.env` está en `.gitignore` y nunca se sube. Verifícalo con `git check-ignore -v .env`.
- En Cloud Run la contraseña vive en Secret Manager, no en una variable de entorno visible.
- En Neon puedes rotarla cuando quieras: **Dashboard → Roles → Reset password**. Hazlo si la cadena se te escapó a un chat, una captura o un log. Después actualiza `.env` y crea una versión nueva del secreto:
  ```bash
  gcloud secrets versions add image-search-db --data-file=-
  gcloud run services update image-search --region southamerica-east1 \
    --set-secrets DATABASE_URL=image-search-db:latest
  ```

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

Solo tiene sentido si el CDN de COCO te falla o si cambias a un dataset propio. Opciones, de más a menos recomendable para este proyecto:

| Opción | Coste | Notas |
|--------|-------|-------|
| **Hugging Face Dataset repo** | Gratis | Los repos de datasets siguen siendo gratuitos pese al cambio en Spaces. Sirve archivos en `https://huggingface.co/datasets/<usuario>/<nombre>/resolve/main/<archivo>` |
| **Cloudflare R2** | Gratis hasta 10 GB, **sin coste de egreso** | La mejor opción de tipo S3 para servir imágenes |
| **Google Cloud Storage** | 5 GB gratis, **pero el egreso se factura** | Cómodo por estar junto a Cloud Run, aunque el tráfico de salida puede sorprender |
| **AWS S3** | Almacenamiento barato, **egreso de pago** | La menos indicada para un demo público |

El cambio en el código sería mínimo: `download_coco.py` escribiría en `metadata.json` la URL de tu almacenamiento en lugar de la de COCO, y se reindexa. El resto del sistema ya trabaja con la columna `image_url`. Si llegas a necesitarlo, dilo y lo implemento.
