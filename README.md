# Serial-to-Parallel Translator

Traductor automático de código C/C++ **serial** a versiones **paralelas** (OpenMP, Kokkos, MPI, Go o Google Highway) mediante modelos de lenguaje accedidos vía [OpenRouter](https://openrouter.ai/). Compila, corrige errores de forma iterativa y verifica que la salida coincida con el programa original.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Dependencias](https://img.shields.io/badge/dependencias-requests%20%7C%20pydantic-green)
![Estado](https://img.shields.io/badge/estado-activo-brightgreen)

---

## 🖥️ Demo

No hay capturas en el repositorio; el flujo principal es por CLI. Ejemplo de traducción a OpenMP:

```bash
cd openmp_translator
python main.py ../OpenMP_intro_tutorial/hello.c -b OpenMP -m qwen3-coder
```

Salida esperada (resumida):

```text
Traduciendo hello.c -> hello_openmp.c
  Backend:    OpenMP
  Modelo:     qwen/qwen3-coder:free

--------------------------------------------------
  Iteración 1 — traducción inicial
--------------------------------------------------
HTTP 200
  Fichero actualizado: .../hello_openmp.c

  Compilando (OpenMP): gcc -O2 -Wall -fopenmp -lm hello_openmp.c -o hello_openmp
  Resultado: compilación OK

Compilado correctamente en la iteración 1.

==================================================
  Verificación: serial vs OpenMP
==================================================
  Salida idéntica (stdout y stderr)
  Speedup: 2.35x
```

---

## ✨ Características principales

- Traduce ficheros `.c` / `.cpp` serial a **OpenMP**, **Kokkos**, **MPI**, **Go** o **Google Highway** con un solo comando
- Selección de **modelo LLM** por alias corto o ID completo de OpenRouter
- **Corrección iterativa** automática ante errores de compilación (hasta 5 intentos, conversación multi-turno)
- **Limpieza de respuestas** del LLM (fences markdown, preámbulos, cierres residuales)
- **Compilación y ejecución** integradas con el toolchain de cada backend
- **Verificación funcional**: compara checksum numérico o stdout/stderr del binario serial frente al traducido
- **Benchmark** con media de 3 ejecuciones y cálculo de speedup (tiempo de pared y tiempo reportado por el programa)
- **Backends extensibles** mediante carpetas en `prompts/` (descubrimiento automático, sin tocar código)
- **Logging estructurado**: nivel `INFO` por defecto, `DEBUG` con `--debug`

---

## 🛠️ Tecnologías

| Capa | Tecnología |
|------|------------|
| Lenguaje | Python 3.10+ |
| API LLM | OpenRouter (`requests`, `pydantic`) |
| Compilación serial / OpenMP | `gcc` / `g++` con `-fopenmp` |
| Compilación MPI | `mpicc` / `mpicxx` |
| Compilación Kokkos | `g++` + flags desde `.env` o `kokkos_config` |
| Compilación Go | `go build` |
| Compilación Google Highway | `g++` con `-lhwy` |
| Ejecución MPI | `mpirun` |

**Dependencias de producción** (`requirements.txt`):

- `requests>=2.31.0`
- `pydantic>=2`

**Dependencias de desarrollo**: no hay `requirements-dev.txt`; basta con el intérprete Python y las herramientas de compilación del sistema.

---

## 📋 Requisitos previos

### Runtime

- **Python** 3.10 o superior
- **pip** o **uv** para instalar dependencias

### Herramientas del sistema (según backend)

| Backend | Herramientas necesarias |
|---------|-------------------------|
| OpenMP | `gcc` o `g++` con soporte OpenMP |
| MPI | `mpicc` / `mpicxx` y `mpirun` (Open MPI, MPICH, etc.) |
| Kokkos | `g++` (C++17+), Kokkos instalado y configurado |
| Go | Toolchain Go (`go build`) |
| Google Highway | `g++` con librería `libhwy` instalada |

### Cuenta externa

- Cuenta en [OpenRouter](https://openrouter.ai/) con API key válida

---

## 📦 Instalación

1. Clona o copia el directorio del proyecto:

```bash
cd openmp_translator
```

2. Crea y activa un entorno virtual (recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instala las dependencias Python:

```bash
pip install -r requirements.txt
```

   Alternativa con [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -r requirements.txt
```

4. Copia el fichero de configuración de ejemplo:

```bash
cp .env.example .env
```

5. Edita `.env` y añade tu `OPENROUTER_API_KEY`.

---

## ⚙️ Configuración

Variables de entorno cargadas desde `openmp_translator/.env`:

| Variable | Obligatoria | Descripción |
|----------|-------------|-------------|
| `OPENROUTER_API_KEY` | Sí | Clave de API de OpenRouter |
| `MPI_PROCS` | No | Procesos MPI al ejecutar (por defecto: `4`) |
| `KOKKOS_CXX` | No* | Compilador C++ para Kokkos (por defecto: `g++`) |
| `KOKKOS_CXXFLAGS` | No* | Flags de compilación Kokkos |
| `KOKKOS_LDFLAGS` | No* | Flags de enlace Kokkos |

\* Necesarias si compilas con backend Kokkos y no tienes `kokkos_config` en el `PATH`.

Ejemplo de `.env.example`:

```bash
OPENROUTER_API_KEY=sk-or-v1-tu-clave-aqui
MPI_PROCS=4
KOKKOS_CXX=g++
KOKKOS_CXXFLAGS=-std=c++20 -I/usr/local/include -DKokkos_ENABLE_OPENMP
KOKKOS_LDFLAGS=-L/usr/local/lib -lkokkosalgorithms -lkokkoscontainers -lkokkoscore -lkokkossimd -fopenmp -lm
```

Constantes configurables en código (`translator/config/settings.py`):

- `MAX_RETRIES = 5` — iteraciones máximas de corrección
- `BENCHMARK_RUNS = 3` — repeticiones para medir tiempos
- `EXECUTION_TIMEOUT_SECONDS = 60` — timeout por ejecución
- `DEFAULT_MODEL = "gpt-oss-120b"`
- `DEFAULT_BACKEND_NAME = "OpenMP"`

---

## 🚀 Uso

### Comandos básicos

```bash
# Traducción por defecto (OpenMP + gpt-oss-120b)
python main.py ruta/al/fichero.c

# Elegir backend y modelo
python main.py ruta/al/fichero.c -b Kokkos -m deepseek-r1
python main.py ruta/al/fichero.c -b MPI -m qwen3-coder
python main.py ruta/al/fichero.c -b Go -m kimi-k2.6
python main.py ruta/al/fichero.c -b GHighway -m qwen3-coder
```

### Listar opciones disponibles

```bash
python main.py --list-backends
python main.py --list-models
```

### Reasoning extendido (modelos compatibles)

```bash
python main.py fichero.c -m deepseek-r1 --thinking
python main.py fichero.c -m deepseek-r1 --thinking --thinking-effort high
```

Niveles de esfuerzo: `minimal`, `low`, `medium`, `high`, `xhigh`.

### Flags extra de compilación

Para benchmarks que requieren ficheros auxiliares (p. ej. PolyBench):

```bash
python main.py fichero.c -b OpenMP --flags "-DPOLYBENCH_TIME utilities/polybench.c"
```

### Referencia de flags CLI

| Flag | Descripción |
|------|-------------|
| `source` | Fichero `.c` o `.cpp` a traducir |
| `-b`, `--backend` | Backend: `OpenMP`, `Kokkos`, `MPI`, `Go`, `GHighway` (por defecto: OpenMP) |
| `-m`, `--model` | Alias de modelo o ID completo de OpenRouter |
| `--list-backends` | Muestra backends detectados y sale |
| `--list-models` | Muestra alias de modelos y sale |
| `--thinking` | Activa reasoning extendido en la API |
| `--no-thinking` | Desactiva reasoning (por defecto) |
| `--thinking-effort` | Nivel de esfuerzo del reasoning (requiere `--thinking`) |
| `--flags` | Flags extra de compilación, como cadena entre comillas |
| `--debug` | Muestra la respuesta completa de la API (activa logging DEBUG) |

### Ficheros de salida

El traducido se escribe junto al original con el sufijo del backend:

| Entrada | Backend | Salida |
|---------|---------|--------|
| `hello.c` | OpenMP | `hello_openmp.c` |
| `hello.c` | MPI | `hello_mpi.c` |
| `hello.c` | Kokkos | `hello_kokkos.cpp` |
| `hello.c` | Go | `hello_go.go` |
| `hello.c` | GHighway | `hello_ghighway.cpp` |
| `pi.cpp` | Kokkos | `pi_kokkos.cpp` |

### Flujo interno

1. Envía el código serial + prompt de traducción al LLM
2. Extrae y limpia el código de la respuesta
3. Compila el fichero traducido
4. Si falla, reenvía errores de compilación al LLM (historial multi-turno)
5. Tras compilar con éxito, ejecuta serial vs traducido y reporta speedup

---

## 📁 Estructura del proyecto

```text
openmp_translator/
├── main.py                          # Punto de entrada CLI
├── requirements.txt                 # Dependencias Python
├── .env.example                     # Plantilla de variables de entorno
├── .gitignore
├── prompts/
│   ├── OpenMP/                      # translate.txt + fix_errors.txt
│   ├── Kokkos/
│   ├── MPI/
│   ├── Go/
│   ├── GHighway/
│   └── _drafts/                     # Prompts experimentales (no usados)
└── translator/
    ├── exceptions.py                # Jerarquía de excepciones del dominio
    ├── registry.py                  # Descubrimiento y resolución de backends
    ├── cli/
    │   └── main.py                  # Argumentos, logging, orquestación
    ├── config/
    │   ├── settings.py              # Modelos, constantes, rutas
    │   └── env.py                   # Carga de .env
    ├── domain/
    │   └── backend.py               # Dataclass Backend (sin dependencias externas)
    ├── services/
    │   └── translate.py             # Pipeline traducción + corrección iterativa
    └── infrastructure/
        ├── llm/
        │   ├── client.py            # Cliente HTTP + retry (OpenRouter)
        │   ├── code_extractor.py    # Limpieza y extracción de código de respuestas LLM
        │   └── models.py            # Modelos Pydantic de la API
        ├── compilation/
        │   ├── common.py            # Utilidades compartidas (CompilationResult)
        │   ├── openmp.py            # gcc/g++ + OpenMP
        │   ├── mpi.py               # mpicc/mpicxx
        │   ├── kokkos.py            # g++ + Kokkos
        │   ├── go.py                # go build
        │   └── ghighway.py          # g++ + libhwy
        └── execution/
            ├── process.py           # Ejecución de procesos con timeout
            ├── output_parser.py     # Extracción de checksum y tiempo del stdout
            └── benchmark.py        # Verificación de correctitud y cálculo de speedup
```

---

## 📜 Scripts disponibles

El proyecto no usa `package.json` ni `Makefile`. Los puntos de entrada son:

| Comando | Descripción |
|---------|-------------|
| `python main.py <fichero> [opciones]` | Ejecuta el traductor con la CLI completa |
| `python main.py --list-backends` | Lista backends detectados en `prompts/` |
| `python main.py --list-models` | Lista alias de modelos configurados |
| `uv run main.py <fichero> [opciones]` | Alternativa si usas uv sin activar el venv |

---

## 🤝 Contribución

1. Haz fork del repositorio
2. Crea una rama con tu cambio: `git checkout -b feature/mi-mejora`
3. Realiza commits descriptivos (se recomienda [Conventional Commits](https://www.conventionalcommits.org/)):
   - `feat:` nueva funcionalidad
   - `fix:` corrección de bug
   - `docs:` documentación
   - `refactor:` cambio interno sin alterar comportamiento
4. Abre un Pull Request describiendo el cambio y cómo probarlo

Para añadir un nuevo backend:

1. Crea una carpeta `prompts/<Nombre>/` con `translate.txt` y `fix_errors.txt` — el backend se descubre automáticamente.
2. Si el toolchain es distinto, implementa una función `compile_<nombre>(source_path, ...)` en `translator/infrastructure/compilation/<nombre>.py` y regístrala en `compilation/__init__.py`.
3. Si el backend necesita prefijo de ejecución o variables de entorno especiales, añádelo en `translator/registry.py` (`_EXTENSION_BY_SLUG`, `_RUN_ENV_BY_SLUG`) o en la lógica de `_build_backend()`.

---

## 📄 Licencia

No se ha detectado un fichero `LICENSE` en el repositorio. Consulta con los mantenedores del proyecto antes de redistribuir o usar el código en entornos con restricciones de licencia.
