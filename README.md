 Aquí tienes el README completo basado en tu notebook de fine-tuning:

---

# 🚀 Fine-Tuning de Qwen2.5-Coder-14B para Generación de Epics de Software

Este proyecto implementa un pipeline completo de fine-tuning supervisado (SFT) utilizando **Unsloth** para adaptar el modelo **Qwen2.5-Coder-14B-Instruct** a la tarea específica de generación de Epics de software en formato JSON, actuando como un Product Manager Senior experto.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación y Configuración](#instalación-y-configuración)
- [Proceso de Fine-Tuning](#proceso-de-fine-tuning)
- [Exportación a GGUF](#exportación-a-gguf)
- [Detalle de Dependencias](#detalle-de-dependencias)
- [Estructura del Dataset](#estructura-del-dataset)
- [Optimizaciones de Memoria](#optimizaciones-de-memoria)
- [Resultados Esperados](#resultados-esperados)

---

## 🎯 Descripción General

Este proyecto entrena un modelo de lenguaje grande (LLM) para que actúe como un **Product Manager Senior** especializado en redactar Epics de software detalladas, estructuradas y precisas en formato JSON. El modelo analiza contexto y requerimientos proporcionados para generar salidas estructuradas con campos como `epic_id`, `title`, `acceptance_criteria`, etc.

### Características Principales

- **Modelo Base**: Qwen2.5-Coder-14B-Instruct (14 mil millones de parámetros)
- **Técnica**: LoRA (Low-Rank Adaptation) + Cuantización 4-bit
- **Framework**: Unsloth (optimizado para 2x velocidad y 70% menos memoria)
- **Formato de Salida**: GGUF (para inferencia local con llama.cpp)
- **Dataset**: 104 ejemplos conversacionales en formato JSONL

---

## 🏗️ Arquitectura del Proyecto

```
proyecto/
├── data/
│   └── epics.jsonl              # Dataset de entrenamiento (104 ejemplos)
├── outputs/                     # Checkpoints del entrenamiento
├── MiPM_Senior/                 # Modelo exportado en formato GGUF
├── notebook.ipynb              # Jupyter notebook con el pipeline completo
├── pyproject.toml              # Dependencias gestionadas por Poetry
└── README.md                   # Este archivo
```

---

## 💻 Requisitos del Sistema

### Hardware Recomendado

| Componente | Especificación |
|------------|---------------|
| **GPU** | NVIDIA RTX 3090/4090 o superior (24GB VRAM) |
| **RAM** | 32GB+ |
| **Almacenamiento** | 50GB libres |
| **Sistema Operativo** | Linux (Ubuntu 22.04 recomendado) |

### Notas de Compatibilidad

- **RTX 4090 (24GB)**: Entrenamiento fluido con batch_size=1 y gradient_accumulation_steps=8
- **RTX 3090 (24GB)**: Configuración similar, posiblemente requiere ajustes menores
- **GPUs con menos VRAM**: Se recomienda reducir `max_seq_length` a 2048 o usar `r=8` en LoRA

---

## 🔧 Instalación y Configuración

### 1. Instalación de Dependencias del Sistema

```bash
# Actualizar repositorios e instalar herramientas de compilación
apt-get update -y && apt-get install -y \
    cmake \
    build-essential \
    libcurl4-openssl-dev
```

**Propósito**:
- `cmake`: Gestión del proceso de compilación para librerías nativas
- `build-essential`: Compiladores GCC/G++ y herramientas básicas
- `libcurl4-openssl-dev`: Transferencias de red seguras para descarga de modelos

### 2. Configuración del Entorno Python con Poetry

```bash
# Instalar dependencias principales
poetry add unsloth peft accelerate bitsandbytes xformers trl unsloth-zoo

# Instalar dependencias para exportación GGUF
poetry add gguf protobuf sentencepiece

# Instalar ipykernel para Jupyter
poetry add ipykernel

# Registrar el kernel en Jupyter
poetry run python -m ipykernel install --user --name=per-training-model --display-name "Python (Poetry)"
```

### 3. Verificación de la Instalación

```python
import torch
import unsloth
from unsloth import FastLanguageModel

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión Unsloth: {unsloth.__version__}")
```

---

## 🧠 Proceso de Fine-Tuning

### Fase 1: Carga del Modelo Base

```python
from unsloth import FastLanguageModel
import torch

# Configuración de hiperparámetros
max_seq_length = 4096  # Contexto extendido para documentación técnica
dtype = None          # Auto-detección (float16/bfloat16)
load_in_4bit = True   # Cuantización 4-bit esencial para GPU de consumo

# Carga optimizada por Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

**Detalles técnicos**:
- **Cuantización 4-bit**: Reduce el modelo de ~28GB a ~8GB usando bitsandbytes
- **bnb-4bit**: Versión pre-cuantizada optimizada para inferencia rápida
- **Auto-detección de dtype**: Usa bfloat16 en GPUs Ampere/Ada (RTX 30xx/40xx) para mejor estabilidad

### Fase 2: Configuración de Adaptadores LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # Rank: tamaño de matrices de bajo rango
    target_modules=[         # Capas a adaptar
        "q_proj", "k_proj", "v_proj", "o_proj",  # Atención
        "gate_proj", "up_proj", "down_proj"      # Redes feed-forward
    ],
    lora_alpha=16,           # Escala de aprendizaje (generalmente = r o 2*r)
    lora_dropout=0,          # Sin dropout para máxima velocidad en Unsloth
    bias="none",             # No entrenar sesgos
    use_gradient_checkpointing="unsloth",  # Optimización de memoria 30% más eficiente
    random_state=3407,       # Reproducibilidad
)
```

**Parámetros LoRA explicados**:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `r` | 16 | Dimensiones de las matrices de adaptación. Mayor = más capacidad, más memoria |
| `lora_alpha` | 16 | Factor de escala del aprendizaje. `alpha/r` = tasa efectiva |
| `target_modules` | 7 capas | Qué partes del transformer se adaptan. Incluir MLP mejora capacidad |
| `lora_dropout` | 0 | Desactivado en Unsloth para velocidad; el modelo base ya tiene regularización |

**Memoria entrenable**: ~12.5M parámetros de 14.7B total (**0.09%** del modelo)

### Fase 3: Preparación del Dataset

#### Formato de Datos (JSONL)

Cada línea en `epics.jsonl` contiene:

```json
{
  "input": {
    "context": "Descripción del proyecto...",
    "requerimientos": ["Req 1", "Req 2", ...]
  },
  "output": {
    "epic_id": "EPIC-001",
    "title": "Título de la Epic",
    "acceptance_criteria": ["Criterio 1", ...]
  }
}
```

#### Procesamiento y Tokenización

```python
from unsloth.chat_templates import get_chat_template
import json

# 1. Configurar plantilla ChatML (<|im_start|>, <|im_end|>)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

# 2. Definir identidad del sistema
system_prompt = """Eres un Product Manager Senior experto. Tu tarea es analizar 
el contexto y los requerimientos proporcionados para redactar Epics de software 
detalladas, estructuradas y precisas en formato JSON."""

# 3. Formatear ejemplos
formatted_texts = []
with open("../data/epics.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        
        # Convertir dicts a JSON legible con indentación
        user_content = json.dumps(record["input"], ensure_ascii=False, indent=2)
        assistant_content = json.dumps(record["output"], ensure_ascii=False, indent=2)
        
        # Estructura conversacional
        convo = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Aplicar plantilla ChatML
        text = tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False
        )
        formatted_texts.append({"text": text})

# 4. Crear dataset HuggingFace
from datasets import Dataset
dataset = Dataset.from_list(formatted_texts)
```

**Por qué ChatML**:
- Estándar de Qwen2.5 para conversaciones
- Etiquetas explícitas `<|im_start|>role` y `<|im_end|>` separan claramente turnos
- Permite que el modelo aprenda cuándo debe generar (assistant) vs escuchar (user)

### Fase 4: Entrenamiento

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Limpieza de memoria GPU
torch.cuda.empty_cache()
gc.collect()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,          # Procesos paralelos para carga de datos
    packing=False,               # No agrupar secuencias (estabilidad)
    args=TrainingArguments(
        per_device_train_batch_size=1,      # 1 ejemplo por paso
        gradient_accumulation_steps=8,      # Efectivo: batch de 8
        warmup_steps=5,                     # Calentamiento gradual
        num_train_epochs=3,                 # 3 pasadas completas
        learning_rate=2e-4,                 # Tasa estándar para LoRA
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,                    # Log cada paso
        optim="paged_adamw_8bit",           # Optimizador con paginación a RAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Iniciar entrenamiento
trainer_stats = trainer.train()
```

**Estrategias de ahorro de memoria**:

| Técnica | Implementación | Ahorro |
|---------|---------------|--------|
| Cuantización 4-bit | `load_in_4bit=True` | ~75% |
| LoRA | `r=16`, solo capas seleccionadas | ~99.9% parámetros congelados |
| Gradient Checkpointing | `"unsloth"` | ~30% extra |
| Paged Optimizer | `paged_adamw_8bit` | Usa RAM si VRAM llena |
| Batch size 1 + accum 8 | Simula batch 8 | Flexible |

**Resultado del entrenamiento**:
- **Total de pasos**: 39 (104 ejemplos × 3 epochs ÷ batch efectivo 8)
- **Tiempo estimado**: ~10-15 minutos en RTX 4090
- **Loss final**: ~1.31 (de 1.86 inicial)

---

## 📦 Exportación a GGUF

El formato GGUF (GPT-Generated Unified Format) permite ejecutar el modelo localmente con **llama.cpp**, compatible con CPUs y GPUs de consumo.

```python
# Guardar modelo fusionado y cuantizado
model.save_pretrained_gguf(
    "MiPM_Senior",           # Carpeta de salida
    tokenizer,
    quantization_method="q4_k_m"  # 4-bit con precisión media
)
```

**Proceso interno**:
1. **Fusión**: Une adaptadores LoRA al modelo base Qwen 14B
2. **Conversión**: Transforma de PyTorch (.safetensors) a formato GGUF
3. **Cuantización**: Comprime de ~28GB a ~9GB

**Métodos de cuantización disponibles**:

| Método | Tamaño | Calidad | Uso recomendado |
|--------|--------|---------|-----------------|
| `q4_k_m` | ~9GB | ⭐⭐⭐⭐ Alta | Balance ideal |
| `q5_k_m` | ~11GB | ⭐⭐⭐⭐⭐ Muy alta | Máxima calidad |
| `q4_0` | ~8GB | ⭐⭐⭐ Buena | VRAM muy limitada |

---

## 📚 Detalle de Dependencias

### Core - Entrenamiento

#### `unsloth (>=2026.2.1,<2027.0.0)`
**Propósito**: Framework de entrenamiento optimizado que proporciona:
- Parcheo automático de modelos para 2x velocidad de entrenamiento
- Implementaciones eficientes de RoPE, SwiGLU, y cross-entropy
- Integración nativa con LoRA y cuantización 4-bit
- **Funciones clave**: `FastLanguageModel.from_pretrained()`, `get_peft_model()`

#### `unsloth-zoo (>=2026.2.1,<2027.0.0)`
**Propósito**: Utilidades y parches adicionales para Unsloth:
- Funciones de utilidad para modelos (`is_bfloat16_supported()`)
- Plantillas de chat optimizadas (`get_chat_template()`)
- Patching de capas específicas de atención

#### `trl (<=0.24.0)`
**Propósito**: Transformer Reinforcement Learning - proporciona:
- `SFTTrainer`: Entrenador especializado en Fine-Tuning Supervisado
- Integración con PEFT para entrenamiento eficiente
- Manejo de datasets conversacionales

#### `peft (>=0.18.1,<0.19.0)`
**Propósito**: Parameter-Efficient Fine-Tuning:
- Implementación de LoRA (Low-Rank Adaptation)
- QLoRA para entrenamiento con cuantización
- Inyección y fusión de adaptadores

### Optimización de Memoria

#### `bitsandbytes (>=0.49.2,<0.50.0)`
**Propósito**: Cuantización y optimizaciones de 8-bit/4-bit:
- `Linear4bit`: Capas lineares cuantizadas en 4-bit
- `bnb.nn.Module`: Módulos optimizados para inferencia
- Integración con optimizadores 8-bit (`adamw_8bit`)

#### `accelerate (>=1.12.0,<2.0.0)`
**Propósito**: Abstracción de entrenamiento distribuido:
- Manejo automático de dispositivos (GPU/CPU)
- Offloading de optimizadores a disco/RAM
- Paralelización de datos y modelos

#### `xformers (>=0.0.34,<0.0.35)`
**Propósito**: Optimizaciones de atención eficiente:
- `memory_efficient_attention`: Atención con menos uso de VRAM
- Operaciones fundidas (fused operations) para velocidad
- Compatible con Flash Attention 2

### Entorno de Desarrollo

#### `ipykernel (>=7.2.0,<8.0.0)`
**Propósito**: Kernel de Jupyter para Poetry:
- Ejecución de notebooks en entornos virtuales aislados
- Registro de kernels personalizados
- Integración con JupyterLab/VS Code

#### `torchvision (>=0.25.0,<0.26.0)`
**Propósito**: Dependencia complementaria de PyTorch:
- Operaciones de visión computacional (no usado directamente, pero requerido por el ecosistema)

### Exportación e Inferencia

#### `gguf (>=0.17.1,<0.18.0)`
**Propósito**: Manejo del formato GGUF:
- Escritura de archivos `.gguf` para llama.cpp
- Cuantización y serialización de tensores
- Metadatos de modelos

#### `protobuf (>=6.33.5,<7.0.0)`
**Propósito**: Serialización de datos estructurados:
- Protocolo de comunicación para modelos serializados
- Dependencia de `sentencepiece` y `gguf`

#### `sentencepiece (>=0.2.1,<0.3.0)`
**Propósito**: Tokenización de sub-palabras:
- Algoritmo BPE (Byte-Pair Encoding) y Unigram
- Tokenizador nativo de Qwen, Llama, Mistral
- Manejo de vocabularios multilingües

---

## 🧮 Optimizaciones de Memoria Detalladas

### Configuración para RTX 4090 (24GB)

```python
# Configuración conservadora pero eficiente
max_seq_length = 4096
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
r = 16  # LoRA rank

# Uso de VRAM estimado: ~20-22GB
```

### Configuración para VRAM limitada (12-16GB)

```python
# Reducciones graduales
max_seq_length = 2048
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
r = 8  # Reducir rank de LoRA
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Sin MLP

# Uso de VRAM estimado: ~10-12GB
```

---

## 📊 Resultados Esperados

### Métricas de Entrenamiento

```
Paso | Training Loss
-----|--------------
1    | 1.8634
10   | 1.6325
20   | 1.3723
30   | 1.3414
39   | 1.3166
```

### Archivos Generados

```
MiPM_Senior/
├── MiPM_Senior-Q4_K_M.gguf    # Modelo cuantizado (~9GB)
├── tokenizer.model              # Tokenizador
└── ...                          # Metadatos y configuración
```

---

## 🚀 Uso del Modelo Exportado

### Con llama.cpp

```bash
# Ejecutar inferencia local
./main -m MiPM_Senior-Q4_K_M.gguf \
       --color \
       --ctx_size 4096 \
       -n -1 \
       -p "<|im_start|>system\nEres un Product Manager Senior experto...<|im_end|>\n<|im_start|>user\n{tu_input}<|im_end|>\n<|im_start|>assistant\n"
```

### Con Ollama

```bash
# Crear Modelfile
echo 'FROM ./MiPM_Senior-Q4_K_M.gguf' > Modelfile
ollama create pm-senior -f Modelfile
ollama run pm-senior
```

---

## 📝 Notas y Troubleshooting

### Error: `No config file found`
**Solución**: El modelo ya está en caché, simplemente carga sin verificación:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

### Error: `CUDA out of memory`
**Soluciones**:
1. Reducir `max_seq_length` a 2048
2. Usar `r=8` en lugar de `r=16`
3. Eliminar capas MLP de `target_modules`
4. Aumentar `gradient_accumulation_steps`

### Advertencia: `IProgress not found`
**Solución**: No crítica, solo afecta barras de progreso visuales:
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

