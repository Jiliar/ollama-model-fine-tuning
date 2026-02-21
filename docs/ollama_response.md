Diccionario de Respuesta JSON de Ollama

| **Clave JSON**               | **Tipo de Dato**    | **Significado / Descripción**                                                                                                                                              |
| ---------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`model`**                | *String*                | El identificador del modelo que procesó la solicitud (ej.`"Product_Expert"`).                                                                                                  |
| **`created_at`**           | *String (ISO 8601)*     | Fecha y hora exacta en la que se completó la generación de la respuesta, en formato UTC.                                                                                        |
| **`response`**             | *String*                | El texto final generado por la IA (la Epic, User Story, etc.). Contiene caracteres de escape como `\n`para saltos de línea.                                                    |
| **`done`**                 | *Booleano*              | Indica si el modelo ha terminado completamente de generar la respuesta (`true`o `false`). Útil cuando se usa `"stream": true`.                                             |
| **`done_reason`**          | *String*                | La razón por la que el modelo dejó de escribir.`"stop"`significa que terminó naturalmente.`"length"`indicaría que alcanzó el límite máximo de tokens configurado.      |
| **`context`**              | *Array de Enteros*      | La representación numérica (tokens) del historial de esta conversación. Se debe enviar en la siguiente petición si quieres que el modelo "recuerde" de qué estaban hablando. |
| **`total_duration`**       | *Entero (Nanosegundos)* | El tiempo total que tardó Ollama en procesar toda la petición, desde que llegó hasta que se envió la respuesta final.                                                         |
| **`load_duration`**        | *Entero (Nanosegundos)* | El tiempo que tomó cargar los pesos del modelo desde el disco duro hacia la memoria (RAM o VRAM de la GPU).                                                                      |
| **`prompt_eval_count`**    | *Entero*                | La cantidad de**tokens de entrada** . Es decir, la longitud de tu pregunta o instrucción inicial.                                                                          |
| **`prompt_eval_duration`** | *Entero (Nanosegundos)* | El tiempo que le tomó al modelo leer, procesar y "entender" tu instrucción antes de empezar a escribir la primera palabra.                                                      |
| **`eval_count`**           | *Entero*                | La cantidad de**tokens de salida** . Es el tamaño real de la respuesta generada por el modelo.                                                                             |
| **`eval_duration`**        | *Entero (Nanosegundos)* | El tiempo dedicado exclusivamente a generar la respuesta, token por token. (Ideal para calcular la velocidad en*tokens por segundo* ).                                          |

---

> **Tip de conversión:** Como Ollama devuelve los tiempos en  **nanosegundos** , si quieres ver estos valores en **segundos** reales dentro de algún script de Python, solo tienes que dividir la cifra entre `1,000,000,000` (**$10^9$**).
