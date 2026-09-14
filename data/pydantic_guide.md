# Guía de Pydantic v2

## ¿Qué es Pydantic?

Pydantic es una biblioteca de Python para validación de datos y gestión de configuraciones basada en type hints. En su versión 2, está construida sobre un core en Rust que ofrece un rendimiento hasta 50 veces superior a la versión anterior. Es el estándar de facto para definir contratos de datos en aplicaciones Python modernas.

## BaseModel: el bloque fundamental

Toda validación en Pydantic comienza definiendo una clase que hereda de `BaseModel`. Los atributos de la clase se declaran con anotaciones de tipo, y Pydantic se encarga de validar, convertir y serializar los datos automáticamente.

```python
from pydantic import BaseModel, Field

class Usuario(BaseModel):
    nombre: str = Field(..., min_length=1, description="Nombre completo del usuario")
    edad: int = Field(..., gt=0, le=150, description="Edad en años")
    email: str = Field(..., description="Dirección de correo electrónico")
```

El operador `...` (Ellipsis) indica que el campo es obligatorio. Si no se proporciona, Pydantic lanza un `ValidationError` detallado.

## Field: validaciones y metadatos

`Field` permite definir restricciones y documentación para cada campo:

- **min_length / max_length**: Longitud mínima y máxima para strings y listas.
- **gt / ge / lt / le**: Restricciones numéricas (mayor que, mayor o igual, menor que, menor o igual).
- **description**: Texto descriptivo usado por LLMs con `with_structured_output()` para entender qué se espera en cada campo.
- **default**: Valor por defecto si no se proporciona el campo.

## Serialización y deserialización

Pydantic v2 ofrece métodos eficientes para convertir entre objetos Python y formatos de intercambio:

- `model.model_dump()`: Convierte la instancia a un diccionario Python.
- `model.model_dump_json()`: Serializa directamente a una cadena JSON.
- `Model.model_validate(data)`: Crea una instancia validada desde un diccionario.
- `Model.model_validate_json(json_str)`: Crea una instancia validada desde una cadena JSON.

## Integración con LangChain: with_structured_output

La integración más potente de Pydantic con LangChain es el método `with_structured_output()`. Este método recibe una clase Pydantic como esquema y configura el LLM para que su respuesta se ajuste exactamente a esa estructura.

```python
structured_llm = model.with_structured_output(MiEsquema)
resultado = await structured_llm.ainvoke(prompt)
# resultado es una instancia validada de MiEsquema
```

Internamente, LangChain convierte el esquema Pydantic a un JSON Schema que el LLM utiliza como guía de formato, garantizando que la salida sea parseable y validada automáticamente.

## PydanticOutputParser

Para casos donde `with_structured_output()` no está disponible o se necesita mayor control, LangChain proporciona `PydanticOutputParser`. Este parser genera instrucciones de formato que se inyectan en el prompt y luego parsea la respuesta del LLM al esquema Pydantic definido.

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MiEsquema)
format_instructions = parser.get_format_instructions()
```

Las instrucciones de formato le indican al LLM exactamente qué estructura JSON debe generar, incluyendo los tipos de cada campo y sus descripciones.
