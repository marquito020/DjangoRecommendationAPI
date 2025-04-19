# Sistema de Recomendación de Productos - API Django

Este proyecto implementa un sistema de recomendación de productos basado en Django REST Framework que recibe un conjunto de IDs de productos como entrada y devuelve recomendaciones generadas por un modelo de machine learning.

## Estructura del Dataset

El sistema utiliza un dataset en formato CSV con la siguiente estructura:

| Campo | Tipo   | Descripción                                          |
|-------|--------|------------------------------------------------------|
| input | string | Lista de IDs de productos usados como entrada (m=2)  |
| target| string | Lista de IDs de productos a predecir (longitud ≥ 1)  |

Ejemplo:
```csv
input,target
"[1001,1003]","[1005]"
"[1001,1005]","[1003,1007]"
```

Para una explicación detallada sobre cómo se construye este CSV a partir de datos de pedidos y cómo funciona la estructura input-target, consulte la [documentación detallada del CSV](csv_structure.md).

## Instalación

1. Clonar el repositorio
2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Aplicar migraciones:
   ```
   python manage.py migrate
   ```
4. Iniciar el servidor:
   ```
   python manage.py runserver
   ```

## Documentación API con Swagger

El proyecto incluye documentación automática con Swagger:

- **Swagger UI**: `/swagger/` - Interfaz interactiva para probar la API
- **ReDoc**: `/redoc/` - Documentación alternativa más detallada
- **JSON Schema**: `/swagger.json` - Documentación en formato JSON

Al acceder a la raíz del proyecto (`/`), serás redirigido automáticamente a la documentación de Swagger.

## Uso de la API

### Entrenar el modelo

```
GET /api/train/
```

Respuesta:
```json
{
  "message": "Modelo entrenado correctamente",
  "products": [1001, 1002, 1003, 1005, 1007]
}
```

### Obtener recomendaciones

```
POST /api/recommendations/
```

Cuerpo de la solicitud:
```json
{
  "input": [1001, 1003]
}
```

Respuesta:
```json
{
  "input": [1001, 1003],
  "suggested": [1005, 1007]
}
```

## Funcionamiento interno

1. El sistema carga los datos desde el archivo CSV
2. Convierte las cadenas de texto a listas de enteros
3. Utiliza MultiLabelBinarizer para vectorizar los datos
4. Entrena un modelo MultiOutputClassifier con RandomForestClassifier
5. Al predecir, vectoriza la entrada y genera recomendaciones
6. Filtra los resultados por umbral o selecciona el top-1

## Ejecución de pruebas

```
python manage.py test recommender
```

## Cliente de prueba

Se incluye un script de cliente de línea de comandos (`app.py`) para probar la API:

```bash
# Entrenar el modelo
python app.py --train

# Obtener recomendaciones
python app.py --recommend 1001 1003
``` 