# Proyecto de Predicción de Intención de Compra
Este proyecto tiene como objetivo predecir la intención de compra de los usuarios utilizando el
dataset `shoppers_intentions` de UCI. El proyecto está estructurado en varios módulos que se
encargan de la carga, preprocesamiento, entrenamiento y evaluación del modelo.
## Contenido
- `Scorer.py`: Este archivo contiene las funciones necesarias para evaluar el rendimiento del
modelo.
- `Trainer.py`: Contiene las funciones y clases necesarias para entrenar el modelo.
- `main.py`: Script principal que orquesta la ejecución del proyecto, incluyendo la carga de datos,
preprocesamiento, entrenamiento y evaluación.
- `class_and_functions.py`: Incluye clases y funciones auxiliares que son utilizadas en otros
módulos.
## Requisitos
Para ejecutar este proyecto, necesitarás tener instalado:
- Python 3.8 o superior
- MLflow
Puedes instalar MLflow ejecutando:
```bash
pip install mlflow
```
## Uso
1. Clonar el repositorio en tu máquina local.
2. Instalar las dependencias mencionadas anteriormente.
3. Ejecutar el script principal `main.py`:
```bash
python main.py
```
## Descripción de los Archivos
### Scorer.py
Contiene las funciones necesarias para evaluar el rendimiento del modelo, incluyendo métricas
como precisión, recall y F1-score.
### Trainer.py
Define las clases y funciones necesarias para entrenar el modelo. Aquí se realiza la carga del
dataset, el preprocesamiento de los datos y el entrenamiento del modelo.
### main.py
Es el script principal que coordina todo el flujo del proyecto. Incluye la carga de datos, el
preprocesamiento, el entrenamiento y la evaluación del modelo.
### class_and_functions.py
Incluye clases y funciones auxiliares que son utilizadas en otros módulos para facilitar diversas
tareas como la manipulación de datos y la visualización de resultados.
## Dataset
El dataset `shoppers_intentions` de UCI es utilizado para este proyecto. Debes descargarlo y
colocarlo en el directorio `data/`.
## Contribución
Si deseas contribuir a este proyecto, por favor sigue los pasos a continuación:
1. Haz un fork del repositorio.
2. Crea una rama (`git checkout -b feature-nueva-funcionalidad`).
3. Realiza tus cambios y haz commit de los mismos (`git commit -am 'Añadir nueva funcionalidad'`).
4. Sube la rama (`git push origin feature-nueva-funcionalidad`).
5. Abre un Pull Request.
## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más
detalles.
