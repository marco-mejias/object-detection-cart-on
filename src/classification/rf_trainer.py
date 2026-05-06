"""
Entrenamiento de un clasificador Random Forest para categorías de producto.

Random Forest es la alternativa que pide vuestro proposal junto al SVM.
Es un conjunto de árboles de decisión que votan: cada árbol se entrena
con un subconjunto distinto de los datos y de las features, y al final
se combina por mayoría. Ventajas frente al SVM:

- No necesita estandarizar features (los árboles no se ven afectados por
  la escala). Aún así lo dejamos en un Pipeline por consistencia.
- Es más interpretable (puedes ver qué features son importantes).
- Suele ir bien con clases desbalanceadas si usas class_weight='balanced'.
- Maneja sin problema dimensiones altas (1806 en nuestro caso).

Hiperparámetros importantes:
- n_estimators: cuántos árboles. Más = mejor pero más lento. 200 es buen default.
- max_depth: profundidad máxima de cada árbol. None = sin límite (overfitting).
- min_samples_leaf: mínimo de muestras por hoja. Subirlo regulariza.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_rf_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Construye un Pipeline con StandardScaler + Random Forest.

    Parameters
    ----------
    n_estimators : int
        Número de árboles del bosque. Más árboles = más estable, más lento.
    max_depth : int o None
        Profundidad máxima de cada árbol. None = ilimitada.
    min_samples_leaf : int
        Mínimo de muestras en cada hoja. 2 regulariza un poco sin perder señal.
    class_weight : str
        'balanced' compensa clases desbalanceadas.
    random_state : int
        Semilla para reproducibilidad.
    n_jobs : int
        Núcleos a usar. -1 = todos.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])
