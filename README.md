# Misión 11 · DevOps, MLOps y AIOps
Proyecto didáctico: clasificador Iris, API FastAPI, GitLab CI/CD, MLflow Tracking e Isolation Forest sobre métricas operativas simuladas.

## Inicio rápido
```bash
python -m venv .venv
# Activar .venv (Windows: .venv\Scripts\activate; Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pytest -q
python src/train.py
uvicorn src.api:app --host 127.0.0.1 --port 8000
# GET http://127.0.0.1:8000/health
# POST http://127.0.0.1:8000/predict con {"features":[5.1,3.5,1.4,0.2]}
python src/anomaly.py
```
MLflow UI: `mlflow ui --backend-store-uri ./mlruns` (abrir http://127.0.0.1:5000).
Docker: `docker build -t mission11:local .` y `docker run --rm -p 8000:8000 mission11:local`.

**Nota:** este kit utiliza seguimiento local de MLflow y modelo cargado desde artefacto empaquetado. La etapa package del pipeline construye Docker, pero no publica ni despliega automáticamente a AWS. Para un CD real hay que añadir credenciales seguras y un entorno autorizado.
