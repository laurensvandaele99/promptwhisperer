# Prompt interaction-risk API

This folder turns the **same first-prompt feature pipeline used in `prompt_efficiency_final_analysis_v4_ready.ipynb`** into a stateless FastAPI scoring service for the experiment.

## Important scientific point

The current v4 paper uses a **five-class XGBoost model** and treats `P(score = 2)` as the focal interaction-risk score. The exporter therefore defaults to `multiclass_class2`.

Because you are considering changing the primary target to **scores 1-2 vs 3-5**, the exporter also supports:

- `multiclass_1plus2`: same five-class model, API risk = `P(score=1)+P(score=2)`. Useful as the quick diagnostic.
- `binary_1plus2`: directly retrains XGBoost on 1-2 vs 3-5. **Only use this as the final experimental model after you have evaluated that binary target with the same grouped CV framework.**

The API always returns `model_version`, `mode`, and `risk_definition`, so the experiment can record exactly what was used.

## 1. Export a frozen deployment model on your own computer

Open Anaconda Prompt / PowerShell in this folder using the same Python environment in which v4 ran.

For the current v4 class-2 definition:

```powershell
python export_model.py --project-dir "C:\Users\lauvdael\OneDrive - UGent\FWO\Digital Pollution onderzoek\chatGPT_study" --mode multiclass_class2 --version jams-v4-xgb-class2
```

The script will:

1. find `strict_efficiency_50k_with_topics.csv`;
2. apply the same defensive data cleaning as v4;
3. recreate the exact handcrafted prompt features;
4. reuse the matching cached MiniLM embeddings if they exist;
5. fit the v4 XGBoost specification on the full clean dataset;
6. save `model/model_bundle.joblib`;
7. save `model/model_manifest.json`;
8. overwrite `model/requirements_model.txt` with the exact versions used for training;
9. run a small inference sanity check.

This final full-data fit is for **deployment after cross-validated evaluation**. It does not replace the OOF metrics reported in the paper.

## 2. Local API test

Install dependencies using the version pins written during export:

```powershell
pip install -r requirements.txt -r model\requirements_model.txt
```

Set a test API key:

```powershell
$env:MODEL_API_KEY="choose-a-long-random-secret"
```

Run:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/health
```

Then in another terminal:

```powershell
$env:MODEL_API_URL="http://127.0.0.1:8000/score"
$env:MODEL_API_KEY="choose-a-long-random-secret"
python test_request.py
```

Expected response shape for a five-class bundle:

```json
{
  "risk": 0.31,
  "risk_definition": "P(strict_efficiency_score = 2)",
  "model_version": "jams-v4-xgb-class2",
  "model_name": "XGBoost",
  "mode": "multiclass_class2",
  "latency_ms": 45.2,
  "probabilities": {"1": 0.05, "2": 0.31, "3": 0.11, "4": 0.20, "5": 0.33},
  "risk_class2": 0.31,
  "risk_1plus2": 0.36
}
```

The numbers above are illustrative. Your real output comes from the exported model.

## 3. Deploy with Docker

The folder includes a Dockerfile. The model bundle must exist before the image is built.

Local Docker test:

```powershell
docker build -t prompt-risk-api .
docker run --rm -p 8000:8000 -e MODEL_API_KEY="your-secret" prompt-risk-api
```

Then test `http://127.0.0.1:8000/health`.

### Render example

A `render.yaml` is included. Put this folder in a **private** Git repository, create a Render Blueprint/Web Service, and set `MODEL_API_KEY` as a secret environment variable.

After deployment, Render gives a URL such as:

```text
https://prompt-risk-model.onrender.com
```

Your Lovable/Supabase secret should then be:

```text
MODEL_API_URL=https://prompt-risk-model.onrender.com/score
```

Do **not** use `/health`; the experiment calls `/score`.

## 4. Lovable / Supabase connection

Use a Supabase Edge Function between the browser and this API. A reference implementation is included as:

`lovable_supabase_edge_function.ts`

Store these as Supabase secrets:

```text
MODEL_API_URL=https://your-host/score
MODEL_API_KEY=the-same-secret-configured-on-the-python-service
```

The participant's browser should call the Supabase function, not your Python API directly. That keeps the model API key out of client-side code.

The Python service is intentionally stateless and does **not** persist participant prompts. Your experiment database should save the pre-treatment prompt, returned score, model version, and timestamp.

## 5. What Lovable should save from every call

At minimum:

- `initial_prompt`
- `model_risk`
- `model_version`
- `risk_definition`
- `model_api_success`
- `model_api_latency_ms`
- `model_api_timestamp`

For a five-class bundle, also save `risk_class2` and `risk_1plus2`. This is useful while you are deciding which outcome becomes final.

## 6. Reproducibility checks before launch

Before collecting participants:

1. Freeze the final target definition.
2. Export the final full-data model once.
3. Do not retrain it during the experiment.
4. Run at least 50-100 known prompts through both your notebook-side inference and the API and confirm equal probabilities to numerical tolerance.
5. Record the `model_manifest.json` with your study materials.
6. Verify that risk is calculated **before randomization** and from the untouched initial prompt only.
7. Confirm that page refreshes never trigger a new treatment assignment.
8. Confirm that API failures do not block participation; log them and continue according to the experimental protocol.

## 7. Privacy

The scoring API does not write prompt text to disk. Avoid enabling request-body logging at the hosting provider. The experiment database, consent wording, retention policy, and access control should follow your institution's approved research protocol.

## Files

- `features.py` — notebook-derived handcrafted feature extraction.
- `export_model.py` — full-data deployment-model exporter.
- `app.py` — FastAPI `/score` and `/health` endpoints.
- `model/` — frozen model bundle, manifest, model dependency pins.
- `Dockerfile` — deployment image.
- `render.yaml` — example Render configuration.
- `lovable_supabase_edge_function.ts` — server-side Lovable/Supabase bridge.
- `.env.example` — local environment settings.
- `test_request.py` — API smoke test.
