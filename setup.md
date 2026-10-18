# CiteLens

PDF question answering where you can see every step: parse, chunk, embed, search, rerank, answer, verify.

Stack: Streamlit, PostgreSQL + pgvector (Neon), fastembed, Ollama (llama3.1:8b).

---

## 1. Install tools (Windows PowerShell)

```powershell
winget install -e --id Python.Python.3.13
winget install -e --id Ollama.Ollama
```

Close PowerShell, open a new window, then check:

```powershell
py -0
ollama --version
```

## 2. Download the language model

```powershell
ollama pull llama3.1:8b
ollama list
```

Optional test:

```powershell
ollama run llama3.1:8b
```

Type `/bye` to exit.

## 3. Create the database (Neon, free)

Sign up at https://neon.tech, create a project, copy the **direct** connection string (pooling off).

In the Neon **SQL Editor** run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname FROM pg_extension;
```

## 4. Project folder

Put `rag_lab.py` and `requirements.txt` in a folder outside OneDrive, for example `C:\citelens`.

```powershell
mkdir C:\citelens
cd C:\citelens
```

`requirements.txt`:

```text
streamlit
pymupdf
pymupdf4llm
fastembed
psycopg[binary]
pgvector
numpy
pandas
ollama
openai
python-dotenv
```

## 5. Environment file

Create a file named `.env` in the same folder:

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
```

## 6. Python environment

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 7. Run

```powershell
streamlit run rag_lab.py
```

Opens at http://localhost:8501

## Every time you come back

```powershell
cd C:\citelens
.\venv\Scripts\Activate.ps1
streamlit run rag_lab.py
```

Stop the app with `Ctrl+C`.

## Reset

```powershell
deactivate
Remove-Item -Recurse -Force venv
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Switch to a free hosted model instead of Ollama (optional)

Replace the model lines in `.env`:

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_key_here
LLM_MODEL=llama-3.3-70b-versatile
```

## Mac / Linux equivalents

```bash
# Ollama
brew install ollama                              # Mac
curl -fsSL https://ollama.com/install.sh | sh    # Linux

ollama pull llama3.1:8b

# Python environment
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run rag_lab.py
```