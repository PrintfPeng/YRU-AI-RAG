from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys

# Ensure project root is in path for imports
sys.path.append('.')

from backend.services.sql_agent import generate_and_run_sql

app = FastAPI(title="SQL Agent Test API", version="0.1.0")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/sql")
async def run_sql(request: QueryRequest):
    try:
        result = generate_and_run_sql(request.query)
        return {"query": request.query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run with: uvicorn backend.api:app --reload
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)
