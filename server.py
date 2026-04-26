from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any
from environment import MedicalTriageEnv, TriageAction, TriageObservation, TriageState

app = FastAPI()

env: Optional[MedicalTriageEnv] = None

class ResetRequest(BaseModel):
    task: str = "easy"

class StepRequest(BaseModel):
    triage_level: str
    reasoning: str

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env
    env = MedicalTriageEnv(task_name=req.task)
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    global env
    if env is None:
        return {"error": "Call /reset first"}
    action = TriageAction(triage_level=req.triage_level, reasoning=req.reasoning)
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": {"correct": env.task["correct_level"]},
    }

@app.get("/state")
def state():
    global env
    if env is None:
        return {"error": "Call /reset first"}
    return env.state.model_dump()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "medical-triage-env",
        "description": (
            "A medical triage environment where an AI agent assesses patient "
            "symptoms, vitals, and history to assign urgency levels."
        ),
        "version": "1.0.0",
        "tags": ["medical", "triage", "healthcare", "real-world"],
    }

@app.get("/schema")
def schema():
    return {
        "action": TriageAction.model_json_schema(),
        "observation": TriageObservation.model_json_schema(),
        "state": TriageState.model_json_schema(),
    }

@app.post("/mcp")
def mcp(request: dict[str, Any] = {}):
    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 1),
        "result": {
            "name": "medical-triage-env",
            "version": "1.0.0",
        },
    }

@app.get("/")
def root():
    return {
        "name": "Medical Triage Environment",
        "description": "OpenEnv-compatible medical triage RL environment",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata", "/schema"],
        "usage": "POST /reset with {\"task\": \"easy\"} to start"
    }
