from typing import Optional, Any
from pydantic import BaseModel

# ── Models ───────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    patient_id: str
    age: int
    symptoms: list[str]
    vitals: dict[str, float]
    history: list[str]
    step_number: int
    reward: float = 0.0
    done: bool = False

class Action(BaseModel):
    triage_level: str
    reasoning: str

class State(BaseModel):
    task: str
    step: int
    done: bool
    last_reward: float

# Use these as our typed models
TriageObservation = Observation
TriageAction = Action
TriageState = State

# ── Tasks ─────────────────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "patient_id": "P001",
        "age": 58,
        "symptoms": ["chest pain", "shortness of breath", "sweating"],
        "vitals": {"heart_rate": 118, "bp_systolic": 90, "bp_diastolic": 60, "spo2": 91, "temperature": 37.1},
        "history": ["hypertension", "smoker"],
        "correct_level": "emergency",
        "description": "Classic heart attack presentation"
    },
    "medium": {
        "patient_id": "P002",
        "age": 34,
        "symptoms": ["fever", "fatigue", "mild headache", "body aches"],
        "vitals": {"heart_rate": 98, "bp_systolic": 118, "bp_diastolic": 76, "spo2": 97, "temperature": 38.6},
        "history": ["no significant history"],
        "correct_level": "urgent",
        "description": "Could be flu, dengue, or typhoid — needs workup but not emergency"
    },
    "hard": {
        "patient_id": "P003",
        "age": 71,
        "symptoms": ["mild chest discomfort", "nausea", "jaw pain", "fatigue for 2 days"],
        "vitals": {"heart_rate": 88, "bp_systolic": 148, "bp_diastolic": 92, "spo2": 95, "temperature": 37.4},
        "history": ["diabetes", "hypertension", "previous stroke 3 years ago"],
        "correct_level": "emergency",
        "description": "Atypical MI presentation in elderly diabetic — easy to miss"
    }
}

# ── Grader ────────────────────────────────────────────────────────────────────

def grade_action(action: TriageAction, task: dict) -> float:
    correct = task["correct_level"]
    chosen = action.triage_level.lower().strip()

    level_score = 0.0
    if chosen == correct:
        level_score = 0.7
    elif (correct == "emergency" and chosen == "urgent") or (correct == "urgent" and chosen == "emergency"):
        level_score = 0.2
    elif chosen == "routine" and correct != "routine":
        level_score = 0.0
    elif chosen == "urgent" and correct == "routine":
        level_score = 0.3

    reasoning_lower = action.reasoning.lower()
    key_terms = task["symptoms"] + list(task["history"])
    matched = sum(1 for term in key_terms if term.lower() in reasoning_lower)
    reasoning_score = min(matched / max(len(key_terms), 1), 1.0) * 0.3

    return round(level_score + reasoning_score, 2)

# ── Environment ───────────────────────────────────────────────────────────────

class MedicalTriageEnv:

    def __init__(self, task_name: str = "easy"):
        
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
        self.task_name = task_name
        self.task = TASKS[task_name]
        self._step = 0
        self._done = False
        self._last_reward = 0.0
        self._current_obs: Optional[TriageObservation] = None

    def reset(self, seed=None, episode_id=None, **kwargs) -> TriageObservation:
        
        self._step = 0
        self._done = False
        self._last_reward = 0.0
        self._current_obs = TriageObservation(
            patient_id=self.task["patient_id"],
            age=self.task["age"],
            symptoms=self.task["symptoms"],
            vitals=self.task["vitals"],
            history=self.task["history"],
            step_number=0,
            reward=0.0,
            done=False
        )
        return self._current_obs

    def step(self, action: TriageAction, timeout_s=None, **kwargs) -> TriageObservation:
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        self._step += 1
        reward = grade_action(action, self.task)
        self._last_reward = reward
        self._done = True

        self._current_obs = TriageObservation(
            patient_id=self.task["patient_id"],
            age=self.task["age"],
            symptoms=self.task["symptoms"],
            vitals=self.task["vitals"],
            history=self.task["history"],
            step_number=self._step,
            reward=reward,
            done=True
        )
        return self._current_obs

    @property
    def state(self) -> TriageState:
        return TriageState(
            task=self.task_name,
            step=self._step,
            done=self._done,
            last_reward=self._last_reward
        )