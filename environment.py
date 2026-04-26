from typing import Optional
from pydantic import BaseModel

# ── Models ─────────────────────────────────────────────────────────────────────

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

TriageObservation = Observation
TriageAction = Action
TriageState = State

# ── Patient Templates ──────────────────────────────────────────────────────────
# 9 patients across 3 difficulty levels (3 per level)
# environment.py only exposes one patient per task name (for the /reset endpoint).
# All 9 are defined here so server.py and train.py share the same source of truth.

PATIENTS = {
    "easy": [
        {
            "patient_id": "P001", "age": 58,
            "symptoms": ["chest pain", "shortness of breath", "sweating"],
            "vitals": {"heart_rate": 118, "bp_systolic": 90, "bp_diastolic": 60, "spo2": 91, "temperature": 37.1},
            "history": ["hypertension", "smoker"],
            "correct_level": "emergency",
            "description": "Classic heart attack presentation",
        },
        {
            "patient_id": "P002", "age": 22,
            "symptoms": ["high fever", "stiff neck", "sensitivity to light", "severe headache"],
            "vitals": {"heart_rate": 122, "bp_systolic": 95, "bp_diastolic": 65, "spo2": 97, "temperature": 39.8},
            "history": ["no significant history"],
            "correct_level": "emergency",
            "description": "Classic meningitis — obvious emergency",
        },
        {
            "patient_id": "P003", "age": 45,
            "symptoms": ["mild sore throat", "runny nose", "mild cough", "low grade fever"],
            "vitals": {"heart_rate": 82, "bp_systolic": 122, "bp_diastolic": 78, "spo2": 98, "temperature": 37.8},
            "history": ["no significant history"],
            "correct_level": "routine",
            "description": "Classic cold/viral URTI — clearly routine",
        },
    ],
    "medium": [
        {
            "patient_id": "P004", "age": 34,
            "symptoms": ["fever", "fatigue", "mild headache", "body aches"],
            "vitals": {"heart_rate": 98, "bp_systolic": 118, "bp_diastolic": 76, "spo2": 97, "temperature": 38.6},
            "history": ["no significant history"],
            "correct_level": "urgent",
            "description": "Could be flu, dengue, or typhoid — needs workup but not emergency",
        },
        {
            "patient_id": "P005", "age": 29,
            "symptoms": ["sudden severe headache", "vomiting", "neck stiffness"],
            "vitals": {"heart_rate": 88, "bp_systolic": 145, "bp_diastolic": 90, "spo2": 98, "temperature": 37.2},
            "history": ["migraines"],
            "correct_level": "emergency",
            "description": "Thunderclap headache — must rule out subarachnoid haemorrhage",
        },
        {
            "patient_id": "P006", "age": 52,
            "symptoms": ["right knee swelling", "pain on walking", "mild fever"],
            "vitals": {"heart_rate": 84, "bp_systolic": 130, "bp_diastolic": 82, "spo2": 98, "temperature": 38.0},
            "history": ["rheumatoid arthritis", "on immunosuppressants"],
            "correct_level": "urgent",
            "description": "Septic arthritis risk in immunosuppressed patient",
        },
    ],
    "hard": [
        {
            "patient_id": "P007", "age": 71,
            "symptoms": ["mild chest discomfort", "nausea", "jaw pain", "fatigue for 2 days"],
            "vitals": {"heart_rate": 88, "bp_systolic": 148, "bp_diastolic": 92, "spo2": 95, "temperature": 37.4},
            "history": ["diabetes", "hypertension", "previous stroke 3 years ago"],
            "correct_level": "emergency",
            "description": "Atypical MI in elderly diabetic — easy to miss",
        },
        {
            "patient_id": "P008", "age": 67,
            "symptoms": ["sudden confusion", "mild slurred speech", "arm weakness resolved after 20 minutes"],
            "vitals": {"heart_rate": 76, "bp_systolic": 158, "bp_diastolic": 94, "spo2": 97, "temperature": 37.0},
            "history": ["atrial fibrillation", "on aspirin"],
            "correct_level": "emergency",
            "description": "TIA — high stroke risk in next 48h, needs emergency workup",
        },
        {
            "patient_id": "P009", "age": 38,
            "symptoms": ["back pain", "mild abdominal pain", "dizziness on standing"],
            "vitals": {"heart_rate": 110, "bp_systolic": 102, "bp_diastolic": 68, "spo2": 97, "temperature": 37.1},
            "history": ["heavy smoker", "known aortic aneurysm"],
            "correct_level": "emergency",
            "description": "Leaking aortic aneurysm — catastrophic if missed",
        },
    ],
}

# For the /reset endpoint, each task name maps to patient index 0.
# This is the single patient the server exposes per task.
TASKS = {
    task: PATIENTS[task][0]
    for task in ["easy", "medium", "hard"]
}

VALID_LEVELS = {"emergency", "urgent", "routine"}

# ── Reward Function ────────────────────────────────────────────────────────────

def grade_action(action: "TriageAction", task: dict) -> float:
    """
    Reward breakdown:
      0.7  — correct triage level
      0.2  — reasoning quality (symptoms + history referenced)
      0.1  — format compliance

    Negative reward for catastrophic under-triage (missing a true emergency).
    This is the canonical reward used by both the server AND train.py.
    """
    correct = task["correct_level"]
    chosen = action.triage_level.lower().strip()

    # Component 1: triage level (0.7 max, negative for catastrophic errors)
    if chosen == correct:
        level_score = 0.7
    elif correct == "emergency" and chosen == "urgent":
        level_score = -0.2  # dangerous under-triage — penalised
    elif correct == "emergency" and chosen == "routine":
        level_score = -0.3  # catastrophic — patient could die
    elif correct == "urgent" and chosen == "emergency":
        level_score = 0.3   # over-triage — patient is safe, just wasteful
    elif correct == "urgent" and chosen == "routine":
        level_score = 0.0   # under-triage — no credit
    elif correct == "routine" and chosen == "urgent":
        level_score = 0.2   # minor over-triage
    elif correct == "routine" and chosen == "emergency":
        level_score = 0.0   # major over-triage
    else:
        level_score = 0.0

    # Component 2: reasoning quality (0.2 max)
    reasoning_lower = action.reasoning.lower()
    key_terms = task["symptoms"] + list(task["history"])
    matched = sum(1 for term in key_terms if term.lower() in reasoning_lower)
    reasoning_score = min(matched / max(len(key_terms), 1), 1.0) * 0.2

    # Component 3: format compliance (0.1)
    has_valid_level = chosen in VALID_LEVELS
    has_reasoning = len(action.reasoning.strip()) >= 10
    format_score = 0.1 if (has_valid_level and has_reasoning) else 0.0

    return round(level_score + reasoning_score + format_score, 2)

# ── Environment ────────────────────────────────────────────────────────────────

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
            done=False,
        )
        return self._current_obs

    def step(self, action: TriageAction, timeout_s=None, **kwargs) -> TriageObservation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

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
            done=True,
        )
        return self._current_obs

    @property
    def state(self) -> TriageState:
        return TriageState(
            task=self.task_name,
            step=self._step,
            done=self._done,
            last_reward=self._last_reward,
        )
