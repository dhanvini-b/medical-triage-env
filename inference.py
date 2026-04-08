import os
import textwrap
from typing import Optional
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "medical-triage-env"
MAX_STEPS = 1
SUCCESS_THRESHOLD = 0.7

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a medical triage assistant. You will receive a patient's details
    including age, symptoms, vitals, and medical history.
    
    You must respond with EXACTLY this format, nothing else:
    LEVEL: <emergency|urgent|routine>
    REASONING: <your clinical reasoning in one sentence>
    
    Triage levels:
    - emergency: life-threatening, needs immediate attention
    - urgent: serious but stable, needs attention within hours
    - routine: non-urgent, can wait for regular appointment
""").strip()

def log_start(task, model):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_prompt(obs: dict) -> str:
    return textwrap.dedent(f"""
        Patient ID: {obs['patient_id']}
        Age: {obs['age']}
        Symptoms: {', '.join(obs['symptoms'])}
        Vitals:
          Heart Rate: {obs['vitals']['heart_rate']} bpm
          Blood Pressure: {obs['vitals']['bp_systolic']}/{obs['vitals']['bp_diastolic']} mmHg
          SpO2: {obs['vitals']['spo2']}%
          Temperature: {obs['vitals']['temperature']} C
        Medical History: {', '.join(obs['history'])}
        
        What is the correct triage level for this patient?
    """).strip()

def parse_model_response(text: str) -> tuple[str, str]:
    level = "routine"
    reasoning = text
    for line in text.splitlines():
        if line.upper().startswith("LEVEL:"):
            level = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return level, reasoning

def get_model_action(client: OpenAI, obs: dict) -> tuple[str, str]:
    prompt = build_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_model_response(text)
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return "routine", "fallback due to error"

import requests

def run_task(client: OpenAI, task_name: str, base_url: str):
    log_start(task_name, MODEL_NAME)
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        res = requests.post(f"{base_url}/reset", json={"task": task_name})
        obs = res.json()

        for step in range(1, MAX_STEPS + 1):
            level, reasoning = get_model_action(client, obs)
            action_str = f"triage_level={level}"

            res = requests.post(f"{base_url}/step", json={
                "triage_level": level,
                "reasoning": reasoning
            })
            result = res.json()

            reward = result.get("reward", 0.0)
            done = result.get("done", True)
            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done)

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
    finally:
        log_end(success, steps_taken, score, rewards)

    return score

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")

    print(f"\n=== Medical Triage Env Baseline ===\n", flush=True)
    scores = []
    for task in TASKS:
        score = run_task(client, task, base_url)
        scores.append(score)
        print(f"  -> {task}: {score:.3f}\n", flush=True)

    avg = sum(scores) / len(scores)
    print(f"=== Average score: {avg:.3f} ===", flush=True)

if __name__ == "__main__":
    main()