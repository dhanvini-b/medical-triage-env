# Teaching an AI to Think Like a Triage Nurse

*A first-year student's journey building a medical RL environment from scratch*

---

## The Problem

Every day in hospitals across India, triage nurses make split-second decisions: who gets seen immediately, who can wait, who can come back tomorrow. Get it wrong and someone dies.

I wanted to know — can a small AI model learn to make these decisions? Not with a massive GPT-4 level model, but with a 0.5B parameter model trained on a free GPU in under 30 minutes?

That's what this project set out to answer.

---

## What I Built

A **Medical Triage OpenEnv environment** — a reinforcement learning playground where an AI agent reads patient data and assigns urgency levels.

The agent sees:
- Patient age and symptoms
- Vitals (heart rate, blood pressure, SpO2, temperature)
- Medical history

And must respond:
```
LEVEL: emergency
REASONING: Chest pain with low BP and SpO2 in a hypertensive smoker is a classic STEMI presentation.
```

The environment rewards correct decisions and — critically — **punishes dangerous under-triage more than over-triage**. Missing a true emergency gets -0.2 to -0.3. That design choice mirrors real clinical ethics: it's better to send a routine patient to emergency than to send an emergency patient home.

---

## The Patients

I designed 12 patient templates across 3 difficulty levels:

**Easy** — Obvious presentations. A 58-year-old with chest pain, sweating, and low BP. Classic heart attack. A 22-year-old with fever, stiff neck, and light sensitivity. Classic meningitis.

**Medium** — Ambiguous cases. A 34-year-old with fever and body aches — is it flu, dengue, or typhoid? A 29-year-old with sudden severe headache and a history of migraines — is this just a migraine, or a subarachnoid haemorrhage?

**Hard** — The ones that kill people when missed. A 71-year-old diabetic with jaw pain, nausea, and mild chest discomfort — atypical MI presentation. A 67-year-old with arm weakness that resolved after 20 minutes — TIA with 48-hour stroke risk.

The hard cases are hard precisely because they don't look like emergencies at first glance. That's the point.

---

## The Training Journey

### Attempt 1: GRPO alone

I started with GRPO directly on the base model. The result? The model said "urgent" for literally everything. Every patient, every time, always urgent.

Why? Because "urgent" was the safe default. With my original reward function, saying "urgent" on an emergency patient got 0.0 — neutral, not punished. The model learned that "urgent" was never catastrophically wrong. Rational, but useless.

**Fix:** Penalise emergency→urgent at -0.2. Make "urgent" actually risky on emergency patients.

### Attempt 2: Still GRPO alone, better rewards

Better, but still struggling. The model had no learning signal — when all 4 generated completions say "urgent" and all get similar rewards, GRPO has nothing to compare. Gradient is zero. Nothing changes.

**Insight:** GRPO needs variance to learn. A model that's already collapsed to one answer can't learn from reward comparison.

### Attempt 3: SFT → GRPO

The breakthrough. First, teach the model correct answers by imitation (SFT). Then, refine with rewards (GRPO).

SFT alone took the average reward from **0.045 to 0.786**. The model learned the format, learned the correct labels, learned to reference symptoms in its reasoning.

GRPO then pushed it further — particularly on the medium cases where over-triaging urgent patients as emergency needed correction.

Final result: **average 0.789**, all three difficulty levels above the 0.7 success threshold.

---

## What Surprised Me

**The hard cases were easier to learn than the medium cases.**

After training, hard patients scored 0.857 and easy patients scored 0.893. But medium only hit 0.617. Why?

Because 6 out of 9 original patients were emergencies. The model learned "when in doubt, say emergency" — which happens to be right for hard cases (they're all emergencies) but wrong for medium cases (which include urgent patients).

The model over-triages. It errs on the side of caution. For a triage AI, that's arguably the right failure mode — better to over-triage than to send someone home to die.

---

## What I Learned

1. **Reward function design is everything.** The "always urgent" problem wasn't a training problem — it was a reward problem. Once the reward made "urgent" genuinely risky, training improved immediately.

2. **SFT before RL is not optional for small models.** GRPO on a blank 0.5B model produces zero signal. It needs something to work with.

3. **Dataset balance matters.** 6 emergency, 2 urgent, 1 routine patients created a biased model. I added 3 more patients to balance it, which helped.

4. **Negative rewards are powerful.** The -0.3 for emergency→routine and -0.2 for emergency→urgent completely changed the model's behaviour. Neutral penalties teach nothing; negative penalties teach consequences.

---

## Try It

The environment is live at: https://huggingface.co/spaces/dhanvini/medical-triage-env

```bash
# Health check
curl https://dhanvini-medical-triage-env.hf.space/health

# Try a patient
curl -X POST https://dhanvini-medical-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard"}'
```

Training notebook: https://colab.research.google.com/drive/1ew0HUAFrmF5UXXt_Pzs6rTU2CUJHitkR?usp=sharing

---

*Built in 2 days at the Meta PyTorch OpenEnv Hackathon 2026.*  
*Dhanvini B — First Year BTech CSE AI, Amrita Vishwa Vidyapeetham*
