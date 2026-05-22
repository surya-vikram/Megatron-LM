"""Shared evaluation and smoke-data assets for Gemma3 SFT."""

from __future__ import annotations


GOLD_MEDICAL_SAMPLES = [
    {
        "id": "gold_medical_1",
        "user": "What are the differential diagnoses for a patient with acute shortness of breath and chest pain?",
        "model": "Differential diagnosis includes myocardial infarction, pulmonary embolism, and aortic dissection.",
    },
    {
        "id": "gold_medical_2",
        "user": "What initial tests should be ordered for suspected acute coronary syndrome?",
        "model": "Stat EKG and troponin levels should be ordered immediately.",
    },
    {
        "id": "gold_medical_3",
        "user": "What are common physical findings in a patient with pulmonary embolism?",
        "model": "Common findings include tachypnea, tachycardia, and potentially decreased oxygen saturation (e.g., 92% on room air).",
    },
    {
        "id": "gold_medical_4",
        "user": "What is the standard emergency management for acute chest pain?",
        "model": "Emergency management involves supplemental oxygen, aspirin, and sublingual nitroglycerin.",
    },
    {
        "id": "gold_medical_5",
        "user": "Which imaging modality is preferred to rule out pulmonary embolism?",
        "model": "A CT pulmonary angiogram (CTPA) is the planned diagnostic imaging to rule out pulmonary embolism.",
    },
]


REASONING_EVAL_TASKS = [
    {
        "id": "clock_arithmetic",
        "kind": "exact",
        "prompt": "A train leaves at 3:00 PM and arrives 2 hours 45 minutes later. Reply with only the arrival time.",
        "accepted_answers": ["5:45 pm", "5:45pm", "17:45"],
        "reference_answer": "5:45 PM",
    },
    {
        "id": "sorting_numbers",
        "kind": "exact",
        "prompt": "Sort these numbers from smallest to largest and reply only with the ordered list: 12, 2, 9, 7",
        "accepted_answers": ["2, 7, 9, 12", "2 7 9 12"],
        "reference_answer": "2, 7, 9, 12",
    },
    {
        "id": "word_classification",
        "kind": "exact",
        "prompt": "In the sentence 'The curious engineer fixed the broken switch,' reply with only the adjective that describes the engineer.",
        "accepted_answers": ["curious"],
        "reference_answer": "curious",
    },
    {
        "id": "set_reasoning",
        "kind": "exact",
        "prompt": "All ravenous wugs are blue. No blue things are tiny. Can a ravenous wug be tiny? Reply only with yes or no and one short reason.",
        "accepted_answers": [
            "no because blue things are not tiny",
            "no - blue things are not tiny",
            "no, because blue things are not tiny",
        ],
        "reference_answer": "No, because blue things are not tiny.",
    },
    {
        "id": "api_design_tradeoff",
        "kind": "keywords",
        "prompt": (
            "You need to choose between a real-time API call and a nightly batch job for fraud detection. "
            "Give a short recommendation that mentions latency, throughput, and operational complexity."
        ),
        "required_keywords": ["latency", "throughput", "complexity"],
        "optional_keywords": ["batch", "real-time", "trade-off"],
        "min_word_count": 25,
        "reference_answer": (
            "A real-time API reduces latency for blocking fraud, while a nightly batch job improves throughput and can lower operational complexity for non-urgent review."
        ),
    },
    {
        "id": "incident_response_plan",
        "kind": "keywords",
        "prompt": (
            "A deployment caused error rates to spike. In 3-4 sentences, describe the first actions you would take. "
            "Include reproduction, logs, and rollback."
        ),
        "required_keywords": ["reproduce", "logs", "rollback"],
        "optional_keywords": ["metrics", "blast radius", "incident"],
        "min_word_count": 30,
        "reference_answer": (
            "First reproduce the issue, inspect logs and metrics to confirm the blast radius, and prepare a rollback if the deployment is the cause."
        ),
    },
    {
        "id": "bounded_refusal",
        "kind": "keywords",
        "prompt": (
            "Predict the exact closing price of Bitcoin 30 days from now. "
            "Give a careful response that acknowledges uncertainty."
        ),
        "required_keywords": ["cannot", "predict", "uncertain"],
        "optional_keywords": ["market", "volatility", "price"],
        "forbidden_keywords": ["guaranteed", "exactly will be"],
        "min_word_count": 20,
        "reference_answer": (
            "I cannot predict the exact price because the market is uncertain and volatile."
        ),
    },
    {
        "id": "brief_planning",
        "kind": "keywords",
        "prompt": (
            "Write a short plan for validating a new model training pipeline before a long production run. "
            "Mention a smoke test, checkpointing, and evaluation."
        ),
        "required_keywords": ["smoke", "checkpoint", "evaluation"],
        "optional_keywords": ["overfit", "resume", "metrics"],
        "min_word_count": 25,
        "reference_answer": (
            "Start with a smoke test, confirm checkpointing and resume work, then run evaluation before the long job."
        ),
    },
]


def build_pair_sample(user_text: str, model_text: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "model", "content": model_text},
        ]
    }

