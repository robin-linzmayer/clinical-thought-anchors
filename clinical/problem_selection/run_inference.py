"""Run model inference on prepared medical QA questions with checkpointing."""

import json
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from config import get_parser, PROMPT_TEMPLATE
from med_eval import extract_answer


def load_model(args):
    """Load model and tokenizer, mirroring generate_rollouts.py:81-118."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.quantize and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )

    model.eval()
    print("Model loaded successfully")
    return model, tokenizer


def generate_batch(model, tokenizer, prompts, args):
    """Generate completions for a batch of prompts, mirroring generate_rollouts.py:166-221."""
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    results = []
    for i, (input_ids, output_ids) in enumerate(zip(inputs["input_ids"], outputs)):
        input_length = len(input_ids)
        generated_text = tokenizer.decode(output_ids[input_length:], skip_special_tokens=True)
        num_tokens = len(output_ids) - input_length
        results.append({"text": generated_text, "num_tokens": int(num_tokens)})

    return results


def process_question(question, model, tokenizer, args, inference_dir):
    """Run NUM_TRIALS inferences for a single question with checkpointing."""
    problem_idx = question["problem_idx"]
    problem_dir = inference_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    results_file = problem_dir / "results.json"

    # Load existing results if resuming
    existing_trials = []
    if args.resume and results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        existing_trials = saved.get("trials", [])
        if len(existing_trials) >= args.num_trials:
            return existing_trials

    num_needed = args.num_trials - len(existing_trials)
    prompt = PROMPT_TEMPLATE.format(question=question["question"])

    # Generate in batches
    all_new_trials = []
    for batch_start in range(0, num_needed, args.batch_size):
        batch_end = min(batch_start + args.batch_size, num_needed)
        batch_prompts = [prompt] * (batch_end - batch_start)

        batch_results = generate_batch(model, tokenizer, batch_prompts, args)

        for j, result in enumerate(batch_results):
            trial_idx = len(existing_trials) + len(all_new_trials)
            raw_output = result["text"]
            extracted = extract_answer(raw_output)

            all_new_trials.append({
                "trial_idx": trial_idx,
                "raw_output": raw_output,
                "extracted_answer": extracted,
                "num_tokens": result["num_tokens"],
            })

    # Save checkpoint with question metadata for easy reference
    all_trials = existing_trials + all_new_trials
    output = {
        "problem_idx": question["problem_idx"],
        "question": question["question"],
        "gt_answer": question["gt_answer"],
        "trials": all_trials,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return all_trials


def main():
    parser = get_parser("Run inference on medical QA questions")
    args = parser.parse_args()

    # Set seeds (matching generate_rollouts.py:69-75)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load prepared questions
    output_dir = Path(args.output_dir)
    questions_path = output_dir / "prepared_questions.json"
    if not questions_path.exists():
        print(f"Prepared questions not found at {questions_path}. Run prepare_data.py first.")
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Filter to included problems if specified
    if args.include_problems:
        include_set = set(int(x) for x in args.include_problems.split(","))
        questions = [q for q in questions if q["problem_idx"] in include_set]

    print(f"Processing {len(questions)} questions, {args.num_trials} trials each")

    # Setup inference output directory
    model_name = args.model.split("/")[-1]
    inference_dir = output_dir / "inference" / model_name
    inference_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args)

    # Process each question
    for question in tqdm(questions, desc="Questions"):
        trials = process_question(question, model, tokenizer, args, inference_dir)
        print(f"  Problem {question['problem_idx']}: {len(trials)} trials complete")

    print("Inference complete.")


if __name__ == "__main__":
    main()
