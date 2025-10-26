import os
import torch
import json
import time
import re
import asyncio
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict 
import argparse


def build_full_mcq_prompt(question_stem: str, options: dict) -> str:
    prompt = question_stem + "\n"
    for letter in sorted(options.keys()):
        prompt += f"({letter}) {options[letter]}\n"
    return prompt.strip()


def pick_choice_by_next_token_logits(prompt: str, tokenizer, model) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc)
    last_logits = out.logits[:, -1, :]  # [B, V]
    ids = {k: tokenizer.encode(k, add_special_tokens=False)[0] for k in ["A","B","C","D"]}
    scores = {k: last_logits[0, tid].item() for k, tid in ids.items()}
    return max(scores, key=scores.get)  # 'A'|'B'|'C'|'D'


def build_rag_prompt_text(question_stem: str, options: dict, documents: list) -> str:
    unique_docs = {doc['id']: doc for doc in documents}
    sorted_docs = sorted(unique_docs.values(), key=lambda x: float(x['score']), reverse=True)
    top_k_docs = sorted_docs 
    context_str = ""
    for i, doc in enumerate(top_k_docs):
        # (수정) 프롬프트 형식을 원본으로 복구
        context_str += f"--- Document {i+1} (ID: {doc.get('id', 'N/A')}) ---\n" 
        context_str += f"{doc.get('text', '')}\n\n"
    prompt = (
        f"--- Context ---\n"
        f"{context_str}\n"
        f"--- Question ---\n"
        f"{build_full_mcq_prompt(question_stem, options)}"
    )
    return prompt



QWEN_RAG_SYSTEM = (
    "You are an expert in multiple-choice QA.\n"
    "Rules:\n"
    "1) Read the question and options first.\n"
    "2) Use the documents only if they contain directly relevant facts.\n"
    "3) If the documents are irrelevant or conflicting, ignore them and use general knowledge.\n"
    "4) Strictly output a single letter among A, B, C, D. No words, no punctuation."
)

def build_rag_prompt_text(stem, options, documents, top_k=4, max_chars=600):
    qa_block = build_full_mcq_prompt(stem, options)
    def _trim(t, n): return (t or "").strip()[:n]
    # 문서 본문만, 메타 제거
    unique = []
    seen = set()
    for i, d in enumerate(documents or []):
        t = d.get("text","")
        if t and t not in seen:
            seen.add(t)
            unique.append(_trim(t, max_chars))
        if len(unique) >= top_k: break
    ctx = "\n\n".join(unique)
    return (
        f"{qa_block}\n\n"
        "Use the following facts only if relevant. If not, ignore them.\n Context:\n"
        f"{ctx}"
    ).strip()



def query_qwen_batched(all_messages: list, batch_size: int = 4) -> list:
    print(f"  > [Qwen] 총 {len(all_messages)}개 항목, 배치 크기 {batch_size}로 추론 시작...")
    all_responses = []
    for i in range(0, len(all_messages), batch_size):
        batch_messages = all_messages[i:i + batch_size]
        batch_texts = [
            qwen_tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in batch_messages
        ]
        model_inputs = qwen_tokenizer(
            batch_texts, return_tensors="pt", padding=True, padding_side='left'
        ).to(qwen_model.device)
        generated_ids = qwen_model.generate(
            **model_inputs, do_sample=False, max_new_tokens=10, pad_token_id=qwen_tokenizer.eos_token_id
        )
        response_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
        responses = qwen_tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        all_responses.extend([r.strip() for r in responses])

        if (i // batch_size) % 50 == 0 or i + len(batch_messages) == len(all_messages):
            print(f"  > [Qwen] {i + len(batch_messages)} / {len(all_messages)} 처리 완료...")
    return all_responses



try:
    gpt_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    print("Warning: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. GPT API 호출은 실패합니다.")
    gpt_client = None

async def query_gpt(
    question_stem: str, 
    options: dict, 
    documents: list = None, 
    model: str = "gpt-4o",
    semaphore: asyncio.Semaphore = None,
    i: int = 0,
    total: int = 0
) -> str:
    if not gpt_client:
        return "OpenAI API Key가 설정되지 않아 GPT를 호출할 수 없습니다."
    async with semaphore:
        if i % 50 == 0 or i == total - 1:
            print(f"  > [GPT] Starting request {i+1} / {total}...")
        if documents:
            system_prompt = "You are an expert in Multiple-Choice QA. Read the question and the provided context documents. First, try to find the answer within the documents. If the answer is not in the documents, use your general knowledge. Choose the best answer from options (A), (B), (C), and (D). Your answer must be only the single corresponding letter (e.g., \"A\")."
            user_prompt = build_rag_prompt_text(question_stem, options, documents)

        else:
            system_prompt = "You are an expert in Multiple-Choice QA. Read the following question and choose the most appropriate answer from options (A), (B), (C), and (D). Your answer must be only the single corresponding letter (e.g., \"A\")."
            user_prompt = build_full_mcq_prompt(question_stem, options)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = await gpt_client.chat.completions.create(
                model=model, messages=messages, max_tokens=10, temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"!! GPT API 에러 발생 (Request {i+1}): {e}")
            return f"Error: {e}"

def calculate_accuracy(predictions: list, truths: list) -> float:
    if not truths: return 0.0
    correct = 0
    total = len(truths)
    for pred, truth in zip(predictions, truths):
        match = re.search(r'([A-D])', pred.upper())
        if match and match.group(1) == truth.upper():
            correct += 1
    return (correct / total) * 100.0



async def main(args):
    
    JSON_FILE_PATH = args.retrieved_document_file
    QWEN_BATCH_SIZE = args.batch_size
    API_CONCURRENCY = args.api_concurrency
    MODEL_NAME = args.model_name
    SUMMARY_ONLY = args.summary_only
    DATASET_NAME = JSON_FILE_PATH.split('/')[-1]
    
    print(f"Loading retrieval results from {JSON_FILE_PATH}...")
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            all_qa_items = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at {JSON_FILE_PATH}")
        exit()
    
    print(f"Successfully loaded {len(all_qa_items)} raw items.")
    print("Start regrouping items into full MCQs...")

    # all_qa_items = all_qa_items[:20]

    # --- data creation ---
    grouped_questions = defaultdict(lambda: {"options": {}, "ctxs": [], "answer": None})
    option_pattern = re.compile(r'\s+\(([A-D])\)\s+(.*)')
    for item in all_qa_items:
        question_text = item["question"]
        match = option_pattern.search(question_text)
        if not match: continue 
        question_stem = question_text[:match.start()].strip() 
        option_letter, option_text = match.group(1), match.group(2)
        group = grouped_questions[question_stem]
        group["options"][option_letter] = option_text
        group["ctxs"].extend(item["ctxs"][:2])

        if not group["answer"]:
            group["answer"] = item["answers"][0] 
    

    items_to_process = [v for v in grouped_questions.values() if len(v["options"]) == 4]
    print(f"Regrouping complete. Found {len(items_to_process)} full MCQ items.")
    
    print(f"Processing all {len(items_to_process)} items...\n")

    print("Preparing all prompts and ground truths...")
    qwen_baseline_messages, qwen_rag_messages, gpt_inputs, ground_truths = [], [], [], []
    QWEN_BASELINE_SYSTEM = "You are an expert in Multiple-Choice QA. Read the following question and choose the most appropriate answer from options (A), (B), (C), and (D). Your answer must be only the single corresponding letter (e.g., \"A\")."
    QWEN_RAG_SYSTEM = "You are an expert in Multiple-Choice QA. Read the question and the provided context documents. First, try to find the answer within the documents. If the answer is not in the documents, use your general knowledge. Choose the best answer from options (A), (B), (C), and (D). Your answer must be only the single corresponding letter (e.g., \"A\")."

    for i, item in enumerate(items_to_process):
        stem = [k for k, v in grouped_questions.items() if v == item][0] 
        options, documents = item["options"], item["ctxs"]
        ground_truths.append(item["answer"]) 
        if args.run_qwen:
            qwen_baseline_messages.append([
                {"role": "system", "content": QWEN_BASELINE_SYSTEM},
                {"role": "user", "content": build_full_mcq_prompt(stem, options)}
            ])
            qwen_rag_messages.append([
                {"role": "system", "content": QWEN_RAG_SYSTEM},
                {"role": "user", "content": build_rag_prompt_text(stem, options, documents)}
            ])
        if args.run_gpt:
            gpt_inputs.append({"stem": stem, "options": options, "documents": documents})

    # Initialize answer list
    total_items_to_process = len(items_to_process)
    qwen_baseline_answers = ["SKIPPED"] * total_items_to_process
    qwen_rag_answers = ["SKIPPED"] * total_items_to_process
    gpt_baseline_answers = ["SKIPPED"] * total_items_to_process
    gpt_rag_answers = ["SKIPPED"] * total_items_to_process


    if args.run_qwen:
        print("\n--- 1. Qwen (Local) Batched Inference ---")
        start_time_qwen = time.time()
        qwen_baseline_answers = query_qwen_batched(qwen_baseline_messages, QWEN_BATCH_SIZE * 2)
        qwen_rag_answers = query_qwen_batched(qwen_rag_messages, QWEN_BATCH_SIZE)
        print(f"Qwen processing finished in {time.time() - start_time_qwen:.2f} seconds.")
    else:
        print("\n--- 1. Qwen (Local) Batched Inference SKIPPED ---")


    if args.run_gpt:
        print("\n--- 2. GPT (API) Concurrent Inference ---")
        print(f"Setting API concurrency limit to: {API_CONCURRENCY}")
        start_time_gpt = time.time()
        semaphore = asyncio.Semaphore(API_CONCURRENCY)
        baseline_tasks, rag_tasks = [], []
        
        total_items_gpt = len(gpt_inputs)
        for i, inputs in enumerate(gpt_inputs):
            baseline_tasks.append(
                query_gpt(inputs["stem"], inputs["options"], documents=None, 
                          semaphore=semaphore, i=i, total=total_items_gpt)
            )
            rag_tasks.append(
                query_gpt(inputs["stem"], inputs["options"], documents=inputs["documents"], 
                          semaphore=semaphore, i=i, total=total_items_gpt)
            )
        
        print(f"Running {total_items_gpt} Baseline GPT tasks concurrently...")
        gpt_baseline_answers = await asyncio.gather(*baseline_tasks)
        print(f"Running {total_items_gpt} RAG GPT tasks concurrently...")
        gpt_rag_answers = await asyncio.gather(*rag_tasks)
        print(f"GPT processing finished in {time.time() - start_time_gpt:.2f} seconds.")
    else:
        print("\n--- 2. GPT (API) Concurrent Inference SKIPPED ---")
    

    if not SUMMARY_ONLY:
        print("\n\n--- FINAL COMPARISON RESULTS ---")    
    qwen_base_acc = calculate_accuracy(qwen_baseline_answers, ground_truths)
    qwen_rag_acc = calculate_accuracy(qwen_rag_answers, ground_truths)
    gpt_base_acc = calculate_accuracy(gpt_baseline_answers, ground_truths)
    gpt_rag_acc = calculate_accuracy(gpt_rag_answers, ground_truths)


    if not SUMMARY_ONLY:
        print("\n--- Detailed Responses (Sample) ---")
    for i, item in enumerate(items_to_process):
        stem = [k for k, v in grouped_questions.items() if v == item][0]
        # print("\n--------------------------------")
        # print(f"Question #{i+1}: {build_full_mcq_prompt(stem, item['options'])}")
        # print(f"Ground Truth: {ground_truths[i]}")
        # print("--------------------------------")
        if args.run_qwen:
            
            # print(f"[Qwen Baseline]: {qwen_baseline_answers[i]}")
            # print(f"[qwen_rag_messages]: {qwen_rag_messages[i]}")

            current_pred = qwen_rag_answers[i]
            current_truth = ground_truths[i]

            # --- print qwen meassage and answer when model prediction is wrong ---
            is_correct = False
            match = re.search(r'([A-D])', current_pred.upper())
            if match and match.group(1) == current_truth.upper():
                is_correct = True            

            if not is_correct and not SUMMARY_ONLY:
                print("\n--------------------------------")
                print(f"Question #{i+1}: {build_full_mcq_prompt(stem, item['options'])}")
                print(f"Ground Truth: {ground_truths[i]}")
                print(f"[w/ retrieved documents] [qwen_rag_messages]: {qwen_rag_messages[i]}")
                
                model_answer = match.group(1) if match else "N/A"
                print(f"    -> Model Answer: {model_answer} (Ground Truth: {current_truth})")
                print("--------------------------------")
            # print(f"[Qwen RAG]     : {qwen_rag_answers[i]}")
        if args.run_gpt:
            current_pred = gpt_rag_answers[i]
            current_truth = ground_truths[i]

            # --- print qwen meassage and answer when model prediction is wrong ---
            is_correct = False
            match = re.search(r'([A-D])', current_pred.upper())
            if match and match.group(1) == current_truth.upper():
                is_correct = True            

            # if not is_correct and not SUMMARY_ONLY::
            #     print("\n--------------------------------")
            #     print(f"Question #{i+1}: {build_full_mcq_prompt(stem, item['options'])}")
            #     print(f"Ground Truth: {ground_truths[i]}")
            #     print(f"[GPT Baseline] : {gpt_baseline_answers[i]}")
            #     print(f"[GPT RAG]      : {gpt_rag_answers[i]}")

    if not SUMMARY_ONLY:
        print("\n--- All test items processed. ---")
    print(f"\n--- Accuracy Summary ({DATASET_NAME})---")
    if args.run_qwen:
        print(f"{MODEL_NAME} (Baseline) : {qwen_base_acc:.2f}%")
        print(f"{MODEL_NAME} (RAG)      : {qwen_rag_acc:.2f}%")
    if args.run_gpt:
        print(f"GPT  (Baseline) : {gpt_base_acc:.2f}%")
        print(f"GPT  (RAG)      : {gpt_rag_acc:.2f}%")

if __name__ == "__main__":
    # -- set argparse -- 
    parser = argparse.ArgumentParser(description="Run RAG vs. Baseline evaluation.")
    
    parser.add_argument('--retrieved_document_file', type=str, required=True, help="Path to the dense retriever's output JSON file")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for the Language Model")
    parser.add_argument( "--no-run-qwen", dest="run_qwen", action="store_false", default=True, help="Disable Qwen (default: enabled)" )
    parser.add_argument( "--no-run-gpt", dest="run_gpt", action="store_false", default=False, help="Enable GPT (default: disabled)" )
    parser.add_argument('--api_concurrency', type=int, default=1, help="Number of concurrent calls for API")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face model name")
    parser.add_argument("--summary_only", type=bool, default=True, help="Hide evaluation log and only show final results")
    
    args = parser.parse_args()


    # --- 1. Qwen batch inference ---
    QWEN_MODEL_ID = args.model_name
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID, dtype="auto", device_map="auto"
    )
    
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_model.config.pad_token_id = qwen_model.config.eos_token_id

    seed_val = 6101
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    asyncio.run(main(args))