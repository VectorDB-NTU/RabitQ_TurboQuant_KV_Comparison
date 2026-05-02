"""
Unified Needle-in-a-Haystack evaluation for RaBitQ and TurboQuant.

This script supports:
1. the original OpenAI judge, and
2. local scoring paths that do not require GPT.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FRAMEWORK_DIR = REPO_DIR / "LLMTest_NeedleInAHaystack"
sys.path.insert(0, str(FRAMEWORK_DIR))
sys.path.insert(0, str(REPO_DIR))

from needlehaystack import LLMNeedleHaystackTester
from needlehaystack.evaluators import OpenAIEvaluator
from needlehaystack.evaluators.evaluator import Evaluator
from needlehaystack.providers import ModelProvider

NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
QUESTION = "What is the best thing to do in San Francisco?"
EXPECTED_ANSWER = "eat a sandwich and sit in Dolores Park on a sunny day."


class KeywordCoverageEvaluator(Evaluator):
    """Score = fraction of answer keywords found in the response (0-1 scale)."""
    CRITERIA = {}

    def __init__(self, expected_answer=EXPECTED_ANSWER):
        self.answer_words = set(expected_answer.lower().split())

    def evaluate_response(self, response: str) -> float:
        response_words = set(response.lower().split())
        return len(response_words & self.answer_words) / len(self.answer_words)


class RougeLikeEvaluator(Evaluator):
    """Approximate Token evaluate.py score: ROUGE-1 F1 * 10."""
    CRITERIA = {}

    def __init__(self, reference=NEEDLE):
        self.reference_tokens = reference.lower().split()

    def evaluate_response(self, response: str) -> float:
        pred_tokens = response.lower().split()
        ref_tokens = self.reference_tokens
        if not pred_tokens or not ref_tokens:
            return 0.0
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        overlap = sum((pred_counter & ref_counter).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1 * 10.0


class KVQuantProvider(ModelProvider):
    def __init__(self, model_name, backend, bits, prompt_style="legacy", max_new_tokens=300):
        if backend == "FP16":
            self.model_name = model_name.replace("/", "_") + "_FP16"
        else:
            self.model_name = model_name.replace("/", "_") + f"_{backend}_{bits}bit"
        self._hf_model_name = model_name
        self.backend = backend
        self.bits = bits
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens

        if backend == "FP16":
            self._load_baseline()
        else:
            self._load_quantized()

    def _load_baseline(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self._hf_model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    def _load_quantized(self):
        from transformers import AutoConfig, AutoTokenizer

        bits_config = {"2.5": (3, 2), "3.5": (4, 3)}
        outlier_bits, residual_bits = bits_config[self.bits]

        if self.backend == "turbo":
            from kvcache_quant.turbo_sketch import TurboSketch as SketchClass
        elif self.backend == "rabitq":
            from kvcache_quant.rabitq_sketch import RaBitQSketch as SketchClass
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        config = AutoConfig.from_pretrained(self._hf_model_name)
        config.attention_dropout = 0.0
        config.use_cache = True

        device = torch.device('cuda')
        generator = torch.Generator(device=device)
        generator.manual_seed(42)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        outlier_dim = 32
        residual_dim = head_dim - outlier_dim

        config.sketch_outlier = SketchClass(dimension=outlier_dim, bit_width=outlier_bits, rng=generator)
        config.sketch_residual = SketchClass(dimension=residual_dim, bit_width=residual_bits, rng=generator)
        config.sketch_value = SketchClass(dimension=head_dim, bit_width=2, rng=generator)

        config.outlier_count_general = outlier_dim
        config.key_quantization_bits = outlier_bits * head_dim
        config.value_quantization_bits = 2
        config.group_size = 32
        config.buffer_size = 128

        is_mistral = config.model_type == "mistral"
        if is_mistral:
            from kvcache_quant.modeling_mistral_kv_quant import MistralForCausalLM_KVQuant as ModelClass
        else:
            from kvcache_quant.modeling_llama_kv_quant import LlamaForCausalLM_KVQuant as ModelClass

        self.model = ModelClass.from_pretrained(
            pretrained_model_name_or_path=self._hf_model_name,
            config=config, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    async def evaluate_model(self, prompt):
        if isinstance(prompt, list):
            text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        seq_len = input_ids.shape[1]
        print(f"Prompt token length: {seq_len}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
                use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True)

        new_tokens = outputs.sequences[0, seq_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        del outputs, input_ids
        torch.cuda.empty_cache()
        return response

    def generate_prompt(self, context, retrieval_question):
        if self.prompt_style == "tsa_text":
            return (
                f"<|im_start|> This is a very long story book: <book> {context} </book>.\n"
                f"Based on the content of the book, Question: {retrieval_question}\n"
                "Answer:"
            )

        if self.prompt_style == "single_user":
            if getattr(self.tokenizer, "chat_template", None):
                return [
                    {
                        "role": "system",
                        "content": (
                            "You answer questions using only the provided document. "
                            "Give a short, direct answer. If the document does not contain the answer, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Document:\n"
                            f"{context}\n\n"
                            f"Question: {retrieval_question}\n"
                            "Answer using only the document and do not add outside information."
                        ),
                    },
                ]
            return (
                "You answer questions using only the provided document. "
                "Give a short, direct answer. If the document does not contain the answer, say so.\n\n"
                f"Document:\n{context}\n\n"
                f"Question: {retrieval_question}\n"
                "Answer using only the document and do not add outside information.\n\n"
                "Answer:"
            )

        if getattr(self.tokenizer, "chat_template", None):
            return [
                {"role": "system", "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"},
                {"role": "user", "content": context},
                {"role": "user", "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"}
            ]
        return f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct\n\n{context}\n\n{retrieval_question} Don't give information outside the document or repeat your findings\n\nAnswer:"

    def encode_text_to_tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens, context_length=None):
        return self.tokenizer.decode(tokens[:context_length], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Unified Needle-in-a-Haystack evaluation")
    parser.add_argument("--backend", type=str, required=True, choices=["FP16", "turbo", "rabitq"])
    parser.add_argument("--bits", type=str, default="2.5", choices=["2.5", "3.5"])
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--evaluator_model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--context_lengths_min", type=int, default=4000)
    parser.add_argument("--context_lengths_max", type=int, default=104000)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=15)
    parser.add_argument("--document_depth_percent_intervals", type=int, default=10)
    parser.add_argument("--results_version", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="result/needle")
    parser.add_argument("--final_context_length_buffer", type=int, default=200)
    parser.add_argument("--prompt_style", type=str, default="legacy",
                        choices=["legacy", "single_user", "tsa_text"])
    parser.add_argument("--evaluator", type=str, default="keyword",
                        choices=["keyword", "rouge", "openai"],
                        help="keyword: Token visualize-style local score; rouge: Token evaluate.py-style ROUGE-1 F1 * 10; openai: GPT judge")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    if args.evaluator == "openai":
        api_key = os.getenv("NIAH_EVALUATOR_API_KEY")
        if not api_key:
            print("ERROR: Set NIAH_EVALUATOR_API_KEY environment variable")
            sys.exit(1)

    work_dir = Path(args.output_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    label = f"{args.backend}_{args.bits}bit" if args.backend != "FP16" else "FP16"
    print(f"Method: {label}")
    print(f"Model: {args.model}")
    print(f"Context: {args.context_lengths_min} - {args.context_lengths_max}")
    print(f"Evaluator: {args.evaluator}")
    if args.evaluator == "openai":
        print(f"Evaluator model: {args.evaluator_model}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Final context buffer: {args.final_context_length_buffer}")
    print(f"Working dir: {work_dir}")

    provider = KVQuantProvider(
        model_name=args.model,
        backend=args.backend,
        bits=args.bits,
        prompt_style=args.prompt_style,
        max_new_tokens=args.max_new_tokens,
    )

    if args.evaluator == "openai":
        evaluator = OpenAIEvaluator(
            model_name=args.evaluator_model,
            question_asked=QUESTION,
            true_answer=NEEDLE)
    elif args.evaluator == "rouge":
        evaluator = RougeLikeEvaluator(reference=NEEDLE)
    else:
        evaluator = KeywordCoverageEvaluator(expected_answer=EXPECTED_ANSWER)

    tester = LLMNeedleHaystackTester(
        model_to_test=provider,
        evaluator=evaluator,
        needle=NEEDLE,
        haystack_dir="PaulGrahamEssays",
        retrieval_question=QUESTION,
        results_version=args.results_version,
        context_lengths_min=args.context_lengths_min,
        context_lengths_max=args.context_lengths_max,
        context_lengths_num_intervals=args.context_lengths_num_intervals,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        final_context_length_buffer=args.final_context_length_buffer,
        save_results=True,
        save_contexts=False,
        print_ongoing_status=True)

    tester.start_test()


if __name__ == "__main__":
    main()
