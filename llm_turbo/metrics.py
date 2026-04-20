import re
import string
import collections


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(prediction, ground_truth):
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    return _f1_score(prediction, ground_truth)


def rouge_score(prediction, ground_truth, **kwargs):
    try:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs.get("all_classes", [])
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    if len(em_match_list) == 1:
        return 1.0 if em_match_list[0] == ground_truth else 0.0
    return 1.0 / len(em_match_list) if ground_truth in em_match_list else 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    return 1.0 if gt_id in numbers else 0.0


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for n in numbers if n == ground_truth)
    return min(right_num / max(len(numbers), 1), 1.0)


def code_sim_score(prediction, ground_truth, **kwargs):
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        from thefuzz import fuzz

    def _filter(code):
        lines = code.split("\n")
        filtered = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            if stripped.startswith("```"):
                continue
            if stripped:
                filtered.append(line)
        return "\n".join(filtered)

    pred_filtered = _filter(prediction)
    gt_filtered = _filter(ground_truth)
    if not pred_filtered or not gt_filtered:
        return 0.0
    return fuzz.ratio(pred_filtered, gt_filtered) / 100.0


def _normalize_zh_answer(s):
    import jieba
    return " ".join(jieba.cut(s))


def rouge_zh_score(prediction, ground_truth, **kwargs):
    try:
        from rouge import Rouge
        prediction = _normalize_zh_answer(prediction)
        ground_truth = _normalize_zh_answer(ground_truth)
        rouge = Rouge()
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction = _normalize_zh_answer(prediction)
    ground_truth = _normalize_zh_answer(ground_truth)
    return _f1_score(prediction, ground_truth)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    gt_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    return 1.0 if gt_id in numbers else 0.0
