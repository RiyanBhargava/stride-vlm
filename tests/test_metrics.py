from stride.evaluation.metrics import normalize_answer, score_prediction
from stride.evaluation.runner import format_evaluation_prompt


def test_normalize_answer():
    assert normalize_answer("The, RED cat!") == "red cat"
    assert normalize_answer("Two donuts") == "2 donuts"
    assert normalize_answer("dont") == "don't"


def test_metrics():
    assert score_prediction("Yes.", ("yes",), "yes_no") == 1
    assert score_prediction("The red cat", ("red cat",), "exact_match") == 1
    assert score_prediction("I choose (C).", ("C",), "multiple_choice") == 1
    references = (
        "two", "two", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
    )
    assert score_prediction("2", references, "vqa_accuracy") == 0.9


def test_choice_parser_does_not_read_letters_inside_words():
    assert score_prediction("The answer is D.", ("D",), "multiple_choice") == 1
    assert score_prediction("The answer is unknown.", ("E",), "multiple_choice") == 0


def test_evaluation_prompt_matches_metric():
    assert format_evaluation_prompt("Is it red?", "yes_no").endswith("yes or no.")
    assert "option letter" in format_evaluation_prompt("Choose", "multiple_choice")
    assert "Do not explain" in format_evaluation_prompt("Where?", "exact_match")
