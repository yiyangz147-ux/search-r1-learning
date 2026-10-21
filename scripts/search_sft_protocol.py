"""Shared text boundaries for Qwen3-Base SFT and generated reward monitoring.

The model sees the original Search-R1-style raw prompt, without an Instruct
chat template. Only whitespace at assistant/environment boundaries is
canonicalized; retrieved passage text and assistant reasoning are preserved.
"""

from prepare_data import PROMPT


PROTOCOL_VERSION = "base-search-r1-newline-observations-v2"
THINK_TOKEN_IDS = {"<think>": 151667, "</think>": 151668}


def build_prompt(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")
    # This deliberately matches the historical preparation/evaluation rule.
    question = question if question.endswith("?") else question + "?"
    return PROMPT.format(question=question)


def canonical_assistant(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Assistant action must be a nonempty string")
    return content.strip()


def canonical_information(content):
    if not isinstance(content, str):
        raise ValueError("Information must be a string")
    content = content.strip()
    if not content.startswith("<information>") or not content.endswith("</information>"):
        raise ValueError("Information must be a complete <information> block")
    return "\n" + content + "\n"


def canonical_training_segments(segments):
    """Give the search-closing newline to the preceding assistant loss region.

    Qwen tokenizes the last characters of ``</search>\\n`` as one ``>\\n``
    token. Masking the newline as part of the environment would mask that
    token's closing ``>`` too. Moving only the region boundary preserves the
    exact inference text while supervising the complete assistant delimiter.
    """
    result = []
    for index, segment in enumerate(segments):
        source_content = segment["content"]
        if segment["supervised"]:
            content = canonical_assistant(source_content)
            if index + 1 < len(segments):
                if segments[index + 1]["supervised"]:
                    raise ValueError("Assistant/environment regions must alternate")
                content += "\n"
        else:
            if index == 0 or not segments[index - 1]["supervised"]:
                raise ValueError("Information must follow an assistant action")
            # Its initial newline was assigned to the assistant region above.
            content = canonical_information(source_content)[1:]
        result.append({**segment, "content": content,
                       "boundary_whitespace_changed": source_content != content})
    return result
