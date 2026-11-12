from __future__ import annotations

import os
import unittest

from stale_qwen_eval.constants import MODEL_ID, MODEL_REVISION
from stale_qwen_eval.prompts import build_messages, chat_template_token_ids


@unittest.skipUnless(
    os.environ.get("RUN_HF_INTEGRATION") == "1",
    "set RUN_HF_INTEGRATION=1 to download the pinned tokenizer",
)
class HuggingFaceIntegrationTests(unittest.TestCase):
    def test_pinned_qwen_tokenizer_disables_thinking(self) -> None:
        try:
            import transformers
        except ImportError as error:
            self.skipTest(f"transformers is not installed: {error}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        messages = build_messages("history", "question", "dim1_query")
        token_ids = chat_template_token_ids(tokenizer, messages)
        rendered = tokenizer.decode(token_ids, skip_special_tokens=False)
        self.assertTrue(token_ids)
        # This pinned Qwen revision represents non-thinking mode by pre-filling a
        # closed, empty reasoning block. Generation begins after </think>.
        self.assertTrue(rendered.endswith("<think>\n\n</think>\n\n"))
        thinking_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        thinking_rendered = tokenizer.decode(thinking_ids, skip_special_tokens=False)
        self.assertNotIn("</think>", thinking_rendered)

