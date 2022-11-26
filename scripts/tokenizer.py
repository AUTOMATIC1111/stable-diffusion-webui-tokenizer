import html

from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from modules import script_callbacks, shared

import gradio as gr


css = """
@media (prefers-color-scheme: dark) {
.tokenizer-token{
    cursor: pointer;
}
.tokenizer-token-0 {background: rgba(255, 0, 0, 0.2);}
.tokenizer-token-0:hover {background: rgba(255, 0, 0, 0.4);}
.tokenizer-token-1 {background: rgba(0, 255, 0, 0.2);}
.tokenizer-token-1:hover {background: rgba(0, 255, 0, 0.4);}
.tokenizer-token-2 {background: rgba(0, 0, 255, 0.2);}
.tokenizer-token-2:hover {background: rgba(0, 0, 255, 0.4);}
.tokenizer-token-3 {background: rgba(255, 156, 0, 0.2);}
.tokenizer-token-3:hover {background: rgba(255, 156, 0, 0.4);}
}
@media (prefers-color-scheme: light) {
.tokenizer-token{
    cursor: pointer;
}
.tokenizer-token-0 {background: rgba(255, 0, 0, 0.1);}
.tokenizer-token-0:hover {background: rgba(255, 0, 0, 0.2);}
.tokenizer-token-1 {background: rgba(0, 255, 0, 0.1);}
.tokenizer-token-1:hover {background: rgba(0, 255, 0, 0.2);}
.tokenizer-token-2 {background: rgba(0, 0, 255, 0.1);}
.tokenizer-token-2:hover {background: rgba(0, 0, 255, 0.2);}
.tokenizer-token-3 {background: rgba(255, 156, 0, 0.1);}
.tokenizer-token-3:hover {background: rgba(255, 156, 0, 0.2);}
}
"""


def tokenize(text, current_step=1, total_step=1, AND_block=0, simple_input=False, input_is_ids=False):
    clip: FrozenCLIPEmbedder = shared.sd_model.cond_stage_model.wrapped

    token_count = None
    if input_is_ids:
        tokens = [int(x.strip()) for x in text.split(",")]
    elif simple_input:
        tokens = clip.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    else:
        from modules import sd_hijack, prompt_parser
        from functools import reduce
        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, int(total_step))
        flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
        prompts = [prompt_text for step, prompt_text in flat_prompts]

        def find_current_prompt_idx(c_step, a_block):
            _idx = 0
            for i, prompts_block in enumerate(prompt_schedules):
                for step_prompt_chunk in prompts_block:
                    if i == a_block:
                        if c_step <= step_prompt_chunk[0]:
                            return _idx
                    _idx += 1

        idx = find_current_prompt_idx(current_step, AND_block)
        tokens, token_count, max_length = [sd_hijack.model_hijack.tokenize(prompt) for prompt in prompts][idx]

    vocab = {v: k for k, v in clip.tokenizer.get_vocab().items()}

    code = ''
    ids = []

    current_ids = []
    class_index = 0

    def dump(last=False):
        nonlocal code, ids, current_ids

        words = [vocab.get(x, "") for x in current_ids]

        def wordscode(ids, word):
            nonlocal class_index
            if ids != [clip.tokenizer.eos_token_id]:
                res = f"""<span class='tokenizer-token tokenizer-token-{class_index%4}' title='{html.escape(", ".join([str(x) for x in ids]))}'>{html.escape(word)}</span>"""
            else:
                res = f"""<span class='tokenizer-token tokenizer-token-4' title='{html.escape(", ".join([str(x) for x in ids]))}'>{html.escape(word)}</span>"""
            class_index += 1
            return res

        try:
            word = bytearray([clip.tokenizer.byte_decoder[x] for x in ''.join(words)]).decode("utf-8")
        except UnicodeDecodeError:
            if last:
                word = "❌" * len(current_ids)
            elif len(current_ids) > 4:
                id = current_ids[0]
                ids += [id]
                local_ids = current_ids[1:]
                code += wordscode([id], "❌")

                current_ids = []
                for id in local_ids:
                    current_ids.append(id)
                    dump()

                return
            else:
                return

        word = word.replace("</w>", " ")

        code += wordscode(current_ids, word)
        ids += current_ids

        current_ids = []

    for token in tokens:
        token = int(token)
        current_ids.append(token)

        dump()

    dump(last=True)

    if token_count is None:
        token_count = len(ids)
    ids_html = f"""
<p>
Token count: {token_count}/{len(ids)}<br>
{", ".join([str(x) for x in ids])}
</p>
"""

    return code, ids_html


def add_tab():
    with gr.Blocks(analytics_enabled=False, css=css) as ui:
        gr.HTML(f"""
<style>{css}</style>
<p>
Before your text is sent to the neural network, it gets turned into numbers in a process called tokenization. These tokens are how the neural network reads and interprets text. Thanks to our great friends at Shousetsu愛 for inspiration for this feature.<br>
Depending on your setting, text will be first parsed by webui to calculate prompt attention like (text) and [text], and scheduler like [a:b:0.5], and the capital AND like a AND b before tokenization. This extension processes your text like this as well.<br>
To disable this feature, check on "Don't parse webui special grammar".
</p>
""")

        with gr.Tabs() as tabs:
            with gr.Tab("Text input", id="input_text"):
                prompt = gr.Textbox(label="Prompt", elem_id="tokenizer_prompt", show_label=False, lines=8, placeholder="Prompt for tokenization")
                go = gr.Button(value="Tokenize", variant="primary")
                is_simple = gr.Checkbox(label="Don't parse webui special grammar", interactive=True)
                with gr.Row():
                    current_step = gr.Number(label='Current sampling steps', value=1, step=1, interactive=True)
                    total_step = gr.Number(label='Total sampling steps', value=28, step=1, interactive=True)
                    and_block = gr.Number(label='Which block of prompts (separated by AND) to tokenize', value=0, step=1, interactive=True)

            with gr.Tab("ID input", id="input_ids"):
                prompt_ids = gr.Textbox(label="Prompt", elem_id="tokenizer_prompt", show_label=False, lines=8, placeholder="Ids for tokenization (example: 9061, 631, 736)")
                go_ids = gr.Button(value="Tokenize", variant="primary")

        with gr.Tabs():
            with gr.Tab("Text"):
                tokenized_text = gr.HTML(elem_id="tokenized_text")

            with gr.Tab("Tokens"):
                tokens = gr.HTML(elem_id="tokenized_tokens")

        go.click(
            fn=tokenize,
            inputs=[prompt, current_step, total_step, and_block, is_simple],
            outputs=[tokenized_text, tokens],
        )

        go_ids.click(
            fn=lambda x: tokenize(x, input_is_ids=True),
            inputs=[prompt_ids],
            outputs=[tokenized_text, tokens],
        )

    return [(ui, "Tokenizer", "tokenizer")]


script_callbacks.on_ui_tabs(add_tab)
