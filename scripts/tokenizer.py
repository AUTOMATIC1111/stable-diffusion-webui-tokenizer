import html

from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from modules import script_callbacks, shared

import gradio as gr


css = """
.tokenizer-token{
    cursor: pointer;
}
.tokenizer-token-0 {background: rgba(255, 0, 0, 0.05);}
.tokenizer-token-0:hover {background: rgba(255, 0, 0, 0.15);}
.tokenizer-token-1 {background: rgba(0, 255, 0, 0.05);}
.tokenizer-token-1:hover {background: rgba(0, 255, 0, 0.15);}
.tokenizer-token-2 {background: rgba(0, 0, 255, 0.05);}
.tokenizer-token-2:hover {background: rgba(0, 0, 255, 0.15);}
.tokenizer-token-3 {background: rgba(255, 156, 0, 0.05);}
.tokenizer-token-3:hover {background: rgba(255, 156, 0, 0.15);}
"""


def tokenize(text):
    clip: FrozenCLIPEmbedder = shared.sd_model.cond_stage_model.wrapped
    tokens = clip.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

    vocab = {v: k for k, v in clip.tokenizer.get_vocab().items()}

    code = ''
    ids = []

    current_ids = []
    class_index = 0

    def dump():
        nonlocal code, ids, current_ids, class_index

        words = [vocab.get(x, "") for x in current_ids]

        try:
            word = bytearray([clip.tokenizer.byte_decoder[x] for x in ''.join(words)]).decode("utf-8")
        except UnicodeDecodeError:
            return

        word = word.replace("</w>", " ")

        code += f"""<span class='tokenizer-token tokenizer-token-{class_index%4}' title='{html.escape(", ".join([str(x) for x in current_ids]))}'>{html.escape(word)}</span>"""
        ids += current_ids
        class_index += 1

        current_ids = []

    for token in tokens:
        token = int(token)
        current_ids.append(token)

        dump()

    dump()

    return code, ids


def add_tab():
    with gr.Blocks(analytics_enabled=False, css=css) as ui:
        gr.HTML(f"""
<style>{css}</style>
<p>
Before your text is sent to the neural network, it gets turned into numbers in a process called tokenization. These tokens are how the neural network reads and interprets text. Thanks to our great friends at Shousetsu愛 for inspiration for this feature.
</p>
""")
        prompt = gr.Textbox(label="Prompt", elem_id="tokenizer_prompt", show_label=False, lines=8, placeholder="Prompt for tokenization")

        go = gr.Button(value="Tokenize", variant="primary")

        with gr.Tabs():
            with gr.Tab("Text"):
                tokenized_text = gr.HTML(elem_id="tokenized_text")

            with gr.Tab("Tokens"):
                tokens = gr.Text(elem_id="tokenized_tokens", show_label=False)

        go.click(
            fn=tokenize,
            inputs=[prompt],
            outputs=[tokenized_text, tokens],
        )

    return [(ui, "Tokenizer", "tokenizer")]


script_callbacks.on_ui_tabs(add_tab)
