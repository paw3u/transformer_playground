from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline, set_seed

device = 0 if torch.cuda.is_available() else -1
generator = pipeline('text-generation', model='gpt2-large', device=device)
set_seed(42)
app = Flask(__name__)

@app.route('/')
def playground_main():
   return render_template('main.html')

@app.route('/generate', methods=['GET', 'POST'])
def playground_generate():
   global generator
   if request.method == 'POST':
        req = request.get_json()
        prompt_len = len(req['prompt'])
        out = generator(req['prompt'], do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens=20, num_return_sequences=3, pad_token_id=50256)
        out0 = out[0]['generated_text'][prompt_len:].replace('\n', '\u21B2')
        out1 = out[1]['generated_text'][prompt_len:].replace('\n', '\u21B2')
        out2 = out[2]['generated_text'][prompt_len:].replace('\n', '\u21B2')
        return jsonify({"choice0":out0, "choice1":out1, "choice2":out2})

if __name__ == '__main__':
   app.run(debug=False, threaded=False)