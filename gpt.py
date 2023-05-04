import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

@app.route('/api/gptqhu_beta', methods=['POST'])
def generate_text():
    prompt = request.json['message']
    max_length = request.json.get('max_length', 100)
    temperature = request.json.get('temperature', 1.0)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = input_ids.clone()

    with torch.no_grad():
        for i in range(max_length):
            logits = model(output)
            if output.numel() == 0:
                break
            next_token_logits = logits[0][0, -1, :] / temperature
            next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            output = torch.cat([output, next_token_id.unsqueeze(0)], dim=-1)
            if next_token_id == tokenizer.eos_token_id:
                break

    response = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
    return jsonify({'response': response[len(prompt):]})

if __name__ == '__main__':
    app.run(debug=True)
