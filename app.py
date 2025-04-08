from flask import Flask, request, render_template
from gpt4all import GPT4All
import os
import re

app = Flask(__name__)

model_path = "models"
model_name = "ggml-gpt4all-j-v1.3-groovy.bin"

if not os.path.exists(os.path.join(model_path, model_name)):
    raise FileNotFoundError(f"Model file '{model_name}' not found in '{model_path}'.")

model = GPT4All(model_name, model_path=model_path, allow_download=False)

@app.route('/', methods=['GET', 'POST'])
def chat():
    response = None
    user_message = None

    if request.method == 'POST':
        raw_message = request.form['message']
        user_message = re.sub(r'[^\x00-\x7F]+', '', raw_message).strip()

        if not user_message or len(user_message) < 2:
            response = "Please enter a valid message."
        else:
            try:
                prompt = f"""
You are a helpful assistant.

User: {user_message}
AI:"""
                response = model.generate(prompt).strip()
            except Exception as e:
                print("Error during model response:", e)
                response = "Oops! Something went wrong. Please try again."

    return render_template("index.html", response=response, user_message=user_message)

if __name__ == '__main__':
    app.run(debug=True)

