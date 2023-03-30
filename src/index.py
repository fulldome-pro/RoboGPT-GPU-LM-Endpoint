from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig


def create_app():
    app = Flask(__name__)
    api = Api(app, version='1.0', title='Model API',
        description='Mopdel API')

    plain_input = api.model('PlainInput', {
        'text': fields.String(required=True, description='Plain text'),
    })

    generate_input = api.model('GenerateInput', {
        'instruction': fields.String(required=True, description='The instruction to evaluate'),
        'input': fields.String(required=True, description='The input to use in the evaluation')
    })

    translate_input = api.model('TranslateInput', {
        'source': fields.String(required=True, description='Source language'),
        'dest': fields.String(required=True, description='Destination language'),
        'input': fields.String(required=True, description='Text to translate')
    })
    
    
    print("🚀 Loading tokenizer...");
    tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    print("🚀 Loading model...");
    model = LLaMAForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        device_map="auto",
    )
    print("🚀 Loading tune model...");
    model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,
    )
    

    def generate_plain(text):
        return f"""{text}"""

    def generate_prompt(instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""

    def generate_translation_prompt(source, dest, input):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Please translate text from {source} language to {dest} language

    ### Input:
    {input}

    ### Response:"""

    @api.route('/plain')
    class Plain(Resource):
        @api.expect(plain_input, validate=True)
        def post(self):
            """Generate plain"""
            text = request.json.get('text')

            prompt = generate_plain(text)
            print(prompt);

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256
            )
            for s in generation_output.sequences:
                output = tokenizer.decode(s)
                response = output.split("### Response:")[1].strip()
            return jsonify(response=response)

    @api.route('/generate')
    class Generate(Resource):
        @api.expect(generate_input, validate=True)
        def post(self):
            """Generate with evaluates instruction and input and returns the result"""
            instruction = request.json.get('instruction')
            input = request.json.get('input')

            prompt = generate_prompt(instruction, input)
            print(prompt);

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256
            )
            for s in generation_output.sequences:
                output = tokenizer.decode(s)
                response = output.split("### Response:")[1].strip()
            return jsonify(response=response)


    @api.route('/translate')
    class Translate(Resource):
        @api.expect(translate_input, validate=True)
        def post(self):
            """Translate"""
            source = request.json.get('source')
            dest = request.json.get('dest')
            input = request.json.get('input')

            prompt = generate_translation_prompt(source, dest, input)
            print(prompt);

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256
            )
            for s in generation_output.sequences:
                output = tokenizer.decode(s)
                response = output.split("### Response:")[1].strip()
            print("response:");
            print(response);
            return jsonify(response=response)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0',debug=True, use_reloader=False)