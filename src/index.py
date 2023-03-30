from flask import Flask
from flask_restx import Api, Resource, fields
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig


def create_app():
    app = Flask(__name__)
    api = Api(app, version='1.0', title='GPT API',
        description='GPT API')
    
    instruction_input = api.model('InstructionInput', {
        'instruction': fields.String(required=True, description='The instruction to evaluate'),
        'input': fields.String(required=True, description='The input to use in the evaluation')
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

    
    @api.route('/hello')
    class HelloWorld(Resource):
        def get(self):
            """Returns 'Hello, World!'"""
            return {'message': 'Hello, World!'}

    @api.route('/evaluate')
    class Evaluate(Resource):
        @api.expect(instruction_input, validate=True)
        def post(self):
            """Evaluates instruction and input and returns the result"""
            instruction = request.json.get('instruction')
            input = request.json.get('input')

            prompt = generate_prompt(instruction, input)
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

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, use_reloader=False)