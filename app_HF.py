import torch
import traceback
import werkzeug
import json
import os
werkzeug.cached_property = (
    werkzeug.utils.cached_property
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizerFast
from flask_restplus import Resource, Api, reqparse
from flask.helpers import make_response
from flask import Flask, jsonify, Response

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app, version="1.0", title="KoGPT2", description="KoGPT2 Model Inference")
parser = reqparse.RequestParser()
parser.add_argument("greedy")
parser.add_argument("temperature")
parser.add_argument("top_p")
parser.add_argument("top_k")
parser.add_argument("text")
args={}

@api.route("/inference", endpoint='inference')
class Inference(Resource):

    @api.doc(
        responses={
            200: "OK",
            400: "Invalid Argument",
            500: "Internal Server Error",
        },
        params = {
            'text': "Write text for context",
            'greedy': "Sampling method (Greedy or Temperature)",
            'temperature': "Sampling temperature (Temperature sampling)",
            'top_p': "Top p sampling (Temperature sampling)",
            'top_k': "Top k sampling (Temperature sampling)",
        }
    )
    def post(self):
        try:
            try:
                request = parser.parse_args()
                args["search_method"] = 'greedy' if request['greedy'].lower() == 'true' else 'sampling'
                args["temperature"] = float(request['temperature'])
                args["top_p"] = float(request['top_p'])
                args["top_k"] = int(request['top_k'])
                text = request["text"].strip()

                #other decoding options
                max_length=500
                search_method=None #consier

            except:
                return Response('Invalid arguments in request.', status=400)

            if len(text) == 0:
                return Response("Input text is empty.", status=400)

            print(f'http request: {request}')
            print(f'text: "{text}", search_method: {args["search_method"]}, temperature: {args["temperature"]:.1f}, top_p: {args["top_p"]:.2f}, top_k: {args["top_k"]:d}')

            input_ids = tokenizer.encode(text, return_tensors='pt')
            input_ids = input_ids.to(device)

            if args["search_method"] == 'greedy':
                greedy_output = model.generate(input_ids, max_length=max_length, early_stopping=True)
                output = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

            elif args["search_method"] == 'beam':
                beam_output = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
                output = tokenizer.decode(beam_output[0], skip_special_tokens=True)

            elif args["search_method"] == 'sampling':
                sample_output = model.generate(input_ids,
                                              do_sample=True,
                                              max_length=max_length,
                                              top_k=args["top_k"],
                                              top_p=args["top_p"],
                                              temperature=args["temperature"],
                                              early_stopping=True
                                              #no_repeat_ngram_size=2
                                              #repetition_penalty=1.5,
                                              )
                output=tokenizer.decode(sample_output[0], skip_special_tokens=True)

            encoded_json = json.dumps(
                {'text': output},
                ensure_ascii=False)
            print(encoded_json)
            return make_response(encoded_json)

        except Exception as e:
            print(traceback.format_exc())
            return Response(e.__doc__, status=500)


if __name__ == '__main__':

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device :', device)

    # load tokenizer
    print('start to load a tokeinzer')
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("taeminlee/kogpt2")
    print('loaded the tokenizer')

    # load model
    print('start to load a model')
    # model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2", pad_token_id=tokenizer.eos_token_id)
    model.to(device)
    model.eval()
    print('loaded the model')

    # run server
    app.run(debug=True, host='0.0.0.0', port=8081)
