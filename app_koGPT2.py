import torch
import torch.nn.functional as F
import traceback
import json
import werkzeug
werkzeug.cached_property = (
    werkzeug.utils.cached_property
)
from flask_restplus import Resource, Api, reqparse
from flask.helpers import make_response
from flask import Flask, jsonify, Response
from generator import inference, load_model

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app, version="1.0", title="KoGPT2 117M model", description="KoGPT2 117M model inference")
parser = reqparse.RequestParser()
parser.add_argument("greedy")
parser.add_argument("temperature")
parser.add_argument("top_p")
parser.add_argument("top_k")
parser.add_argument("text")
args={}

@api.route("/inference", endpoint="inference")
class Inference(Resource):

    @api.doc(
        responses={
            200: "OK",
            400: "Invalid Argument",
            500: "Internal Server Error",
        },
        params={
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
                args["search"] = 'greedy' if request['greedy'].lower() == 'true' else 'sampling'
                args["temperature"] = float(request['temperature'])
                args["top_p"] = float(request['top_p'])
                args["top_k"] = int(request['top_k'])
                text = request["text"].strip()

            except Exception:
                return Response('Invalid arguments in request.', status=400)

            if len(text) == 0:
                return Response("Input text is empty.", status=400)

            print(f'http request: {request}')
            print(f'text: "{text}", search: {args["search"]}, temperature: {args["temperature"]:.1f}, top_p: {args["top_p"]:.2f}, top_k: {args["top_k"]:d}')

            # result = generate_samples(model, text).strip()
            result = inference(model, vocab, tok, args["search"], temperature=args["temperature"], top_p=args["top_p"], top_k=args["top_k"], text=text, text_size=500)
            encoded_json = json.dumps(
                {'text': result},
                ensure_ascii=False)
            print(encoded_json)
            return make_response(encoded_json)
        except Exception as e:
            print(traceback.format_exc())
            return Response(e.__doc__, status=500)


if __name__ == "__main__":

    # model load
    model, vocab, tok = load_model()

    app.run(host="0.0.0.0", port=8080, threaded=False, debug=True)
