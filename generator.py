import os
import torch
import gluonnlp
import argparse
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.model.sample import sample_sequence, greedy_sequence
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel


parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.7,
					help="temperature value")
parser.add_argument('--top_p', type=float, default=0.9,
					help="Top p sampling (Temperature sampling)")
parser.add_argument('--top_k', type=int, default=40,
					help="Top k sampling (Temperature sampling)")
parser.add_argument('--text_size', type=int, default=250,
					help="text size to generate")
parser.add_argument('--text', type=str, default="",
					help="sentence(text) to start")

args = parser.parse_args()


def load_model():
	ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cachedir = './.cache/'

	# Device 설정
	device = torch.device(ctx)

	config = GPT2Config.from_json_file('./model/pt_model/kogpt2_hf/config.json')
	kogpt2model = GPT2LMHeadModel.from_pretrained('./model/pt_model/kogpt2_hf/pytorch_model.bin', config=config)

	kogpt2model.eval()
	vocab_path = './tokenizer/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
	vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
															  mask_token=None,
															  sep_token=None,
														      cls_token=None,
															  unknown_token='<unk>',
															  padding_token='<pad>',
															  bos_token='<s>',
															  eos_token='</s>')

	# tok_path = get_tokenizer(cachedir)
	tok_path=vocab_path
	model, vocab = kogpt2model, vocab_b_obj
	tok = SentencepieceTokenizer(tok_path, 0, 0)

	return model, vocab, tok


def inference(model, vocab, tok, search, temperature=0.7, top_p=0.8, top_k=10, text="", text_size=1024):

	if text=="":
		text='2019 한해를 보내며,'
	assert len(tok(text)) <= 1022, 'sentence is too long'

	if search == 'greedy':
		generated_sentence = greedy_sequence(model, tok, vocab, text, text_size)
	elif search == 'sampling':
		generated_sentence = sample_sequence(model, tok, vocab, text, text_size, temperature, top_p, top_k)

	print('generated sentence :', generated_sentence)
	generated_sentence = generated_sentence.replace("</s>", " ")

	return generated_sentence


if __name__ == "__main__":
	# execute only if run as a script
	model, vocab, tok = load_model()
	inference(model, vocab, tok, search='greedy', temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, text="", text_size=args.text_size)
