import torch
import torch.nn.functional as F


def top_k_logits(logits, k):
    """Top k sampling"""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Top p sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def greedy_sequence(model, tokenizer, vocab, sent, text_size, max_seq=1024, eos='</s>'):
    device = 'cuda'

    tokenized_sent = tokenizer(sent)
    count = 0
    generated_text = ''

    if len(tokenized_sent) > max_seq:
        return 0

    while True:
        input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[tokenized_sent]).unsqueeze(0)
        input_ids = input_ids.to(device)

        model = model.to(device)
        pred = model(input_ids)[0]
        gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]

        if gen == eos or count > 100:
            break

        sent += gen.replace('▁', ' ')
        generated_text += gen.replace('▁', ' ')
        tokenized_sent = tokenizer(sent)
        count += 1

    return sent


def sample_sequence(model, tokenizer, vocab, sent, text_size, temperature, top_p, top_k, max_seq=1024, eos='</s>'):
    device = 'cuda'

    tokenized_sent = tokenizer(sent)
    count = 0
    generated_text = ''

    if len(tokenized_sent) > max_seq:
        return 0

    while True:
        input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[tokenized_sent]).unsqueeze(0)

        input_ids = input_ids.to(device)
        model = model.to(device)

        predicts = model(input_ids)
        pred = predicts[0]

        # temperature
        logits = pred
        logits = logits[:, -1, :] / temperature
        # top k
        logits = top_k_logits(logits, top_k)
        # top p
        logits = top_p_logits(logits, top_p=top_p)

        log_probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)
        gen = vocab.to_tokens(prev.squeeze().tolist())

        if gen == eos or count > text_size:
            sent += gen.replace('▁', ' ')
            generated_text += gen.replace('▁', ' ')
            sent += '\n'
            generated_text += '\n'
            tokenized_sent = tokenizer(sent)
            count = 0
            break

        sent += gen.replace('▁', ' ')
        generated_text += gen.replace('▁', ' ')
        tokenized_sent = tokenizer(sent)
        count += 1

    return sent
