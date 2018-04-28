from pyrouge import Rouge155
from trainer import encode_inputs
import torch
from torch.autograd import Variable
from torchtext import data
from tqdm import tqdm
import json
decode_path = "model_summary"
ref_path = "reference_summary"
decoder_hidden_size = 400
DEVICE=0

def rouge_eval(decode_path, ref_path):
    r = Rouge155()
    r.model_dir = decode_path
    r.system_dir = ref_path
    r.system_filename_pattern = '(\d+)_reference.txt'
    r.model_filename_pattern = '#ID#_decoded.txt'
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def decode_outputs(decoder, prev_hidden, encoder_hidden_states, batch_summ):
    target_length = batch_summ.size(0)
    batch_size = batch_summ.size(1)
    curr_tok = batch_summ[0]
    decoder.init_hidden(prev_hidden)
    decoder_out, hidden = None, None
    decoder_hidden_states = Variable(torch.zeros(target_length - 1, batch_size, decoder_hidden_size)).cuda(device=DEVICE)
    for w in range(1, target_length):  # Iteration skips the SOS token at w=0
        decoder_hidden_so_far = None if w == 0 else decoder_hidden_states[:w - 1]  # Empty on first pass
        decoder_out, hidden = decoder(curr_tok, encoder_hidden_states, decoder_hidden_so_far)
        decoder_hidden_states[w - 1] = hidden[0]
        _, top_ind = decoder_out.data.topk(1)
        curr_tok = batch_summ[w]
    return decoder_out, hidden


def evaluate(encoder, decoder, dataset, rev_field):
    dataset.init_epoch()
    count_art = 0
    for batch in dataset:
        enc_out, enc_hidden, encoder_hidden_states = encode_inputs(encoder, batch.article)
        dec_out, dec_hidden = decode_outputs(decoder, enc_hidden, encoder_hidden_states, batch.summary)
        rev_batch = rev_field.reverse(dec_out)  # Reverse tokenize
        for summ in rev_batch:  # Write to file
            with open("model_summary/{}_decoded.txt".format(count_art), "w") as f:
                f.write(summ)
            count_art += 1
    # Evaluate with rouge
    r_dict = rouge_eval(decode_path, ref_path)
    with open("results.txt", 'w') as f:
        f.write(json.dumps(r_dict))
