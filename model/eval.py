import sys
import torch
from transformers import RobertaTokenizer
from RoBERTa import CustomRobertaModel
# Set path to SentEval
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def evaluate(model, tokenizer, device):
    
    def prepare(params, samples):
        return
    
    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
            pooler_output = outputs.pooler_output
        return pooler_output.cpu()
    
    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)
    
    
    tasks = ['STSBenchmark', 'SICKRelatedness']
    model.eval()
    results = se.eval(tasks)

    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman,
                "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}

    return metrics


def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = CustomRobertaModel()
    evaluate(model, tokenizer, 'cuda')

if __name__ == "__main__":
    main()