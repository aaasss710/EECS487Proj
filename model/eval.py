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
            outputs = model(batch)
            pooler_output = outputs
        return pooler_output.cpu()
    
    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)
    
    tasks = ['STSBenchmark', 'SICKRelatedness', 'STS12']
    model.eval()
    results = se.eval(tasks)
    
    sts12_spearman = results['STS12']['all']['spearman']['all']
    sts13_spearman = results['STS13']['all']['spearman']['all']
    sts14_spearman = results['STS14']['all']['spearman']['all']
    sts15_spearman = results['STS15']['all']['spearman']['all']
    sts16_spearman = results['STS16']['all']['spearman']['all']
    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    metrics = {"eval_stsb_spearman": stsb_spearman,
               'STS12': sts12_spearman,
               'STS13': sts13_spearman,
               'STS14': sts14_spearman,
               'STS15': sts15_spearman,
               'STS16': sts16_spearman,
               "eval_avg_sts": (stsb_spearman + sts12_spearman + sts13_spearman + sts14_spearman + sts15_spearman + sts16_spearman) / 6
            }


    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    wiki_model = CustomRobertaModel()
    wiki_model.load_state_dict(torch.load('real_simcse_1wiki_model.pth', map_location=device))
    metrics = evaluate(wiki_model, tokenizer, device)
    print(metrics)

if __name__ == "__main__":
    main()