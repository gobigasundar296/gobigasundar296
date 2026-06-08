from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = 'out_model'

def main():
    print('Loading config from', MODEL_DIR)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    print('config keys:')
    for k in ('model_type','n_layer','n_head','n_inner','hidden_size','vocab_size','output_attentions'):
        print(' ', k, getattr(cfg, k, None))

    print('\nReloading config with output_attentions=True')
    cfg2 = AutoConfig.from_pretrained(MODEL_DIR, output_attentions=True)
    print('cfg2.output_attentions =', cfg2.output_attentions)

    print('\nLoading model with cfg2...')
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, config=cfg2)
    print('model.config.output_attentions =', getattr(model.config, 'output_attentions', None))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    text = 'This is a short document about cooking pasta.'
    enc = tokenizer(text, return_tensors='pt')

    print('\nRunning forward with output_attentions=True...')
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    att = getattr(out, 'attentions', None)
    print('attentions is None?', att is None)
    print('attentions type:', type(att))
    try:
        print('len(attentions) =', len(att))
    except Exception as e:
        print('Could not len(attentions):', e)
    if att:
        for i, a in enumerate(att):
            print(f' layer {i} shape:', tuple(a.shape))

if __name__ == '__main__':
    main()
