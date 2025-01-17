from pathlib import Path

def get_config():
    return {
        "batch_size" : 64,
        "d_model" : 512,
        "num_epochs" : 10,
        "lr" : 1e-4,
        "seq_len" : 512,
        "lang_src" : "en",
        "lang_tgt" : "it",
        "model_folder" : "weights",
        "model_basename" : "tmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel_"
    }
    
def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)