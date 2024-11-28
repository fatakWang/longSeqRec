from .lru import LRU

MODELS = {
    # todo
    
}

def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)