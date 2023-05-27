def print_dict(d, prefix=""):
    for k, v in d.items():
        kk = prefix + k
        if isinstance(v, dict):
            print_dict(v, kk+"/")
        elif isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            print(kk, v)
        elif isinstance(v, list):
            print(kk, len(v))
        else:
            print(kk, v.shape)