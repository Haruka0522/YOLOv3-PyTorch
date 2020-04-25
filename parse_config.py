def parse_model_config(cfgpath):
    """
    モデルのconfigファイルを読み込む関数
    cfgファイルのパスを渡すと解析して全てのブロックをdictとして保存する
    pathの例は"cfg/yolov3.cfg"
    """
    cfgfile = open(cfgpath, 'r')
    lines = cfgfile.read().split('\n')
    # コメントアウトされているものを除く
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """
    データのconfigファイルを読み込む関数
    cfgファイルのパスを渡すと解析して全てのブロックをdictとして保存する
    pathの例は"data/coco.names"
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
