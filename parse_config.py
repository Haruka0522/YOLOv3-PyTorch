def parse_model_config(path):
    """
    モデルのconfigファイルを読み込む関数
    cfgファイルのパスを渡すと解析して全てのブロックをdictとして保存する
    pathの例は"cfg/yolov3.cfg"
    """
    cfg_file = open(path, "r")
    lines = cfg_file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]  # 空の行を無視する
    lines = [x for x in lines if x[0] != "#"]  # コメントアウトを無視する
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

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
