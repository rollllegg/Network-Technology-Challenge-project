import config as cf
from config import action, LogColor
from config import csv_dir, rules_dir
import os
import csv
import time
import json


if cf.MODE == "soft":
    from mode_b import Engine
    LogColor.info("mode b imported")
else:
    from mode_a import Engine
    LogColor.info("mode a imported")

def ReadLinks(csv_path):
    links = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            tmp = dict()
            for k, v in row.items():
                tmp[k] = v
            links.append(tmp)
    return links

def ReadRules(json_path):
    with open(json_path, 'r', newline='', encoding='utf-8') as f:
        data = json.load(f)

        tmp_action = action.NOP
        for rule in data['rules']:
            if rule['action'] == 'del':
                tmp_action = action.DEL
            elif rule['action'] == 'add':
                tmp_action = action.ADD
            elif rule['action'] == 'replace':
                tmp_action = action.REPLACE
            
            rule['action'] = tmp_action
        return data['meta'], data['rules']

def GetAllFiles(relative_path):
    """
    返回指定文件夹下的所有csv和json文件的绝对路径，按文件名中最后一个下划线后的数字排序

    :param relative_path: 文件夹的相对路径
    """
    # 获取绝对路径
    absolute_path = os.path.abspath(relative_path)
    
    # 检查路径是否存在
    if not os.path.exists(absolute_path):
        LogColor.error(f"路径 {absolute_path} 不存在")
        return
    
    # 检查是否为目录
    if not os.path.isdir(absolute_path):
        LogColor.error(f"路径 {absolute_path} 不是一个文件夹")
        return
    
    resp = []
    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(absolute_path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.json'):
                resp.append(os.path.join(root, file))
            else:
                LogColor.warning(f"文件 {file} 不是csv或json文件，已跳过")

    # 按文件名中最后一个下划线后的数字排序
    def extract_number(file_path):
        file_name : str = os.path.basename(file_path)
        parts = file_name.rsplit('_', 1)
        if len(parts) > 1 and parts[1].split('.')[0].isdigit():
            return int(parts[1].split('.')[0])
        return float('inf')  # 如果没有数字，则放在最后

    resp.sort(key=extract_number)
    return resp

def run():
    engine = Engine()
    timer = 0
    for csv_file, rules_file in zip(GetAllFiles(csv_dir), GetAllFiles(rules_dir)):
        LogColor.info(f"csv file: {csv_file}\nrules file: {rules_file}\n")
        links = ReadLinks(csv_file)
        meta, rules = ReadRules(rules_file)
        engine.AddContent('UAV_01', 'test.jpg', filesize=50)
        tmp_timer = 0
        rule_ind = 0
        req_ind = 0
        edge_ind = 0
        reqs = [
        ]

        try:
            while tmp_timer < 60000:
                LogColor.info(f'time : {timer}')
                while edge_ind < len(links) and int(links[edge_ind]['time_ms']) <= timer:
                    engine.addLink(links[edge_ind])
                    LogColor.debug(f'edge {edge_ind} applied')
                    edge_ind += 1

                while rule_ind < len(rules) and rules[rule_ind]['time_ms'] <= timer:
                    engine.UpdateRule(rules[rule_ind], meta)
                    LogColor.debug(f'rule {rule_ind} applied')
                    rule_ind += 1
                while req_ind < len(reqs) and reqs[req_ind]['time']  <= timer:
                    req = reqs[req_ind]
                    engine.ExecuteReq(req['node_id'], req['content_id'], timer, 'output/networks.csv')
                    req_ind += 1
                timer += 100
                tmp_timer += 100
                time.sleep(0.1)
        except KeyboardInterrupt:
            return

if __name__ == '__main__':
    run()