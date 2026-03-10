import csv
import networkx as nx
import os
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from config import action, LogColor

class Engine:
    def __init__(self):
        self.content = dict()
        self.hosts = dict()
        self.switches = dict()
        self.rules = dict()
        self.G = nx.DiGraph()   # 用于计算最短路
        self.net = Mininet(switch=OVSSwitch)
        self.net.start()
    
    def StopNet(self):
        self.net.stop()
    
    def __ensure_host(self, node):
        if node in self.hosts.keys():
            return self.net.get(node)
        h = self.net.addHost(node)
        self.hosts[node] = 0
        return h
    
    def __ensure_switch(self, sw):
        if sw in self.switches.keys():
            return self.net.get(sw)
        s = self.net.addSwitch(sw)
        self.switches[sw] = 0
        s.start([])
    
    def addLink(self, link):
        # 先在mininet中加入链路
        n1 : str = link['src']
        n2 : str = link['dst']
        # LogColor.debug(f'src: {n1}')
        # LogColor.debug(f'dst: {n2}')
        # 处理节点可能不存在的问题
        for n in (n1, n2):
            if n.startswith('GS'):
                self.__ensure_host(n)
                intf_name = n + '-eth' + str(self.hosts[n])
                self.hosts[n] += 1
            else:
                self.__ensure_switch(n)
                intf_name = n + '-eth' + str(self.switches[n])
                self.switches[n] += 1
            if n == n1:
                intf_name1 = intf_name
            else:
                intf_name2 = intf_name
                
        # 处理链路可能不存在的问题
        links = self.net.linksBetween(n1, n2)
        LogColor.info(f'links between {n1} and {n2} : {links}')
        if not links:
            lk = self.net.addLink(
                n1, n2,
                cls=TCLink,
                intfName1=intf_name1,
                intfName2=intf_name2,
                bw=int(link['bw_mbps']),
                delay=float(link['delay_ms']),
                jitter=float(link['jitter_ms']),
                loss=int(float(link['loss_pct'])),
                use_htb=True
            )
            # 启用接口
            lk.intf1.ifconfig('up')
            lk.intf2.ifconfig('up')

            if link['direction'] == 'UNIDIR':
                dst_intf = lk.intf2.name
                self.net.get(n2).cmd(f'tc qdisc add dev {dst_intf} root netem loss 100%')

        else:
            lk = links[0]
            for intf in (lk.intf1, lk.intf2):
                intf.config(
                    bw=link['bw_mbps'],
                    delay=link['delay_ms'],
                    jitter=link['jitter_ms'],
                    loss=link['loss_pct']
                )
            if link['direction'] == 'UNIDIR':
                dst_intf = lk.intf2.name
                self.net.get(n2).cmd(f'tc qdisc add dev {dst_intf} root netem loss 100%')
        
        # 再在模拟链路中加
        edge_attr = {k : v for k, v in link.items() if k != 'direction' and k != 'src' and k != 'dst'}
        self.G.add_node(link['src'])
        self.G.add_node(link['dst'])

        if link['direction'] == 'BIDIR':
            self.G.add_edge(link['src'], link['dst'], **edge_attr)
            self.G.add_edge(link['dst'], link['src'], **edge_attr)
        elif link['direction'] == 'UNIDIR':
            self.G.add_edge(link['src'], link['dst'], **edge_attr)
        else:
            raise RuntimeError('wrong direction type')

    def PrintGraph(self):
        for u, v, data in self.G.edges(data=True):
            LogColor.info(f'{u} -> {v} {data}')

    def UpdateRule(self, rule, meta):
        self.version = meta['version']

        if rule['action'] == action.REPLACE and rule['node'] not in self.rules.keys():
            rule['action'] = action.ADD
        if rule['action'] == action.ADD:
            if rule['node'] in self.rules.keys():
                raise RuntimeError('invaild rule : add existed rule')
            self.rules[rule['node']] = {k: v for k, v in rule.items() if k != "node"}
        elif rule['action'] == action.REPLACE:
            cur_node = rule['node']
            for k, v in self.rules[cur_node].items():
                self.rules[cur_node][k] = rule[k]
        elif rule['action'] == action.DEL:
            if rule['node'] in self.rules.keys():
                self.rules.pop(rule['node'])
        
    def AddContent(self, target, filename, **fileinfo):
        if target in self.content.keys():
            self.content[target][filename] = fileinfo 
        else:
            self.content[target] = {filename : fileinfo }
    
    def DeleteContent(self, target, filename):
        if target in self.content.keys():
            self.content[target].pop(filename)

    def UpdateContent(self, target, filename, **fileinfo):
        self.AddContent(target, filename, **fileinfo)

    def GetContent(self, target, filename):
        if target in self.content.keys() and (filename in self.content[target].keys()):
            return self.content[target][filename]
        return None


    def ExecuteReq(self, client, content_id, time, log_path):
        if client not in self.rules.keys():
            raise RuntimeError('invalid request: no such client')
        
        target = self.rules[client]['next_hop']
        algo = self.rules[client]['algo']
        cache_status = 'HIT'
        http_code = 200

        if (target in self.content.keys()) and (content_id in self.content[target].keys()):
            def edge_cost(u, v, data):
                if data['status'] != 'UP':
                    return float('inf')
                return data['delay_ms']
            
            path = nx.shortest_path(
                self.G,
                source=client,
                target=target,
                weight=edge_cost
            )

            content_info = self.content[target][content_id]
            file_size_MB = content_info['filesize']
            
            # TODO 利用mininet计算真实的延迟与下载速度
            tmp = dict()
            self.WriteLog(tmp, log_path)
        else:
            raise RuntimeError('invalid request: no such content')


    def WriteLog(self, row, csv_path):
        if len(row) == 0:
            return
        file_exists = os.path.exists(csv_path)
        write_header = True

        if file_exists:
            # 文件存在但大小为 0，仍然要写表头
            write_header = os.path.getsize(csv_path) == 0

        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if write_header:
                writer.writeheader()

            writer.writerow(row)