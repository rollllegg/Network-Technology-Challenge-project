import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import json
import math
import networkx as nx
import glob
import os

# --- 全局配置 ---
MAX_LINK_RANGE = 5000 * 1000  # 5000km
MIN_ELEVATION_DEG = 10.0      
SPEED_OF_LIGHT = 3e8          

# 模拟业务配置 (根据真实节点ID修改)
# 假设 UAV_01 有地图，某个卫星有视频
# 注意：这里的 Key 必须和 CSV 里的 node_id 一致
CONTENT_LOCATIONS = {
    "UAV_01": ["map.tif"],       
    "SAT_63188": ["video.ts"]
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ---------------------------------------------------------
# 模块 0: 数据加载与融合 (Data Fusion)
# ---------------------------------------------------------
def load_and_merge_traces(sat_dir='sat_trace', uav_dir='uav_trace'):
    print(">>> Loading Trace Data...")
    
    # 1. 读取所有卫星数据
    sat_files = glob.glob(os.path.join(sat_dir, "*.csv"))
    sat_dfs = []
    for f in sat_files:
        print(f"  - Reading {f}")
        df = pd.read_csv(f)
        # 统一列名: 只要 id, type, x, y, z, ip, time
        # 卫星 CSV: node_id, type, ecef_x, ecef_y, ecef_z, ip, time_ms
        sat_dfs.append(df)
    
    df_sat = pd.concat(sat_dfs) if sat_dfs else pd.DataFrame()
    
    # 2. 读取所有 UAV/地面站数据
    uav_files = glob.glob(os.path.join(uav_dir, "*.csv"))
    uav_dfs = []
    for f in uav_files:
        print(f"  - Reading {f}")
        df = pd.read_csv(f)
        # UAV CSV: node_id, type, ecef_x, ecef_y, ecef_z, ip, time_ms
        uav_dfs.append(df)
        
    df_uav = pd.concat(uav_dfs) if uav_dfs else pd.DataFrame()

    # 3. 关键：时间对齐
    # 卫星数据是 1000ms 一次，UAV 是 100ms 一次
    # 我们以 UAV 的高频时间轴为准，卫星位置在 1 秒内保持不变 (向前填充)
    
    # 获取所有唯一的时间点 (以 100ms 为单位)
    all_timestamps = sorted(df_uav['time_ms'].unique())
    print(f">>> Total Time Steps: {len(all_timestamps)} (from {min(all_timestamps)} to {max(all_timestamps)} ms)")
    
    # 建立索引以加速查询
    # 我们不直接合并成一个巨大的 DataFrame (内存会爆)，而是按需提取
    return df_sat, df_uav, all_timestamps

def get_nodes_at_timestamp(df_sat, df_uav, target_time_ms):
    """
    获取指定时刻的所有节点数据。
    对于卫星：找到最近的一个 <= target_time_ms 的记录
    对于 UAV：找到精确匹配的记录
    """
    # UAV: 精确匹配
    uav_current = df_uav[df_uav['time_ms'] == target_time_ms]
    
    # SAT: 找到所属的整秒 (例如 1200ms -> 1000ms)
    sat_time_key = (target_time_ms // 1000) * 1000
    sat_current = df_sat[df_sat['time_ms'] == sat_time_key]
    
    # 合并
    # 统一需要的列
    cols = ['node_id', 'type', 'ecef_x', 'ecef_y', 'ecef_z', 'ip']
    
    # 容错：防止某些时刻数据缺失
    if sat_current.empty and uav_current.empty:
        return pd.DataFrame(columns=cols)
        
    nodes = pd.concat([
        sat_current[cols], 
        uav_current[cols]
    ], ignore_index=True)
    
    return nodes

# ---------------------------------------------------------
# 模块 1 & 2: 物理层计算 (保持逻辑不变，适配数据)
# ---------------------------------------------------------
def calculate_delay(dist_m): return (dist_m / SPEED_OF_LIGHT) * 1000
def calculate_jitter(delay_ms): return delay_ms * 0.1
def calculate_bandwidth(type_a, type_b):
    types = {type_a, type_b}
    if 'GS' in types and 'UAV' in types: return 54
    if 'UAV' in types and 'SAT' in types: return 20
    if 'SAT' in types and 'GS' in types: return 20
    if 'SAT' in types: return 100
    return 10
def calculate_bdp_queue(bw_mbps, delay_ms):
    queue = int((bw_mbps * 1e6) * (delay_ms * 2 * 1e-3) / 12000)
    return max(10, queue)

def calculate_elevation(pos_a, pos_b):
    # 简单的地心坐标系仰角近似计算
    # 假设 pos_a 是地面观察者
    # 向量 A (地心 -> 观察者), 向量 AB (观察者 -> 目标)
    vec_a = np.array(pos_a)
    vec_ab = np.array(pos_b) - np.array(pos_a)
    
    dist_a = np.linalg.norm(vec_a)
    dist_ab = np.linalg.norm(vec_ab)
    
    if dist_a == 0 or dist_ab == 0: return 90.0
    
    # 计算余弦角：cos(theta) = (A . AB) / (|A|*|AB|)
    # 仰角 = 90 - theta
    cos_theta = np.dot(vec_a, vec_ab) / (dist_a * dist_ab)
    # 限制范围防报错
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta_rad = np.arccos(cos_theta)
    
    elevation = 90 - math.degrees(theta_rad)
    return elevation

def compute_topology(nodes_df, time_ms):
    links = []
    if len(nodes_df) < 2: return links
    
    coords = nodes_df[['ecef_x', 'ecef_y', 'ecef_z']].values
    ids = nodes_df['node_id'].values
    types = nodes_df['type'].values
    
    tree = cKDTree(coords)
    # k=20: 真实数据密度可能较大，增加搜索范围
    dists, indices = tree.query(coords, k=20, distance_upper_bound=MAX_LINK_RANGE)
    
    processed_pairs = set()

    for i in range(len(ids)):
        # 只让 GS 和 UAV 主动去连别人，卫星之间暂不建链(节省计算)，除非需要
        # 或者为了全连接，保留双向遍历
        for j_idx, neighbor_idx in enumerate(indices[i]):
            if dists[i][j_idx] == float('inf') or i == neighbor_idx: continue
            
            # 按字母序排序 key，保证无向图去重
            n1_id, n2_id = ids[i], ids[neighbor_idx]
            pair_key = tuple(sorted([n1_id, n2_id]))
            
            if pair_key in processed_pairs: continue
            
            type_a, type_b = types[i], types[neighbor_idx]
            
            # 策略：不建立 卫星-卫星 链路 (除非你需要做星间路由仿真)
            # 为了减少输出量，我们只关心 GS/UAV 的连接
            # if type_a == 'SAT' and type_b == 'SAT': continue 

            # 仰角检查 (针对 SAT - GS/UAV)
            is_sat_a = (type_a == 'SAT')
            is_sat_b = (type_b == 'SAT')
            
            if is_sat_a != is_sat_b: # 这是一个跨层链路
                sat_idx = i if is_sat_a else neighbor_idx
                gnd_idx = neighbor_idx if is_sat_a else i
                
                elev = calculate_elevation(coords[gnd_idx], coords[sat_idx])
                if elev < MIN_ELEVATION_DEG: continue

            # 生成链路
            dist_m = dists[i][j_idx]
            delay = calculate_delay(dist_m)
            bw = calculate_bandwidth(type_a, type_b)
            
            links.append({
                'time_ms': time_ms,
                'src': n1_id,
                'dst': n2_id,
                'direction': 'BIDIR',
                'distance_km': round(dist_m / 1000, 3),
                'delay_ms': round(delay, 2),
                'jitter_ms': round(calculate_jitter(delay), 3),
                'loss_pct': 0.0,
                'bw_mbps': bw,
                'max_queue_pkt': calculate_bdp_queue(bw, delay),
                'type': f"{type_a}-{type_b}",
                'status': 'UP'
            })
            processed_pairs.add(pair_key)
    return links

# ---------------------------------------------------------
# 模块 3: 路由策略 (适配真实 IP)
# ---------------------------------------------------------
def build_graph(links):
    G = nx.Graph()
    for l in links:
        G.add_edge(l['src'], l['dst'], weight=l['delay_ms'])
    return G

def find_route(G, src_id, content, mode, node_ip_map):
    # 1. 确定候选者
    candidates = []
    if mode == 'Content-Aware':
        for node, files in CONTENT_LOCATIONS.items():
            if content in files and node in G.nodes:
                candidates.append(node)
    
    # 默认回源：找任意一个可见的卫星
    if not candidates or mode == 'Greedy':
        # 在真实数据中，SAT ID 是动态的，不能写死 'SAT_02'
        # 我们寻找图中所有 type='SAT' 的节点
        # 这里需要从 ID 命名规则推断，你的 ID 是 SAT_xxxxx
        sat_candidates = [n for n in G.nodes if str(n).startswith('SAT_')]
        if sat_candidates:
            # Greedy 策略：找延迟最小的那个卫星
            candidates = sat_candidates
    
    if not candidates: return None, None, None

    # 2. 找最短路
    best_target = None
    best_path = None
    min_cost = float('inf')

    for target in candidates:
        try:
            cost = nx.shortest_path_length(G, src_id, target, weight='weight')
            if cost < min_cost:
                min_cost = cost
                best_path = nx.shortest_path(G, src_id, target, weight='weight')
                best_target = target
        except: continue
        
    # 3. 提取结果
    if best_path and len(best_path) > 1:
        nh_id = best_path[1]
        # 从全局 IP 表查 IP
        nh_ip = node_ip_map.get(nh_id, "0.0.0.0")
        return nh_id, nh_ip, best_target
        
    return None, None, None

def main():
    # 0. 准备输出目录
    output_link_dir = 'output/links'
    output_rule_dir = 'output/rules'
    os.makedirs(output_link_dir, exist_ok=True)
    os.makedirs(output_rule_dir, exist_ok=True)

    # 1. 加载数据
    df_sat, df_uav, timelines = load_and_merge_traces()
    
    # 2. 构建全局 IP 映射表
    print(">>> Building IP Map...")
    node_ip_map = {}
    for _, row in df_sat[['node_id', 'ip']].drop_duplicates().iterrows():
        node_ip_map[row['node_id']] = row['ip']
    for _, row in df_uav[['node_id', 'ip']].drop_duplicates().iterrows():
        node_ip_map[row['node_id']] = row['ip']
    print(f"   Mapped {len(node_ip_map)} nodes.")

    # 3. 分片配置
    CHUNK_SIZE_MS = 60000 # 60秒一个文件
    
    # 临时缓存当前分片的数据
    chunk_links = []
    chunk_rules = []
    current_chunk_idx = 0
    
    print(f">>> Start Processing {len(timelines)} time steps...")

    # 4. 主循环 (跑全量数据，去掉切片)
    for i, t in enumerate(timelines): 
        t_val = int(t)
        
        # --- A. 计算当前时刻数据 ---
        current_nodes = get_nodes_at_timestamp(df_sat, df_uav, t_val)
        
        # 计算拓扑
        links = compute_topology(current_nodes, t_val)
        chunk_links.extend(links)
        
        # 计算路由 (GS_01 -> map.tif)
        if 'GS_01' in current_nodes['node_id'].values:
            G = build_graph(links)
            nh, nh_ip, target = find_route(G, 'GS_01', 'map.tif', 'Content-Aware', node_ip_map)
            if nh:
                chunk_rules.append({
                    "time_ms": t_val,
                    "node": "GS_01",
                    "dst_cidr": "10.0.0.254/32",
                    "action": "replace",
                    "next_hop": nh,
                    "next_hop_ip": nh_ip,
                    "algo": "Content-Aware-CGR",
                    "debug_info": f"Target: {target}"
                })
        
        # 进度日志
        if i % 100 == 0:
            print(f"   [Progress] Step {i}/{len(timelines)} (Time {t_val}ms)")

        # --- B. 检查是否需要保存分片 ---
        # 如果当前是该分片的最后一帧，或者总数据的最后一帧
        is_last_step = (i == len(timelines) - 1)
        # 判断是否跨越了 60秒 的边界
        # 逻辑：如果下一个时间点属于下一个 60秒 区间，则保存当前
        next_t = timelines[i+1] if not is_last_step else -1
        
        if is_last_step or (int(next_t / CHUNK_SIZE_MS) > int(t_val / CHUNK_SIZE_MS)):
            # 生成文件名
            # 例如: topology_links_0_59900.csv
            start_ms = current_chunk_idx * CHUNK_SIZE_MS
            # 实际上结束时间是当前帧的时间
            end_ms = t_val 
            
            link_filename = f"topology_links_{start_ms}_{end_ms}.csv"
            rule_filename = f"routing_rules_{start_ms}_{end_ms}.json"
            
            # 保存链路
            if chunk_links:
                df_links = pd.DataFrame(chunk_links)
                cols = ['time_ms', 'src', 'dst', 'direction', 'distance_km', 'delay_ms', 
                        'jitter_ms', 'loss_pct', 'bw_mbps', 'max_queue_pkt', 'type', 'status']
                # 确保列存在，防止空数据报错
                for c in cols:
                    if c not in df_links.columns: df_links[c] = None
                df_links[cols].to_csv(os.path.join(output_link_dir, link_filename), index=False)
            
            # 保存规则
            rule_data = {"meta": {"version": "v1", "chunk_id": current_chunk_idx}, "rules": chunk_rules}
            with open(os.path.join(output_rule_dir, rule_filename), 'w') as f:
                json.dump(rule_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"   >>> [Saved Chunk {current_chunk_idx}] {link_filename}")
            
            # 重置缓存
            chunk_links = []
            chunk_rules = []
            current_chunk_idx += 1

    print(">>> All Done.")

if __name__ == "__main__":
    main()