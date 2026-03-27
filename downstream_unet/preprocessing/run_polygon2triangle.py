import os
import glob
import numpy as np
import torch
import geopandas as gpd
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mae_pretrain.src.datasets.geometry_polygon import PolyFourierConverter

def process_and_save(input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    engine = PolyFourierConverter(device='cpu') # 仅用作剖分工具
    
    file_list = []
    for d in input_dirs:
        file_list.extend(glob.glob(os.path.join(d, '**', '*.shp'), recursive=True))
        file_list.extend(glob.glob(os.path.join(d, '**', '*.geojson'), recursive=True))

    all_triangles = []
    
    # 全局统计变量
    global_total_polys = 0
    global_valid_polys = 0
    global_max_nodes = 0
    global_min_nodes = float('inf')
    
    for f in file_list:
        print(f"\n开始处理文件: {f}")
        try:
            gdf = gpd.read_file(f)
        except Exception as e:
            print(f"读取文件失败 {f}: {e}")
            continue
        
        # 单文件统计变量
        file_total_polys = 0
        file_valid_polys = 0
        file_max_nodes = 0
        file_min_nodes = float('inf')
        
        for geom in tqdm(gdf.geometry, desc="Triangulating"):
            if geom is None or geom.is_empty: continue
            
            # 展开 MultiPolygo
            if geom.geom_type == 'Polygon':
                polys = [geom]
            elif geom.geom_type == 'MultiPolygon':
                polys = list(geom.geoms)
            else:
                continue
                
            for poly in polys:
                file_total_polys += 1 # 准确记录拆解后的实际 Polygon 数目
                coords = np.array(poly.exterior.coords)
                
                # 记录 Polygon 原本的节点数 (减1是因为闭合多边形首尾点重复)
                num_nodes = len(coords) - 1 
                if num_nodes < 3: 
                    continue # 过滤无效的几何形状(无法构成面)
                
                # 独立 Bounding Box 归一化
                minx, miny = coords[:, 0].min(), coords[:, 1].min()
                maxx, maxy = coords[:, 0].max(), coords[:, 1].max()
                cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                max_range = max(maxx - minx, maxy - miny) / 2.0
                
                if max_range < 1e-6: continue # 过滤退化为点的多边形
                
                norm_coords = (coords - np.array([cx, cy])) / max_range
                
                # 三角剖分
                tris = engine.triangulate_polygon(norm_coords)
                num_tris = tris.shape[0]
                
                if num_tris > 0: # 如果成功剖分出至少一个三角形
                    file_valid_polys += 1
                    file_max_nodes = max(file_max_nodes, num_nodes)
                    file_min_nodes = min(file_min_nodes, num_nodes)
                    all_triangles.append(tris)

        # 更新全局统计
        global_total_polys += file_total_polys
        global_valid_polys += file_valid_polys
        if file_valid_polys > 0:
            global_max_nodes = max(global_max_nodes, file_max_nodes)
            global_min_nodes = min(global_min_nodes, file_min_nodes)
        
        # 打印单文件统计信息
        _print_min = file_min_nodes if file_min_nodes != float('inf') else 0
        print(f"[{os.path.basename(f)}] 统计信息:")
        print(f"  - 原始Polygon总数目: {file_total_polys}")
        print(f"  - 成功剖分Polygon数目: {file_valid_polys}")
        print(f"  - 最大节点数(Polygon原始顶点): {file_max_nodes}")
        print(f"  - 最小节点数(Polygon原始顶点): {_print_min}")

    # 打印合并后的全局统计信息
    _global_print_min = global_min_nodes if global_min_nodes != float('inf') else 0
    print("\n" + "="*45)
    print("所有文件合并后的全局统计信息:")
    print(f"  - 原始Polygon总数目: {global_total_polys}")
    print(f"  - 成功剖分Polygon总数目: {global_valid_polys}")
    print(f"  - 全局最大节点数(Polygon原始顶点): {global_max_nodes}")
    print(f"  - 全局最小节点数(Polygon原始顶点): {_global_print_min}")
    print("="*45 + "\n")
    
    # 保存为非冗余格式
    if all_triangles:
        output_path = os.path.join(output_dir, 'polygon_triangles_normalized.pt')
        torch.save(all_triangles, output_path)
        print(f"数据已成功保存至: {output_path}")
    else:
        print("未提取到任何有效的多边形，没有保存文件。")

if __name__ == "__main__":
    input_dirs = ["/mnt/ssd-data/Geo_dataset/City/hangzhou_osm/2025/",
    "/mnt/ssd-data/Geo_dataset/City/hangzhou_osm/2024/"] # 请在此配置数据文件夹
    output_dir = "../data/"
    process_and_save(input_dirs, output_dir)