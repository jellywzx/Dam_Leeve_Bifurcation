# CoLM 四流域掩膜制作

本目录是 OpenBench 后处理中的 CoLM 流域掩膜生产目录：

```text
/share/home/dq134/wzx/openbench/postprocess/basin_mask
```

正式流程读取 CoLM/CaMa unit-catchment routing NetCDF，生成 Amazon、Missouri、Lower Mississippi 和 Mekong 的标准掩膜，以及控制断面和增量分区产品。源 routing 文件始终以只读方式打开。

## 目录结构与职责

```text
basin_mask/
├── basin_mask_config.py              # 分辨率、routing 文件、输出目录和控制单元 ID
├── basin_mask_core.py                # 拓扑追踪、网格映射、校验和导出
├── basin_mask_plot.py                # 正式产品统一预览图
├── run_all_basin_masks_06min.py      # 06min 标准四流域入口
├── run_all_basin_masks_15min.py      # 15min 标准四流域入口
├── run_all_basin_parts_06min.py      # 06min 详细分区入口
├── run_all_basin_parts_15min.py      # 15min 详细分区入口
├── make_lower_mississippi_levee_effect_mask.py  # Lower Mississippi 专项分析
├── extract_lonlat.py                 # 统计当前标准产品的有效经纬度范围
├── tests/test_basin_mask_core.py     # 小型拓扑回归测试
├── trash/                            # 旧版原型和诊断脚本
└── outputs/                          # 正式结果和专项结果
```

`trash/trace_colm_upstream.py` 是独立的定位/诊断工具。四个 `run_all_basin_*` 正式入口不导入也不执行它；生产流程使用 `basin_mask_config.py` 中已经批准的单元 ID。

## 输入数据与配置

配置集中在 `basin_mask_config.py`：

| 分辨率 | routing 文件 | 网格大小 |
|---|---|---|
| 06min | `/share/home/dq013/zhwei/colm/data/CoLMruntime/unitcatchment/grid_routing_data_06min.nc` | 3600 × 1800 |
| 15min | `/share/home/dq013/zhwei/colm/data/CoLMruntime/unitcatchment/grid_routing_data_15min_sed.nc` | 1440 × 720 |

如需调整源文件、输出目录或控制单元 ID，只修改 `basin_mask_config.py`，不要在四个入口脚本中重复配置。

## 标准四流域产品

所有生产脚本应在计算节点运行。以 `node111` 为例：

```bash
ssh dq134@node111
cd /share/home/dq134/wzx/openbench/postprocess/basin_mask

MPLBACKEND=Agg /share/home/dq134/.conda/envs/wzx/bin/python3 run_all_basin_masks_06min.py
MPLBACKEND=Agg /share/home/dq134/.conda/envs/wzx/bin/python3 run_all_basin_masks_15min.py
```

每个入口生成四套产品：

1. `amazon`：Amazon 主河口的完整上游集合；
2. `missouri`：Missouri 汇入 Mississippi 前控制单元的上游集合；
3. `lower_mississippi`：Mississippi 主河口上游减去 Cairo 汇流点下游边界的上游集合，表示 Cairo 到河口的增量汇水区，不等同于 USGS HUC08；
4. `mekong`：Mekong 主网络河口的完整上游集合。

输出根目录为：

```text
outputs/basin_masks_06min/
outputs/basin_masks_15min/
```

每个流域使用独立子目录，例如：

```text
outputs/basin_masks_06min/amazon/
├── amazon_basin.nc
├── amazon_basin_mask.tif
├── amazon_preview.png
├── amazon_summary.json
└── amazon_upstream_units.csv
```

根目录中的 `manifest.json` 汇总本次运行的分辨率、routing 文件、产品定义、控制单元 ID、单元数量和 NetCDF 路径。GeoTIFF 仅在环境安装 `rasterio` 时生成。

## 详细分区产品

需要控制断面和增量分区时运行：

```bash
cd /share/home/dq134/wzx/openbench/postprocess/basin_mask

MPLBACKEND=Agg /share/home/dq134/.conda/envs/wzx/bin/python3 run_all_basin_parts_06min.py
MPLBACKEND=Agg /share/home/dq134/.conda/envs/wzx/bin/python3 run_all_basin_parts_15min.py
```

输出目录为：

```text
outputs/basin_parts_06min/
outputs/basin_parts_15min/
```

每个入口固定生成 10 套产品：

- Amazon：`amazon_mainstem_mouth`；
- Missouri：`missouri_hermann`、`missouri_before_confluence`；
- Mississippi：`cairo_below_confluence`、`mississippi_vicksburg`、`mississippi_main_mouth`；
- Mississippi 增量段：`mississippi_cairo_to_vicksburg_added_catchments`、`mississippi_cairo_to_mouth_added_catchments`；
- Mekong：`mekong_stungtreng`、`mekong_main_mouth`。

两个 Mississippi 增量产品是 CoLM `seq_next` 上游集合之差，不等同于 USGS HUC08。Amazon 暂不按 Óbidos 候选点切分。

## 校验与依赖

生产流程会检查：

- routing 网格尺寸及一维变量长度；
- `topo_uparea` 和 `inpmat_area` 单位；
- 控制单元 ID 范围及主河口 `seq_next=-9`；
- 河网拓扑环路和反向追踪计数；
- 增量分区所需的集合包含关系；
- 运行前后源 routing 文件大小和修改时间是否保持不变。

快速回归测试：

```bash
cd /share/home/dq134/wzx/openbench/postprocess/basin_mask
/share/home/dq134/.conda/envs/wzx/bin/python3 -m unittest tests/test_basin_mask_core.py
```

主要依赖为 `numpy`、`xarray` 和可用的 NetCDF 后端；`matplotlib` 用于预览图，`rasterio` 用于可选 GeoTIFF。

## 路径迁移提示

当前正式目录已经从旧的 `/share/home/dq134/wzx/openbench/basin_mask` 移至：

```text
/share/home/dq134/wzx/openbench/postprocess/basin_mask
```

旧运行生成的 `manifest.json` 或 NetCDF 属性可能仍保存旧绝对路径；重新运行对应入口后会按当前配置写入新路径。实际文件位置应以本目录下的 `outputs/` 为准。

标准四流域的当前经纬度范围和 NetCDF 路径见 `README_basin_lonlat.md`。
