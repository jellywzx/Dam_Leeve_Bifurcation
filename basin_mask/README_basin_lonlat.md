# 标准四流域掩膜经纬度范围

本文记录当前正式标准产品中 `basin_mask > 0` 网格单元覆盖的外边界。范围包含半个网格间距，即表示网格单元边界，而不是最外层网格中心坐标。

基准目录：

```text
/share/home/dq134/wzx/openbench/postprocess/basin_mask
```

可用以下命令从当前 NetCDF 重新计算：

```bash
cd /share/home/dq134/wzx/openbench/postprocess/basin_mask
/share/home/dq134/.conda/envs/wzx/bin/python3 extract_lonlat.py --resolution all
```

也可将 `--resolution` 设为 `06min` 或 `15min` 仅检查一种分辨率。

## 06min 产品

| 流域 | 经度范围 | 纬度范围 | NetCDF |
|---|---:|---:|---|
| Amazon | -79.700° ～ -49.500° | -20.500° ～ 5.300° | `outputs/basin_masks_06min/amazon/amazon_basin.nc` |
| Missouri | -114.000° ～ -90.100° | 37.000° ～ 49.800° | `outputs/basin_masks_06min/missouri/missouri_basin.nc` |
| Lower Mississippi | -106.600° ～ -88.300° | 28.900° ～ 39.400° | `outputs/basin_masks_06min/lower_mississippi/lower_mississippi_basin.nc` |
| Mekong | 93.800° ～ 108.800° | 9.500° ～ 33.900° | `outputs/basin_masks_06min/mekong/mekong_basin.nc` |

06min 全局网格大小为 3600 × 1800，单元间距为 0.1°。

## 15min 产品

| 流域 | 经度范围 | 纬度范围 | NetCDF |
|---|---:|---:|---|
| Amazon | -79.750° ～ -49.500° | -20.500° ～ 5.500° | `outputs/basin_masks_15min/amazon/amazon_basin.nc` |
| Missouri | -114.000° ～ -90.000° | 37.000° ～ 49.750° | `outputs/basin_masks_15min/missouri/missouri_basin.nc` |
| Lower Mississippi | -106.750° ～ -88.250° | 28.750° ～ 39.500° | `outputs/basin_masks_15min/lower_mississippi/lower_mississippi_basin.nc` |
| Mekong | 93.750° ～ 109.000° | 9.500° ～ 34.000° | `outputs/basin_masks_15min/mekong/mekong_basin.nc` |

15min 全局网格大小为 1440 × 720，单元间距为 0.25°。

## 计算方法

对每个标准 NetCDF：

1. 选择 `basin_mask > 0` 的有效单元；
2. 读取有效单元中心的最小/最大经纬度；
3. 经度和纬度两端分别扩展半个网格间距。

等价计算为：

```python
active = ds["basin_mask"].where(ds["basin_mask"] > 0, drop=True)
dx = abs(float(ds["lon"].diff("lon").median()))
dy = abs(float(ds["lat"].diff("lat").median()))

west = float(active["lon"].min()) - dx / 2
east = float(active["lon"].max()) + dx / 2
south = float(active["lat"].min()) - dy / 2
north = float(active["lat"].max()) + dy / 2
```

这些范围来自当前 `outputs/basin_masks_06min` 和 `outputs/basin_masks_15min` 中的实际 NetCDF。重新生成掩膜或修改控制单元 ID 后，应重新计算并更新本表。
