# -*- coding: utf-8 -*-
"""
带参数因子分析模块，承载 ParameterizedFactorAnalyzer。
本模块已在 yinzifenxi1119_split.py 中被引用，用于生成带参数因子报告。
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from .fa_config import DEFAULT_DATA_FILES, FACTOR_COLUMNS, RETURN_COLUMN
from .fa_nonparam_analysis import DEFAULT_GROUP_COUNT, DEFAULT_USE_PEARSON
from .fa_param_helpers import _fa_score_parameterized_factors
from .fa_param_report import _fa_generate_parameterized_report
from .fa_stat_utils import (
    custom_spearman_corr,
    rolling_window_analysis,
    sample_sensitivity_analysis,
)


class ParameterizedFactorAnalyzer:
    """专门针对带参数因子的综合分析器"""
    
    def __init__(self, data, file_path=None):
        """初始化综合因子分析器"""
        self.data = data
        default_paths = DEFAULT_DATA_FILES if file_path is None else file_path
        if isinstance(default_paths, (list, tuple, set)):
            resolved_paths = [str(p) for p in default_paths if p]
        elif default_paths:
            resolved_paths = [str(default_paths)]
        else:
            resolved_paths = []
        self.file_paths = resolved_paths
        self.file_path = self.file_paths[0] if self.file_paths else None
        self.factors = list(FACTOR_COLUMNS)
        self.factor_list = self.factors  # 修复：添加factor_list属性
        self.return_col = RETURN_COLUMN
        self.sqrt_annualization_factor = np.sqrt(252)
        self.annualization_factor = 252
        
        # 确保数据有效
        if self.data is None or self.data.empty:
            print("错误: 没有有效数据")
            # 不返回值，让对象仍可被创建但处于无效状态
    
    def load_data(self):
        """从文件加载数据"""
        if self.data is not None:
            return True
        if not self.file_paths:
            print("数据加载失败: 未指定数据文件")
            return False

        loaded_frames = []
        for path in self.file_paths:
            if not path:
                continue
            normalized_path = os.path.abspath(path)
            if not os.path.exists(normalized_path):
                print(f"警告: 数据文件不存在 -> {normalized_path}")
                continue
            try:
                if normalized_path.lower().endswith('.csv'):
                    df = pd.read_csv(normalized_path, encoding='utf-8-sig')
                elif normalized_path.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(normalized_path)
                else:
                    print(f"警告: 不支持的文件格式 -> {normalized_path}")
                    continue
                loaded_frames.append(df)
                print(f"加载文件 {os.path.basename(normalized_path)}: {len(df)} 行")
            except Exception as e:
                print(f"数据加载失败 ({normalized_path}): {e}")

        if not loaded_frames:
            print("数据加载失败: 未能成功读取任何文件")
            return False

        if len(loaded_frames) == 1:
            self.data = loaded_frames[0]
        else:
            self.data = pd.concat(loaded_frames, ignore_index=True, sort=False)
        return True
# 已删除复杂的自适应年化计算方法，使用优化版本（第1085行）
# 优化版本特点：
# 1. 使用标准复利年化方法作为主要计算方式
# 2. 保留CAGR方法作为对比方法
# 3. 删除线性年化方法（忽视复利效应）
# 4. 增强数据特征分析和验证机制
    
    def preprocess_data(self):
        """预处理数据"""
        if self.data is None or self.data.empty:
            print("错误: 没有数据可处理")
            return False
        
        try:
            # 复制数据
            df = self.data.copy()
            
            # 统一处理所有因子列及收益列的字符串/百分比格式
            columns_to_normalize = list(dict.fromkeys(list(self.factors) + [self.return_col]))
            for col in columns_to_normalize:
                if col not in df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                try:
                    col_series = df[col].astype(str).str.strip()
                    has_percent = col_series.str.contains('%').any()
                    cleaned = col_series.str.replace('%', '', regex=False)
                    numeric_series = pd.to_numeric(cleaned, errors='coerce')
                    if has_percent:
                        numeric_series = numeric_series / 100
                        print(f"已自动将列 '{col}' 的百分比字符串转换为小数")
                    else:
                        print(f"已自动尝试将列 '{col}' 转换为数值类型")
                    df[col] = numeric_series
                except Exception as e:
                    print(f"转换列 '{col}' 时出错: {e}")
            
            # 处理收益率列
            if not pd.api.types.is_numeric_dtype(df[self.return_col]):
                try:
                    if df[self.return_col].dtype == 'object':
                        df[self.return_col] = df[self.return_col].str.replace('%', '')
                    df[self.return_col] = pd.to_numeric(df[self.return_col], errors='coerce')
                    print(f"收益率列 {self.return_col} 转换为数值型")
                except:
                    print(f"警告：无法将 {self.return_col} 转换为数值型")
            
            # 确保日期列正确处理
            if '信号日期' in df.columns:
                try:
                    df['信号日期'] = pd.to_datetime(df['信号日期'], errors='coerce')
                except:
                    print("警告：无法转换信号日期列")
            
            # 删除缺失值
            original_len = len(df)
            df = df.dropna(subset=[self.return_col] + self.factors)
            print(f"数据预处理完成，分析使用 {len(df)} 行有效数据 (删除了 {original_len - len(df)} 行缺失值)")
            
            self.processed_data = df
            return True
            
        except Exception as e:
            print(f"数据预处理失败: {e}")
            return False
    
    def calculate_comprehensive_metrics(self, factor_col):
        """计算综合指标"""
        df_clean = self.processed_data.dropna(subset=[factor_col, self.return_col])
        
        if len(df_clean) < 10:
            print(f"警告: 因子 {factor_col} 有效数据不足")
            return None
        
        try:
            # 计算分组收益（默认使用配置中的等分数量）
            group_count = DEFAULT_GROUP_COUNT
            df_clean['分组'] = pd.qcut(df_clean[factor_col], q=group_count, labels=False, duplicates='drop')
            
            # 计算每组的统计指标
            group_stats = []
            total_samples = len(df_clean)
            
            for group_id in range(group_count):
                group_data = df_clean[df_clean['分组'] == group_id]
                
                if len(group_data) == 0:
                    continue
                
                # 获取该组的因子值范围
                factor_values = group_data[factor_col]
                min_val = factor_values.min()
                max_val = factor_values.max()
                param_range = f"[{min_val:.3f}, {max_val:.3f}]"
                
                # 计算该组的收益统计
                returns = group_data[self.return_col]
                avg_return = returns.mean()
                return_std = returns.std()
                win_rate = (returns > 0).mean()
                
                # 计算最大回撤
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # 自适应年化指标计算
                try:
                    # 修复：传递正确的参数格式（字典而不是Series）
                    avg_returns_dict = {
                        '平均收益': avg_return,
                        '收益标准差': return_std
                    }
                    # 获取数据特征用于自适应年化计算
                    data_characteristics = {
                        'total_trades': len(group_data),
                        'actual_annual_trades': len(group_data),  # 使用组内数据作为估算
                        'avg_trade_interval': 30,  # 默认30天
                        'observation_period_years': len(group_data) / 252,  # 估算观测期
                        'holding_period_days': 2,  # 修正：持股天数应该是2
                        'trade_frequency_category': '低频'
                    }
                    # 已删除复杂的自适应年化计算，使用简单的252日年化
                    annualized_return = avg_return * 252
                    annualized_std = return_std * np.sqrt(252)
                    adaptive_results = {
                        'base_frequency': 252,
                        'main_annual_return': annualized_return,
                        'annual_std': annualized_std
                    }
                    
                    # 选择主选年化收益率
                    annualized_return = adaptive_results['main_annual_return']
                    annualized_std = adaptive_results['annual_std']
                    
                except Exception as e:
                    print(f"自适应年化计算失败，使用备用方法: {e}")
                    # 备用：使用传统252日年化
                    annualized_return = avg_return * 252
                    annualized_std = return_std * np.sqrt(252)
                
                group_stats.append({
                    '分组': group_id + 1,
                    '参数区间': param_range,
                    '平均收益': avg_return,
                    '收益标准差': return_std,
                    '胜率': win_rate,
                    '最大回撤': max_drawdown,
                    '年化收益率': annualized_return,
                    '年化收益标准差': annualized_std,
                    '样本数量': len(group_data)
                })
            
            if not group_stats:
                return None
            
            group_stats_df = pd.DataFrame(group_stats)
            
            # 计算年化夏普比率和索提诺比率
            sharpe_ratios = []
            sortino_ratios = []
            
            for _, row in group_stats_df.iterrows():
                if row['收益标准差'] > 0:
                    sharpe = row['年化收益率'] / row['收益标准差']
                    sharpe_ratios.append(sharpe)
                    
                    # 计算下行标准差和索提诺比率
                    group_data = df_clean[df_clean['分组'] == row['分组'] - 1][self.return_col]
                    downside_returns = group_data[group_data < 0]
                    if len(downside_returns) > 0:
                        try:
                            # 使用自适应年化计算系统的年化因子
                            downside_std_dict = {'下行收益标准差': downside_returns.std()}
                            # 获取数据特征用于自适应年化计算
                            data_characteristics = {
                                'total_trades': len(group_data),
                                'actual_annual_trades': len(group_data),  # 使用组内数据作为估算
                                'avg_trade_interval': 30,  # 默认30天
                                'observation_period_years': len(group_data) / 252,  # 估算观测期
                                'holding_period_days': 2,  # 修正为合理的持股天数
                                'trade_frequency_category': '低频'
                            }
                            # 选择年化方法
                            # 已删除复杂的自适应年化计算，使用简单的252日年化
                            sqrt_annualization_factor = np.sqrt(252)
                            downside_std = downside_returns.std() * sqrt_annualization_factor
                            sortino = row['年化收益率'] / downside_std if downside_std > 0 else 0
                        except Exception as e:
                            print(f"自适应年化计算失败，使用备用方法: {e}")
                            # 备用：使用传统252日年化
                            downside_std = downside_returns.std() * np.sqrt(252)
                            sortino = row['年化收益率'] / downside_std if downside_std > 0 else 0
                    else:
                        sortino = np.inf if row['年化收益率'] > 0 else 0
                    sortino_ratios.append(sortino)
                else:
                    sharpe_ratios.append(0)
                    sortino_ratios.append(0)
            
            group_stats_df['年化夏普比率'] = sharpe_ratios
            group_stats_df['年化索提诺比率'] = sortino_ratios
            
            # 计算多空收益（最高组 - 最低组）
            long_short_return = group_stats_df['年化收益率'].max() - group_stats_df['年化收益率'].min()
            
            return {
                'group_stats': group_stats_df,
                'long_short_return': long_short_return,
                'total_samples': total_samples,
                'factor_col': factor_col
            }
            
        except Exception as e:
            print(f"计算因子 {factor_col} 综合指标时出错: {e}")
            return None
    
    def calculate_ic(self, factor_col, use_pearson=None):
        """计算信息系数IC"""
        if use_pearson is None:
            use_pearson = DEFAULT_USE_PEARSON
        if not hasattr(self, 'processed_data'):
            df_clean = self.data.dropna(subset=[factor_col, self.return_col])
        else:
            df_clean = self.processed_data.dropna(subset=[factor_col, self.return_col])
        
        if len(df_clean) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        try:
            factor_values = df_clean[factor_col].values
            return_values = df_clean[self.return_col].values

            if use_pearson:
                ic = np.corrcoef(factor_values, return_values)[0, 1]
            else:
                ic = custom_spearman_corr(factor_values, return_values)
            
            # 计算IC的均值和标准差（使用滚动窗口）
            window_size = min(30, len(df_clean) // 3)
            if window_size < 5:
                return ic, np.nan, np.nan, np.nan
            
            rolling_ic = []
            for i in range(window_size, len(df_clean)):
                subset = df_clean.iloc[i-window_size:i]
                if use_pearson:
                    corr = subset[factor_col].corr(subset[self.return_col])
                else:
                    corr = custom_spearman_corr(
                        subset[factor_col].values,
                        subset[self.return_col].values
                    )
                if not np.isnan(corr):
                    rolling_ic.append(corr)
            
            if len(rolling_ic) < 2:
                return ic, np.nan, np.nan, np.nan
            
            ic_mean = np.mean(rolling_ic)
            ic_std = np.std(rolling_ic)
            
            # 计算t统计量和p值
            if ic_std > 0:
                t_stat = ic_mean / (ic_std / np.sqrt(len(rolling_ic)))
                try:
                    from scipy.stats import t
                    p_value = 2 * (1 - t.cdf(abs(t_stat), len(rolling_ic) - 1))
                except:
                    p_value = np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
            
            return ic_mean, ic_std, t_stat, p_value
        
        except Exception as e:
            print(f"计算IC时出错: {e}")
            return np.nan, np.nan, np.nan, np.nan
    
    def score_factors(self, factor_results):
        """对因子进行综合评分"""
        return _fa_score_parameterized_factors(factor_results)

    def generate_parameterized_report(self):
        """生成带参数因子的详细TXT/CSV报告"""
        return _fa_generate_parameterized_report(self)

    def analyze_rolling_ic(self, factor_col, window_sizes=(30, 60), compute_ic_decay=True, save_plots=False):
        """
        针对带参数数据提供滚动IC分析便捷入口。
        """
        if getattr(self, 'processed_data', None) is None or factor_col not in self.processed_data.columns:
            print(f"滚动窗口分析失败：缺少因子 {factor_col} 的有效数据")
            return {}
        df = self.processed_data[['信号日期', factor_col, self.return_col]].dropna()
        return rolling_window_analysis(
            df,
            factor_col,
            self.return_col,
            window_sizes=window_sizes,
            compute_ic_decay=compute_ic_decay,
            save_plots=save_plots,
        )

    def analyze_sample_sensitivity(self, factor_col, sample_sizes=(0.8, 0.9, 1.0), n_iterations=100):
        """
        利用通用样本敏感性工具复核带参数分组的稳健性。
        """
        if getattr(self, 'processed_data', None) is None or factor_col not in self.processed_data.columns:
            print(f"样本敏感性分析失败：缺少因子 {factor_col} 的有效数据")
            return {}
        df = self.processed_data[[factor_col, self.return_col]].dropna()
        return sample_sensitivity_analysis(
            df,
            factor_col,
            self.return_col,
            sample_sizes=sample_sizes,
            n_iterations=n_iterations,
        )

