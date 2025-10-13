from typing import List

from vllm.logger import init_logger

import math

logger = init_logger(__name__)
class ScoreDim:
    """
    归一化评分维度
    """
    def __init__(self, name: str, median: float, norm_scale=0.0, weight=0.5, reverse=False):
        self.name = name
        self.median = median
        if norm_scale != 0.0:
            self.norm_scale = norm_scale
        else:
            self.norm_scale = 1/median
        self.weight = weight
        self.reverse = reverse

class NormalizedScorer:
    """
    使用 Sigmoid 函数对无界的N个维度值进行归一化评分。
    """

    def __init__(self, dim_list: List[ScoreDim]) -> None:
        """
        :param dim_list: 评分的维度，每个维度需定义中位参考点、缩放系数和权重
        """
        self.dim_list = dim_list
        self.dim_count = len(dim_list)

    @staticmethod
    def _sigmoid_normalize(value, median, norm_scale):
        """Sigmoid 函数: 将 value 映射到 (0, 1)"""
        return 1 / (1 + math.exp(-norm_scale * (value - median)))

    @staticmethod
    def _inv_sigmoid_normalize(value, median, norm_scale):
        """反向 Sigmoid: 用于value越大，评分越低的维度"""
        # 相当于 sigmoid(-x)，但更稳定
        return 1 / (1 + math.exp(norm_scale * (value - median)))

    def score(self, *dims: float) -> float:
        """
        计算综合评分。
        value越大，评分越高 -> 使用正向 Sigmoid
        value越小，评分越高 -> 使用反向 Sigmoid
        """
        if len(dims) > self.dim_count:
            raise ValueError(f"Dim num({len(dims)}) exceeds max num dim({self.dim_count})")

        final_score = 0.0
        for idx, dim_value in enumerate(dims):
            dim_info = self.dim_list[idx]
            if dim_info.reverse:
                score = self._inv_sigmoid_normalize(dim_value, dim_info.median, dim_info.norm_scale)
            else:
                score = self._sigmoid_normalize(dim_value, dim_info.median, dim_info.norm_scale)
            logger.debug(f"{dim_info.name}({dim_info.reverse}) : {score:.10f}")

            # 加权求和
            final_score += score * dim_info.weight
        return max(0.0, min(1.0, final_score))  # 限制到 [0,1]

class TimeAndLengthScorer(NormalizedScorer):
    """
    时间与长度两个维度评分，默认正向，权重0.5
    """
    def __init__(self,
                 time_median, length_median,
                 time_scale=0.0, length_scale=0.0,
                 time_weight=0.5, length_weight=0.5,
                 reverse_time=False, reverse_len=False) -> None:
        dim_list = [ScoreDim("time", time_median, time_scale, time_weight, reverse_time),
                    ScoreDim("length", length_median, length_scale, length_weight, reverse_len)]
        super().__init__(dim_list)

    def score(self, time: float, length: float) -> float:
        return super().score(time, length)


# --- 示例 ---
if __name__ == "__main__":
    # 假设：
    # - 时间：10秒时得分为0.5，80秒时接近1.0 (k=1/10)
    # - 长度：32K时得分为0.5，256K时接近1.0 (k=1/32K)
    scorer: TimeAndLengthScorer = TimeAndLengthScorer(
        time_median=10, time_weight=0.5,
        length_median=32*1024, length_weight=0.5
    )

    test_cases = [
        (5, 16*1024),
        (15, 64*1024),
        (80, 24*1024),
        (2, 256*1024),
    ]

    print("-" * 50)
    for t, l in test_cases:
        s = scorer.score(t, l)
        print(f"time:{t}, len:{l}, score:{s:.3f}")
