from functools import total_ordering

@total_ordering
class WeightedScore:
    def __init__(self, score: float, strategy: str = "arrival_time_with_sjf"):
        self.score = score
        self.strategy = strategy

    def __lt__(self, other_request_weighted_score: 'WeightedScore') -> bool:
        if self.strategy == "base_sjf":
            return self.score < other_request_weighted_score.score
        elif self.strategy == "arrival_time_with_sjf":
            return self.score > other_request_weighted_score.score
        else:
            raise ValueError(f"Unsupported strategy for sorting: {self.strategy}")

    def __eq__(self, other_request_weighted_score: 'WeightedScore') -> bool:
        return self.score == other_request_weighted_score.score