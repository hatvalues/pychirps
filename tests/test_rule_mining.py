from tests.forest_paths_helper import rf_paths, weighted_paths  # noqa # mypy can't cope with pytest fixtures
from pychirps.build_rules.pattern_miner import PatternMiner
from src.pychirps.build_rules.rule_miner import RuleMiner
import data_preprocs.data_providers as dp


def test_pattern_miner_weighted_paths(cervicalb_enc, rf_paths):  # noqa # mypy can't cope with pytest fixtures
    pattern_miner = PatternMiner(forest_path=rf_paths)
    rule_miner = RuleMiner(
        pattern_miner, cervicalb_enc.features, cervicalb_enc.target
    )
