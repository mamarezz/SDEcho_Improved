import pandas as pd

from src.predicates import Predicate, enumerate_predicates, predicate_mask


def test_predicate_mask_applies_every_condition():
    frame = pd.DataFrame(
        {
            "Country": ["USA", "USA", "Other"],
            "Education": ["Doctoral", "Standard", "Doctoral"],
        }
    )
    predicate = Predicate({"Country": "USA", "Education": "Doctoral"})

    assert predicate_mask(frame, predicate).tolist() == [True, False, False]
    assert str(predicate) == "Country=USA & Education=Doctoral"


def test_enumeration_respects_order_and_value_limit():
    source = pd.DataFrame({"A": ["x", "x", "y"], "B": [1, 2, 1]})
    target = pd.DataFrame({"A": ["x", "z"], "B": [1, 2]})
    predicates = enumerate_predicates(
        source,
        target,
        candidate_attrs=["A", "B"],
        max_order=1,
        max_values_per_attr=1,
    )

    assert predicates == [Predicate({"A": "x"}), Predicate({"B": 1})]
