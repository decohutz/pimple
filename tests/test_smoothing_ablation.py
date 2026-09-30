import copy
import pytest
from scripts.compare_smoothing_ablation import assess, validate_ablation


def test_ablation_rejects_second_factor_and_different_seeds():
    a={'dataset':'v2','seeds':[42,43,44],'criteria':{},'recipes':{'r':{'label_smoothing':.05,'learning_rate':.0003}}}
    b=copy.deepcopy(a); b['recipes']['r']['label_smoothing']=0
    validate_ablation(a,b)
    b['recipes']['r']['learning_rate']=.0001
    with pytest.raises(ValueError,match='Only label_smoothing'): validate_ablation(a,b)
    b=copy.deepcopy(a); b['seeds']=[42]
    with pytest.raises(ValueError,match='identical'): validate_ablation(a,b)


@pytest.mark.parametrize('scores,recalls,expected',[
    ([.72,.73,.74],[.56,.56,.56],True),
    ([.72,.73,.74],[.50,.50,.50],False),
    ([.6,.6,.9],[.60,.60,.60],False),
    ([.7,.7,.7],[.60,.60,.60],False),
])
def test_decision_uses_mean_and_melanoma_guard(scores,recalls,expected):
    a=[{'seed':s,'macro_f1':.71,'mel_recall':.55} for s in [42,43,44]]
    b=[{'seed':s,'macro_f1':f,'mel_recall':r} for s,f,r in zip([42,43,44],scores,recalls)]
    assert assess(a,b)['improvement_under_registered_rule']==expected


def test_missing_seed_cannot_be_hidden():
    rows=[{'seed':s,'macro_f1':.7,'mel_recall':.5} for s in [42,43,44]]
    with pytest.raises(ValueError,match='paired'): assess(rows,rows[:2])
