"""面板重画时控件会删掉重建。自己带状态的控件（色带的当前色标、曲线的当前控制点）
按「数据对象 + 属性名」把状态记在这里，重建出来的新控件接着用。"""
from __future__ import annotations

_states: dict[tuple, dict] = {}


def state_for(target, attr: str) -> dict:
    key = (id(target), attr)
    state = _states.get(key)
    if state is None:
        if len(_states) > 512:
            _states.clear()
        state = _states[key] = {}
    return state
