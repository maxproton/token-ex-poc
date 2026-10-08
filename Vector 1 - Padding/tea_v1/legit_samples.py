"""Legitimate inputs for false positive testing.

The built-in set is only a smoke test. Put real long inputs (one per .txt file) in corpus/legit/<category>/
for credible false positive numbers: contracts, logs, source files, pasted articles, non-English text.
"""
import inspect
import json
import pkgutil
import random
from pathlib import Path

WARMUP_PROMPTS = [
    "What is the capital of France?",
    "Can you explain what a hash map is in two sentences?",
    "Summarise the plot of Romeo and Juliet briefly.",
    "How do I reverse a list in Python?",
    "Give me three tips for writing a clear email.",
]

_CJK = [
    "今天的天气非常好，我们决定去公园散步。", "这个系统会记录每一次请求的令牌数量。",
    "安全团队需要及时发现异常的资源消耗。", "他花了一整个下午阅读那份技术报告。",
    "请把会议纪要发送给所有参与者。", "新的模型在多个测试集上表现更好。",
    "我们计划在下个月发布更新版本。", "数据库的备份每天凌晨自动执行。",
    "这篇文章介绍了如何设计可靠的监控系统。", "用户可以在设置页面修改自己的密码。",
    "团队正在评估不同供应商的报价和服务。", "晚上的火车比早上的便宜很多。",
]


def _prose():
    """Long English technical prose from stdlib docstrings."""
    parts, seen = [], set()
    for mod in (json, random, pkgutil, inspect):
        for name in dir(mod):
            doc = inspect.getdoc(getattr(mod, name, None))
            if doc and len(doc) > 200 and doc not in seen:
                seen.add(doc)
                parts.append(doc)
    return "\n\n".join(parts)


def _code():
    return inspect.getsource(inspect)


def _access_log(n=60):
    rng = random.Random(7)
    paths = ["/api/items", "/api/items/42", "/login", "/static/app.js", "/health"]
    lines = []
    for i in range(n):
        lines.append(f'10.0.{rng.randint(0, 9)}.{rng.randint(1, 250)} - - [07/Oct/2026:10:{i % 60:02d}:00 +0000] '
                     f'"GET {rng.choice(paths)} HTTP/1.1" {rng.choice([200, 200, 200, 304, 404])} {rng.randint(200, 9000)}')
    return "\n".join(lines)


def builtin():
    return {
        "prose": [_prose()[:12000]],
        "code": [_code()[:12000]],
        "log": [_access_log(60), _access_log(200)],
        "cjk": ["".join(_CJK)],
        "short": list(WARMUP_PROMPTS),
    }


def load(legit_dir="corpus/legit"):
    """Built-in samples plus any files under legit_dir/<category>/*.txt."""
    data = builtin()
    root = Path(legit_dir)
    if root.is_dir():
        for cat in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(cat.glob("*.txt")):
                data.setdefault(cat.name, []).append(f.read_text(encoding="utf-8", errors="ignore"))
    return data
