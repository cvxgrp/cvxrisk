Search.setIndex({"docnames": ["docs/api", "docs/index", "docs/nb", "docs/notebooks/demo", "docs/notebooks/difference", "docs/notebooks/monkey", "docs/notebooks/pairs", "docs/notebooks/quantstats", "docs/reports"], "filenames": ["docs/api.md", "docs/index.md", "docs/nb.md", "docs/notebooks/demo.ipynb", "docs/notebooks/difference.ipynb", "docs/notebooks/monkey.ipynb", "docs/notebooks/pairs.ipynb", "docs/notebooks/quantstats.ipynb", "docs/reports.md"], "titles": ["API", "Simulator", "Notebooks", "Long only 1/n portfolio", "Portfolio difference", "Monkey portfolios", "Almost pairs trading", "With quantstats", "ci/cd reports"], "terms": {"given": [1, 3], "univers": [1, 6], "m": [1, 3], "asset": [1, 3, 4, 5, 6, 7], "we": [1, 3, 7], "ar": [1, 3], "price": [1, 4, 5, 6, 7], "each": [1, 3, 6, 7], "them": 1, "t_1": 1, "t_2": 1, "ldot": 1, "t_n": 1, "e": [1, 3], "g": [1, 3], "oper": 1, "us": 1, "an": [1, 6], "n": [1, 2, 5, 7], "matrix": 1, "where": [1, 3], "column": [1, 3], "correspond": 1, "particular": 1, "In": 1, "backtest": 1, "iter": 1, "row": [1, 3], "alloc": 1, "posit": [1, 3, 4], "all": 1, "some": 1, "thi": [1, 3, 6], "tool": 1, "shall": 1, "help": [1, 3], "simplifi": 1, "account": 1, "It": 1, "keep": 1, "track": 1, "avail": 1, "cash": [1, 4], "profit": [1, 3], "achiev": 1, "etc": 1, "The": [1, 6], "complet": [1, 3, 6], "agnost": [1, 6], "trade": [1, 2], "polici": [1, 6], "strategi": [1, 6], "our": 1, "approach": [1, 3], "follow": 1, "rather": [1, 3], "common": 1, "pattern": 1, "demonstr": [1, 6], "those": 1, "step": 1, "somewhat": 1, "silli": [1, 6], "thei": 1, "never": 1, "good": [1, 6], "alwai": [1, 6], "valid": [1, 6], "ones": [1, 4], "user": 1, "defin": [1, 3], "load": [1, 6], "frame": [1, 3], "initi": 1, "amount": [1, 6], "experi": 1, "from": [1, 3, 4, 5, 6, 7], "pathlib": [1, 6], "import": [1, 3, 4, 5, 6, 7], "path": [1, 6], "panda": [1, 3, 4, 5, 6, 7], "pd": [1, 3, 4, 5, 6, 7], "cvx": [1, 3, 4, 5, 6, 7], "read_csv": [1, 4, 5, 6, 7], "resourc": 1, "csv": [1, 4, 5, 6, 7], "index_col": [1, 4, 5, 6, 7], "0": [1, 3, 4, 5, 6, 7], "parse_d": [1, 4, 5, 6, 7], "true": [1, 3, 4, 5, 6, 7], "header": [1, 4, 5, 6, 7], "ffill": [1, 6], "b": [1, 3, 4, 5, 6, 7], "initial_cash": [1, 3, 4, 5, 6, 7], "1e6": [1, 3, 4, 5, 6, 7], "also": 1, "possibl": 1, "specifi": 1, "model": [1, 6], "cost": [1, 6], "fill": [1, 3], "up": 1, "onli": [1, 2], "onc": [1, 3], "done": 1, "construct": [1, 3], "actual": 1, "portfolio": [1, 2, 6, 7], "have": [1, 3], "overload": 1, "__iter__": 1, "__setitem__": 1, "method": 1, "custom": 1, "let": [1, 6], "s": [1, 6], "start": 1, "first": 1, "dai": [1, 6, 7], "choos": [1, 6], "two": [1, 3, 6], "name": [1, 6], "random": [1, 4, 5, 6], "bui": [1, 6], "one": [1, 6], "sai": [1, 6], "1": [1, 2, 4, 5, 6, 7], "your": [1, 3, 6], "wealth": [1, 6], "short": [1, 4, 6], "same": [1, 6], "t": [1, 3, 6, 7], "state": [1, 3, 4, 5, 6, 7], "pick": [1, 6], "pair": [1, 2], "np": [1, 4, 5, 6], "choic": [1, 6], "2": [1, 3, 6], "replac": [1, 6], "fals": [1, 3, 6], "comput": [1, 6], "stock": [1, 3, 6], "seri": [1, 4, 5, 6], "index": [1, 3, 4, 5, 6], "data": [1, 3, 4, 5, 6, 7], "nav": [1, 3, 4, 5, 6, 7], "valu": [1, 6], "updat": 1, "here": [1, 6], "grow": 1, "list": [1, 3], "timestamp": 1, "t1": 1, "t2": 1, "second": 1, "t3": 1, "A": 1, "lot": 1, "magic": 1, "hidden": 1, "variabl": 1, "give": 1, "access": 1, "current": 1, "valuat": 1, "hold": 1, "slightli": 1, "more": 1, "realist": 1, "set": 1, "4": [1, 3], "want": 1, "implmen": 1, "popular": 1, "invest": [1, 3, 7], "quarter": [1, 3, 7], "capit": [1, 3, 7], "25": [1, 3], "note": 1, "last": 1, "element": 1, "than": 1, "weight": [1, 4], "cashposit": 1, "class": 1, "expos": 1, "setter": 1, "convent": 1, "set_weight": [1, 4], "finish": 1, "build": [1, 3, 4, 5, 6, 7], "abov": 1, "desir": 1, "after": 1, "trigger": 1, "readi": 1, "further": 1, "analysi": 1, "dive": 1, "equiti": 1, "mai": 1, "know": 1, "enter": 1, "etern": 1, "run": 1, "non": 1, "python": 1, "wast": 1, "case": 1, "submit": 1, "togeth": 1, "when": 1, "equityportfolio": 1, "assum": 1, "you": [1, 3, 6], "share": [1, 3], "alreadi": 1, "love": 1, "instal": 1, "can": [1, 3], "perform": 1, "replic": 1, "virtual": 1, "environ": 1, "pyproject": 1, "toml": 1, "jupyterlab": 1, "within": [1, 3], "new": 1, "execut": [1, 3], "create_kernel": 1, "sh": 1, "dedic": 1, "project": 1, "long": [2, 4], "differ": [2, 3], "monkei": 2, "almost": [2, 3], "With": 2, "quantstat": 2, "option": [3, 4, 5, 6, 7], "plot": [3, 4, 5, 6, 7], "backend": [3, 4, 5, 6, 7], "plotli": [3, 4, 5, 6, 7], "yfinanc": 3, "yf": 3, "simul": [3, 4, 5, 6, 7], "builder": [3, 4, 5, 6, 7], "resample_index": 3, "download": 3, "ticker": 3, "spy": 3, "aapl": 3, "goog": 3, "msft": 3, "period": 3, "10y": 3, "time": [3, 4, 5, 6], "interv": 3, "1d": 3, "prepost": 3, "pre": 3, "post": 3, "market": 3, "hour": 3, "repair": 3, "obviou": 3, "error": 3, "100x": 3, "50": 3, "75": 3, "3": 3, "100": 3, "adj": 3, "close": [3, 4], "cumsum": 3, "usual": 3, "would": 3, "daili": 3, "basi": 3, "rebal": 3, "everi": [3, 6], "week": 3, "month": 3, "There": 3, "deal": 3, "problem": 3, "cvxsimul": 3, "see": 3, "effect": 3, "hesit": 3, "most": 3, "flexibl": 3, "irregular": 3, "portfolio_resampl": 3, "rule": 3, "truncat": 3, "datafram": 3, "origin": 3, "monthli": 3, "date": 3, "2013": 3, "05": [3, 6], "20": [3, 6], "000000e": 3, "06": [3, 6], "21": 3, "9": 3, "964454e": 3, "22": 3, "890195e": 3, "890028e": 3, "23": 3, "836173e": 3, "836271e": 3, "24": 3, "833095e": 3, "833657e": 3, "2023": 3, "15": 3, "7": 3, "312625e": 3, "294692e": 3, "16": 3, "362801e": 3, "347875e": 3, "17": 3, "430481e": 3, "415860e": 3, "18": 3, "531697e": 3, "517900e": 3, "19": 3, "524960e": 3, "510930e": 3, "2519": 3, "print": [3, 7], "18203": 3, "071629": 3, "11048": 3, "047272": 3, "8552": 3, "064917": 3, "1802": 3, "079651": 3, "10510": 3, "307528": 3, "16525": 3, "679695": 3, "5838": 3, "041749": 3, "4283": 3, "846124": 3, "x": 3, "hard": 3, "between": 3, "number": 3, "trades_stock": 3, "iloc": 3, "els": 3, "forward": 3, "lead": 3, "150k": 3, "had": 3, "realloc": 3, "turnov": 3, "i": 3, "don": 3, "believ": 3, "bring": 3, "render": 3, "signal": 3, "spars": 3, "stai": 3, "numpi": [4, 5, 6], "stock_pric": [4, 5, 6, 7], "len": [4, 5, 7], "w": [4, 5], "rand": [4, 5], "sum": [4, 5], "one_over_n": 4, "diff": 4, "d": 4, "remain": 4, "1m": 4, "financ": 4, "round": 4, "littl": 6, "exercis": 6, "goe": 6, "back": 6, "idea": 6, "stephen": 6, "boyd": 6, "should": 6, "even": 6, "like": 6, "Not": 6, "Of": 6, "cours": 6, "termin": 6, "go": 6, "bust": 6, "which": 6, "seem": 6, "loguru": 6, "logger": 6, "trading_cost": 6, "linearcostmodel": 6, "info": 6, "pai": 6, "fee": 6, "10": 6, "bp": 6, "factor": 6, "0000": 6, "trading_cost_model": 6, "assert": 6, "game": 6, "over": 6, "32m2023": 6, "04": 6, "43": 6, "240": 6, "0m": 6, "1minfo": 6, "36m__main__": 6, "36m": 6, "modul": 6, "36m1": 6, "1mload": 6, "249": 6, "36m5": 6, "1mbuild": 6, "250": 6, "36m8": 6, "support": 7, "recommend": 7, "qs": 7, "snapshot": 7, "titl": 7, "show": 7, "findfont": 7, "font": 7, "famili": 7, "arial": 7, "found": 7, "sharp": 7, "ratio": 7, "stat": 7, "pct_chang": 7, "dropna": 7, "6379901607052793": 7, "6389878459352839": 7, "line": 8, "code": 8}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"api": 0, "sphinx": 0, "depend": 0, "simul": 1, "modu": 1, "operandi": 1, "creat": 1, "builder": 1, "object": 1, "loop": 1, "through": 1, "time": 1, "analys": 1, "result": 1, "bypass": 1, "poetri": 1, "kernel": 1, "notebook": 2, "long": 3, "onli": 3, "1": 3, "n": [3, 4], "portfolio": [3, 4, 5], "rebalanc": 3, "resampl": 3, "an": 3, "exist": 3, "trade": [3, 6], "dai": 3, "predefin": 3, "grid": 3, "why": 3, "price": 3, "differ": 4, "monkei": [4, 5], "One": 4, "over": 4, "almost": 6, "pair": 6, "With": 7, "quantstat": 7, "ci": 8, "cd": 8, "report": 8, "loc": 8, "test": 8}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})