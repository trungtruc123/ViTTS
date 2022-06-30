import re

abbreviations_vi = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("dn", "đà nẵng"),
        ("qn", "quảng nam"),
        ("dm", "địt mẹ"),
        ("đm", "địt mẹ"),
        ("ck", "chồng"),
        ("vk", "vợ"),
    ]
]
