COLOR_BLACK = "#000000"
COLOR_GREY = "#777777"
COLOR_WHITE = "#FFFFFF"

COLOR_1A = "#5D85C3"
COLOR_2A = "#009CDA"
COLOR_3A = "#50B695"
COLOR_4A = "#AFCC50"
COLOR_5A = "#DDDF48"
COLOR_6A = "#FFE05C"
COLOR_7A = "#F8BA3C"
COLOR_8A = "#EE7A34"
COLOR_9A = "#E9503E"
COLOR_10A = "#C9308E"
COLOR_11A = "#804597"

LIST_A = (
    COLOR_1A,
    COLOR_2A,
    COLOR_3A,
    COLOR_4A,
    COLOR_5A,
    COLOR_6A,
    COLOR_7A,
    COLOR_8A,
    COLOR_9A,
    COLOR_10A,
    COLOR_11A
)

LIST_NICE_A = (
    COLOR_1A,
    COLOR_9A,
    COLOR_3A,
    COLOR_7A,
    COLOR_4A,
    COLOR_6A
)

GRADIENT_1A_LIGHT = (COLOR_1A, "#81a0d0", "#a5bbde", "#c9d6eb", "#edf1f8")
GRADIENT_2A_LIGHT = (COLOR_2A, "#39b2e2", "#71c8ea", "#aadef3", "#e3f4fb")
GRADIENT_3A_LIGHT = (COLOR_3A, "#77c6ad", "#9ed6c4", "#c5e7dc", "#ecf7f3")
GRADIENT_4A_LIGHT = (COLOR_4A, "#c1d777", "#d3e39e", "#e4eec5", "#f6f9ec")
GRADIENT_5A_LIGHT = (COLOR_5A, "#e5e671", "#eced99", "#f4f4c2", "#fbfbeb")
GRADIENT_6A_LIGHT = (COLOR_6A, "#ffe780", "#ffeea4", "#fff5c9", "#fffced")
GRADIENT_7A_LIGHT = (COLOR_7A, "#fac967", "#fbd993", "#fde8be", "#fef7e9")
GRADIENT_8A_LIGHT = (COLOR_8A, "#f29861", "#f6b58e", "#f9d3bb", "#fdf0e8")
GRADIENT_9A_LIGHT = (COLOR_9A, "#ee7769", "#f39e94", "#f8c5bf", "#fdecea")
GRADIENT_10A_LIGHT = (COLOR_10A, "#d55ea7", "#e18cc0", "#edbad9", "#f9e8f2")
GRADIENT_11A_LIGHT = (COLOR_11A, "#9c6eae", "#b898c5", "#d5c1dc", "#f1eaf3")

GRADIENT_1A_DARK = (COLOR_1A, "#486798", "#344a6c", "#1f2c41", "#0a0f16")
GRADIENT_2A_DARK = (COLOR_2A, "#0079aa", "#005779", "#003449", "#001118")
GRADIENT_3A_DARK = (COLOR_3A, "#3e8e74", "#2c6553", "#1b3d32", "#091411")
GRADIENT_4A_DARK = (COLOR_4A, "#889f3e", "#61712c", "#3a441b", "#131709")
GRADIENT_5A_DARK = (COLOR_5A, "#acad38", "#7b7c28", "#4a4a18", "#191908")
GRADIENT_6A_DARK = (COLOR_6A, "#c6ae48", "#8e7c33", "#554b1f", "#1c190a")
GRADIENT_7A_DARK = (COLOR_7A, "#c1912f", "#8a6721", "#533e14", "#1c1507")
GRADIENT_8A_DARK = (COLOR_8A, "#b95f28", "#84441d", "#4f2911", "#1a0e06")
GRADIENT_9A_DARK = (COLOR_9A, "#b53e30", "#812c22", "#4e1b15", "#1a0907")
GRADIENT_10A_DARK = (COLOR_10A, "#9c256e", "#701b4f", "#43102f", "#160510")
GRADIENT_11A_DARK = (COLOR_11A, "#643675", "#472654", "#2b1732", "#0e0811")

GRADIENT_1A = tuple(reversed(GRADIENT_1A_DARK)) + GRADIENT_1A_LIGHT[1:]
GRADIENT_2A = tuple(reversed(GRADIENT_2A_DARK)) + GRADIENT_2A_LIGHT[1:]
GRADIENT_3A = tuple(reversed(GRADIENT_3A_DARK)) + GRADIENT_3A_LIGHT[1:]
GRADIENT_4A = tuple(reversed(GRADIENT_4A_DARK)) + GRADIENT_4A_LIGHT[1:]
GRADIENT_5A = tuple(reversed(GRADIENT_5A_DARK)) + GRADIENT_5A_LIGHT[1:]
GRADIENT_6A = tuple(reversed(GRADIENT_6A_DARK)) + GRADIENT_6A_LIGHT[1:]
GRADIENT_7A = tuple(reversed(GRADIENT_7A_DARK)) + GRADIENT_7A_LIGHT[1:]
GRADIENT_8A = tuple(reversed(GRADIENT_8A_DARK)) + GRADIENT_8A_LIGHT[1:]
GRADIENT_9A = tuple(reversed(GRADIENT_9A_DARK)) + GRADIENT_9A_LIGHT[1:]
GRADIENT_10A = tuple(reversed(GRADIENT_10A_DARK)) + GRADIENT_10A_LIGHT[1:]
GRADIENT_11A = tuple(reversed(GRADIENT_11A_DARK)) + GRADIENT_11A_LIGHT[1:]