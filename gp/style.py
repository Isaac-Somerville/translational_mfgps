"""The font the published figures were typeset in.

Aptos is loaded from a local Microsoft Word installation; if it is unavailable,
matplotlib falls back to its default face.

Import :func:`apply_publication_font` once per figure-producing script, before
any figure is created.
"""

import os

from matplotlib import font_manager, rcParams

#: Where Microsoft Word keeps the face the paper's figures use.
APTOS_PATH = "/Applications/Microsoft Word.app/Contents/Resources/DFonts/Aptos.ttf"


def apply_publication_font():
    """Register Aptos and make it the default family, if it is installed.

    Returns:
        The family name that was set, or None if the font was not found.
    """
    if not os.path.exists(APTOS_PATH):
        return None
    font_manager.fontManager.addfont(APTOS_PATH)
    family = font_manager.FontProperties(fname=APTOS_PATH).get_name()
    rcParams["font.family"] = family
    # Without this, a label like $\mathrm{CRPS_{HF+LF}}$ renders in DejaVu.
    rcParams.update({
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": family,
        "mathtext.bf": family,
    })
    return family
