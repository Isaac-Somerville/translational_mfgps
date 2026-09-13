"""Drug channel block via the Hill equation.

This module is duplicated verbatim in the companion simulation repository. It
depends on nothing else in either package, so each ships its own copy under its
own licence.

A :class:`Drug` holds the CiPA Hill-equation parameters for one compound and
turns a concentration into a per-channel block fraction.

For channel :math:`c` at concentration :math:`x`, the fraction of current
blocked is

.. math::

    b_c(x) = \\frac{x^{h_c}}{\\mathrm{IC50}_c^{h_c} + x^{h_c}}

evaluated in the equivalent form ``1 / (1 + (IC50 / x) ** h)``, which is
better conditioned for small ``x``.  The cell models consume the *multiplier*
``1 - b_c(x)``, which scales the corresponding maximal conductance.

Parameters come from ``data/cipa/CiPA_optimal_combined.csv``: one row per drug,
columns ``<channel>_IC50`` and ``<channel>_h`` for each of the seven channels
(ICaL, IK1, IKs, INa, INaL, Ito, hERG), plus the maximum tested concentration
``c_max``, the free therapeutic concentration ``c_therapeutic``, and the CiPA
torsade-risk category.

A channel whose IC50 or Hill coefficient is missing is treated as unblocked at
every concentration, as is any concentration of exactly zero.
"""

import numpy as np
import pandas as pd

from paths import CIPA_TABLE

#: The seven ion channels the CiPA dataset characterises. Species models cover
#: different subsets of these; see ``data/channels/channels_per_model.csv``.
ALL_CHANNELS = ["ICaL", "IK1", "IKs", "INa", "INaL", "Ito", "hERG"]

_CIPA_RISK = {0: "Low / None", 1: "Intermediate", 2: "High"}

_DRUG_TABLE = None


def load_drug_table():
    """Return the CiPA parameter table, reading it from disk once."""
    global _DRUG_TABLE
    if _DRUG_TABLE is None:
        _DRUG_TABLE = pd.read_csv(CIPA_TABLE, index_col=0)
    return _DRUG_TABLE


class Drug:
    """Hill-equation channel block for one compound.

    Args:
        name: Row label in the CiPA table, e.g. ``"quinidine"``.
        drug_data: Parameter table to read from. Defaults to the shipped CiPA
            table; pass a DataFrame with the same columns to override.
        channel_list: Channels to model. Pass the host cell model's channel
            list so that block is only computed for currents the model has.
            Defaults to every channel with non-missing parameters for this drug.

    Attributes:
        hill_parameters: The drug's ``<channel>_IC50`` / ``<channel>_h`` row.
        c_max: Maximum tested concentration (nM).
        c_therapeutic: Free therapeutic concentration (nM).
        cipa_risk: CiPA torsade-risk category as a string.
    """

    def __init__(self, name, drug_data=None, channel_list=None):
        if drug_data is None:
            drug_data = load_drug_table()

        self.name = name
        # The first 14 columns are the seven (IC50, h) pairs; the rest are
        # concentrations and the risk label.
        self.hill_parameters = drug_data.iloc[:, :14].loc[name]

        if channel_list is not None:
            self.channel_list = channel_list
        else:
            self.channel_list = [
                ch
                for ch in ALL_CHANNELS
                if not pd.isna(drug_data.loc[name, f"{ch}_IC50"])
            ]

        self.c_max = drug_data["c_max"].loc[name]
        self.c_therapeutic = drug_data["c_therapeutic"].loc[name]
        self.cipa_risk = _CIPA_RISK[drug_data["cipa_category"].loc[name]]

    def calculate_inhibition(self, concentrations, channel, return_multipliers=False):
        """Block fraction of one channel at each concentration.

        Args:
            concentrations: Array of concentrations (nM).
            channel: Channel name, which must be in ``self.channel_list``.
            return_multipliers: If True return ``1 - block`` — the factor the
                cell model applies to that channel's conductance — instead of
                the block fraction itself.

        Returns:
            Array shaped like ``concentrations``. Entries at zero concentration
            are 0 (or 1 for multipliers), as is every entry if this drug has no
            parameters for the channel.
        """
        if channel not in self.channel_list:
            raise ValueError(f"Channel {channel} not in drug channel list!")

        IC50 = self.hill_parameters[channel + "_IC50"]
        h = self.hill_parameters[channel + "_h"]

        out = np.ones(concentrations.shape) if return_multipliers else np.zeros(
            concentrations.shape
        )
        if np.isnan(IC50) or np.isnan(h):
            return out

        nonzero = concentrations != 0
        block = 1 / (1 + (IC50 / concentrations[nonzero]) ** h)
        out[nonzero] = 1 - block if return_multipliers else block
        return out

    def calculate_all_inhibitions(
        self, concentrations, return_multipliers=False, return_as_list=True
    ):
        """Block fractions for every channel in ``self.channel_list``.

        Args:
            concentrations: Array of concentrations (nM).
            return_multipliers: As in :meth:`calculate_inhibition`.
            return_as_list: If True return an array of shape
                ``(n_channels, n_concentrations)`` ordered as
                ``self.channel_list``; if False return a dict keyed by channel.
        """
        if return_as_list:
            result = np.zeros((len(self.channel_list), len(concentrations)))
            for i, channel in enumerate(self.channel_list):
                result[i] = self.calculate_inhibition(
                    concentrations, channel, return_multipliers=return_multipliers
                )
            return result

        return {
            channel: self.calculate_inhibition(
                concentrations, channel, return_multipliers=return_multipliers
            )
            for channel in self.channel_list
        }

    def endpoint_magnitude(self, c_end="c_therapeutic", c_end_multiplier=1.0, ord=2):
        """Norm of the block vector at a single concentration.

        How hard the drug blocks overall, ignoring which channels it favours.

        Args:
            c_end: ``"c_therapeutic"``, ``"c_max"``, or a concentration in nM.
            c_end_multiplier: Scales ``c_end``, e.g. 3.0 for 3x therapeutic.
            ord: Norm order passed to :func:`numpy.linalg.norm`.

        Returns:
            Scalar in ``[0, n_channels ** (1 / ord)]``.
        """
        if c_end == "c_max":
            c_end = self.c_max * c_end_multiplier
        elif c_end == "c_therapeutic":
            c_end = self.c_therapeutic * c_end_multiplier
        else:
            c_end = c_end * c_end_multiplier

        endpoint = self.calculate_all_inhibitions(np.array([c_end])).flatten()
        return np.linalg.norm(endpoint, ord=ord)
