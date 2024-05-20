"""
This module contains functions for transformer modeling.

Transformer models calculate AC power output at a different voltage from the
voltage at the AC input terminals.
"""

import numpy as np


def simple_efficiency(
        input_power, no_load_loss_fraction, load_loss_fraction,
        transformer_rating
):
    r'''
    Calculate the energy at the output terminal of the transformer
     after taking into account efficiency using a simple calculation.

    The equation used in this function can be derived from the reference.

    First, assume that the load loss is proportional to the square of output
    power.

        .. math::
        L_{load}(P_{out}) = L_{load}(P_{out}) * P^2_{out}
        L_{load}(P_{out}) = L_{full, load} * P^2_{out}

    Total loss is the variable load loss, plus a constant no-load loss:

        .. math::
        L_{total}(P_{out}) = L_{no, load} + L_{load}(P_{out})
        L_{total}(P_{out}) = L_{no, load} + L_{full, load} * P^2_{out}


    Conservation of energy:

        .. math::
        P_{in} = P_{out} + L_{total}(P_{out})
        P_{in} = P_{out} + L_{no, load} + L_{full, load} * P^2_{out}

    Now use quadratic formula to solve for $P_{out}$ as a function of $P_in$.

        ..math::
        P_{out} = \frac{-b +- \sqrt{b^2 - 4ac}}{2a}
        a = L_{full, load}
        b = 1
        c = L_{no, load} - P_{in}

    Therefore:

        ..math::
        P_{out} = \frac{-1 +- \sqrt{1 - 4*L_{full, load}*L_{no, load} -
        P_{in}}}{2*L_{no, load} - P_{in}}

    Note that the positive root must be the correct one if the output power is
    positive.


    Parameters
    ----------
    input_power : numeric
        The real power input to the transformer. [W]

    no_load_loss_fraction : numeric
        The constant losses experienced by a transformer, even
        when the transformer is not under load. [% from 0 to 1]

    load_loss_fraction:  numeric
        The load dependent losses experienced by the transformer.
        [% from 0 to 1]

    transformer_rating: numeric
        The nominal output power of the transformer. [VA]

    Returns
    -------
    output_power : numeric
        Real power output. [W]


    References
    ----------
    .. [1] Central Station Engineers of the Westinghouse Electric Corporation,
    "Electrical Transmission and Distribution Reference Book" 4th Edition.
    pg. 101.

    '''

    # calculate the load loss in terms of VA instead of percent
    loss_at_full_load = (
        (no_load_loss_fraction + load_loss_fraction) * transformer_rating
    )
    no_load_loss = no_load_loss_fraction * transformer_rating
    load_loss = loss_at_full_load - no_load_loss

    # calculate how much power is lost
    combined_loss = (
        (1 / (2 * load_loss)) *
        (
            (transformer_rating ** 2) +
            (2 * load_loss * input_power) -
            (transformer_rating * np.sqrt(
                (transformer_rating ** 2) +
                (4 * load_loss) * (input_power - no_load_loss)
            ))
        )
    )

    # calculate final output power given calculated losses
    output_power = input_power - combined_loss

    return output_power
