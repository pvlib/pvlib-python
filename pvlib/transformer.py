"""
This module contains functions for transformer modeling.

Transformer models calculate AC power output and losses at a given input power.
"""


def simple_efficiency(
        input_power, no_load_loss, load_loss, transformer_rating
):
    r'''
    Calculate the power at the output terminal of the transformer
    after taking into account efficiency using a simple calculation.

    The equation used in this function can be derived from [1]_.

    For a zero input power, the output power will be negative.
    This means the transformer will consume energy from the grid at night if
    it stays connected (due to the parallel impedance in the equivalent
    circuit).
    If the input power is negative, the output power will be even more
    negative; so the model can be used bidirectionally when drawing
    energy from the grid.

    Parameters
    ----------
    input_power : numeric
        The real AC power input to the transformer. [W]

    no_load_loss : numeric
        The constant losses experienced by a transformer, even
        when the transformer is not under load. Fraction of transformer rating,
        value from 0 to 1. [unitless]

    load_loss:  numeric
        The load dependent losses experienced by the transformer.
        Fraction of transformer rating, value from 0 to 1. [unitless]

    transformer_rating: numeric
        The nominal output power of the transformer. [VA]

    Returns
    -------
    output_power : numeric
        Real AC power output. [W]

    Notes
    -------
    First, assume that the load loss :math:`L_{load}` (as a fraction of rated power
    :math:`P_{nom}`) is proportional to the square of output power:

    .. math::

       L_{load}(P_{out}) &= L_{load}(P_{rated}) \times (P_{out} / P_{nom})^2

                         &= L_{full, load} \times (P_{out} / P_{nom})^2

    Total loss is the constant no-load loss plus the variable load loss:

    .. math::

       L_{total}(P_{out}) &= L_{no, load} + L_{load}(P_{out})

                          &= L_{no, load} + L_{full, load} \times (P_{out} / P_{nom})^2


    By conservation of energy, total loss is the difference between input and
    output power:

    .. math::

       \frac{P_{in}}{P_{nom}} &= \frac{P_{out}}{P_{nom}} + L_{total}(P_{out})

                              &= \frac{P_{out}}{P_{nom}} + L_{no, load} + L_{full, load} \times (P_{out} / P_{nom})^2

    Now use the quadratic formula to solve for :math:`P_{out}` as a function of
    :math:`P_{in}`.

    .. math::

       \frac{P_{out}}{P_{nom}} &= \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}

       a &= L_{full, load}

       b &= 1

       c &= L_{no, load} - P_{in} / P_{nom}

    Therefore:

    .. math::

       P_{out} = P_{nom} \frac{-1 \pm \sqrt{1 - 4 L_{full, load}

       \times (L_{no, load} - P_{in}/P_{nom})}}{2 L_{full, load}}

    The positive root should be chosen, so that the output power is
    positive.


    References
    ----------
    .. [1] Central Station Engineers of the Westinghouse Electric Corporation,
       "Electrical Transmission and Distribution Reference Book" 4th Edition.
       pg. 101.
    '''  # noqa: E501

    input_power_normalized = input_power / transformer_rating

    a = load_loss
    b = 1
    c = no_load_loss - input_power_normalized

    output_power_normalized = (-b + (b**2 - 4*a*c)**0.5) / (2 * a)

    output_power = output_power_normalized * transformer_rating
    return output_power
