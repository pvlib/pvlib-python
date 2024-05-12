"""
This module contains functions for transformer modeling.

Transformer models calculate AC power output at a different voltage from the voltage 
at the AC input terminals. Model parameters should be passed as a single dict.

"""

import numpy as np
import pandas as pd


def simple_efficiency(input_power, no_load_loss, load_loss, kva_rating):
    r'''
    Calculate the energy at the output terminal of the transformer
     after taking into account efficiency using a simple calculation.

    Parameters
    ----------
    input_power : numeric
        The power that is input into the transformer. [W]

    no_load_loss : numeric
        The constant losses experienced by a transformer, even
        when the transformer is not under load. [% from 0 to 1]

    load_loss:  numeric
        The load dependent losses experienced by the transformer.
        [% from 0 to 1]

    Returns
    -------
    output_power : numeric
        AC power output. [W]


    References
    ----------
    .. [1] Central Station Engineers of the Westinghouse Electric Corporation,
    "Electrical Transmission and Distribution Reference Book" 4th Edition. pg. 101.

    '''

    full_load_loss = (no_load_loss + load_loss) * kva_rating
    no_load_loss_power = no_load_loss * kva_rating
    load_loss_power = full_load_loss - no_load_loss_power
    loss_power = (
        1 / 
        (2 * load_loss_power) *
        (
            (kva_rating ** 2) +
            (2 * load_loss_power * input_power) - 
            (
                kva_rating * np.sqrt(
                    (kva_rating ** 2) +
                    (4 * load_loss_power * (input_power - no_load_loss))
                )
            )
        )
    )

    output_power = input_power - loss_power

    return output_power


if __name__ == '__main__':
    
    input_power = pd.Series([0, 100, 200, 300])
    no_load_loss = 0.01
    load_loss = 0.005
    kva_rating = 100

    test = simple_efficiency(
        input_power=input_power,
        no_load_loss=no_load_loss,
        load_loss=load_loss,
        kva_rating=kva_rating
    )

    print(test)

    print('done')

