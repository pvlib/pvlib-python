import numpy as np
import pandas as pd

from typing import Literal


def power_mismatch_deline(
    rmad,
    model: Literal[
        "fixed-tilt", "single-axis-tracking"
    ] = "single-axis-tracking",
    fillfactor: float = None,
):
    r"""
    Estimate DC power loss due to irradiance non-uniformity.

    This model is described for bifacial modules in [1]_, where the backside
    irradiance is less uniform due to mounting and site conditions.

    Depending on the mounting type, the power loss is estimated with either
    equation (11) or (12) of [1]_. Passing a custom polynomial is also valid.

    Use ``fillfactor`` to account for different fill factors between the
    trained model and the module of interest.
    For example, if the fill factor of the module of interest is
    0.65, then set ``fillfactor=0.65``.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    rmad : numeric
        The Relative Mean Absolute Difference of the cell-by-cell total
        irradiance. [Unitless]
        Check out the *Notes* section for the equation to calculate it from the
        bifaciality and the front and back irradiances.

    model : str, numpy.polynomial.polynomial.Polynomial or list, default ``"single-axis-tracking"``
        The model coefficients to use.
        If a string, it must be one of the following:

        * ``"fixed-tilt"``: Eq. (11) of [1]_.
        * ``"single-axis-tracking"``: Eq. (12) of [1]_.

        If a :external:class:`numpy.polynomial.polynomial.Polynomial`,
        it is evaluated as is.

        If neither a string nor a ``Polynomial``, it must be the coefficients
        of a polynomial in ``rmad``, where the first element is the constant
        term and the last element is the highest order term. A
        :external:class:`~numpy.polynomial.polynomial.Polynomial`
        will be created internally.

    fillfactor : float, optional
        Fill factor at standard test condition (STC) of the module.
        Accounts for different fill factors between the trained model and the
        module under non-uniform irradiance.
        Models from [1]_ were calculated for a ``fillfactor`` of 0.79.
        This parameter will only be used if ``model`` is a string.
        Raises a ``ValueError`` if the model is a custom polynomial.
        Internally, this argument applies :ref:`Equation (7) <fillfactor_eq>`.

    Returns
    -------
    loss : numeric
        The power loss.

    Raises
    ------
    ValueError
        If the model is not a string and ``fillfactor`` is not ``None``.

    Notes
    -----
    The models implemented are equations (11) and (12) of [1]_:

    .. math::

       \text{model="fixed-tilt"} & \Rightarrow M[\%] =
       0.142 \Delta[\%] + 0.032 \Delta[\%]^2 \qquad & \text{(11)}

       \text{model="single-axis-tracking"} & \Rightarrow M[\%] =
       0.054 \Delta[\%] + 0.068 \Delta[\%]^2 \qquad & \text{(12)}

    where :math:`\Delta[\%]` is the Relative Mean Absolute Difference of the
    global irradiance, Eq. (4) of [1]_ and [2]_.

    The losses definition is Eq. (1) of [1]_, and it's defined as a loss of the
    output power:

    .. math::

       M[\%] = 1 - \frac{P_{Array}}{\sum P_{Cells}} \qquad \text{(1)}

    To account for a module with a fill factor distinct from the one used to
    train the model (0.79), the output of the model can be modified by Eq. (7):

    .. _fillfactor_eq:

    .. math::

       M[\%]_{FF_1} = M[\%]_{FF_0} \frac{FF_1}{FF_0} \qquad \text{(7)}

    In the section *See Also*, you will find two packages that can be used to
    calculate the irradiance at different points of the module.

    .. note::
       The global irradiance RMAD is different from the backside irradiance
       RMAD.

    In case the RMAD of the backside irradiance is known, the global RMAD can
    be calculated as follows, assuming the front irradiance RMAD is
    negligible [2]_:

    .. math::

       RMAD(k \cdot X + c) = RMAD(X) \cdot k \frac{k \bar{X}}{k \bar{X} + c}
       = RMAD(X) \cdot k \frac{1}{1 + \frac{c}{k \bar{X}}}

    by similarity with equation (2) of [1]_:

    .. math::

       G_{total\,i} = G_{front\,i} + \phi_{Bifi} G_{rear\,i} \qquad \text{(2)}

    See Also
    --------
    `solarfactors <https://github.com/pvlib/solarfactors/>`_
        Calculate the irradiance at different points of the module.
    `bifacial_radiance <https://github.com/NREL/bifacial_radiance>`_
        Calculate the irradiance at different points of the module.

    References
    ----------
    .. [1] C. Deline, S. Ayala Pelaez, S. MacAlpine, and C. Olalla, 'Estimating
       and parameterizing mismatch power loss in bifacial photovoltaic
       systems', Progress in Photovoltaics: Research and Applications, vol. 28,
       no. 7, pp. 691-703, 2020, :doi:`10.1002/pip.3259`.
    .. [2] “Mean absolute difference,” Wikipedia, Sep. 05, 2023.
       https://en.wikipedia.org/wiki/Mean_absolute_difference#Relative_mean_absolute_difference
       (accessed 2024-04-14).
    """  # noqa: E501
    if isinstance(model, str):
        _MODEL_POLYNOMS = {
            "fixed-tilt": [0, 0.142, 0.032],  # Eq. (11), [1]
            "single-axis-tracking": [0, 0.054, 0.068],  # Eq. (12), [1]
        }
        try:
            model_polynom = np.polynomial.Polynomial(_MODEL_POLYNOMS[model])
        except KeyError:
            raise ValueError(
                f"Invalid model '{model}'. Available models are "
                f"{list(_MODEL_POLYNOMS.keys())}."
            )
        else:
            if fillfactor:
                # Use fillfactor to modify output of a known trained model
                # Eq. (7), [1]
                model_polynom = model_polynom * fillfactor / 0.79
    else:
        if fillfactor:
            raise ValueError(
                "Fill factor can only be used with predefined models. "
                "Modify polynomial or multiply output by "
                "'module_fillfactor / training_fillfactor'."
            )
        if isinstance(model, np.polynomial.Polynomial):
            model_polynom = model
        else:  # expect an iterable
            model_polynom = np.polynomial.Polynomial(coef=model)

    mismatch = model_polynom(rmad)
    if isinstance(rmad, pd.Series):
        mismatch = pd.Series(mismatch, index=rmad.index)
