def relative_error(value, value_ref):
    """Return the relative error 

    :param value: [int,float]
        Value to compare
    :param value: [int,float]
        Reference value
    :return err: float
        Relative error.
    """ 
    return (value-value_ref)/value_ref

def basquin_to_sn(Sf, b, range = False):
    """Converts Basquin equation parameters Sf and b to fatigue life parameters C and k,
    as defined in [2]. Basic form of Basquin equation is used here: Sa = Sf* (2*N)**k 

    :param Sf: [int,float]
        Fatigue strength coefficient [MPa**k].
    :param b : [int,float]
        Fatigue strength exponent [/]. Represents S-N curve slope.
    :param range : bool
        False/True sets returned value C with regards to amplitude / range count, respectively.
    :return C, k: float
        C - S-N curve intercept [MPa**k], k - S-N curve inverse slope [/].
    """ 
    if not range:
        k = -1/b 
        C = 0.5*Sf**k
        return C,k

    else: 
        k = -1/b 
        C = 0.5*(2*Sf)**k
        return C,k