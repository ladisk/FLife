FLife - Vibration Fatigue by Spectral Methods
---------------------------------------------

Obtaining vibration fatigue life in the spectral domain.


Installing this package
-----------------------

Use `pip` to install it by:

.. code-block:: console

    $ pip install FLife


Supported methods in the frequency-domain
-----------------------------------------

    - Dirlik,
    - Tovo Benasciutti,
    - Zhao Baker,
    - Narrowband,
    - Alpha 0.75,
    - Wirsching Light,
    - Rice,
    - Gao Moan,
    - Petrucci Zuccarello.

Rainflow (time-domain) is supported using the `fatpack` package.


Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

    import FLife
    import numpy as np


    dt = 1e-4
    x = np.random.normal(scale=100, size=10000)

    C = 1.8e+22  # S-N curve intercept [MPa**k]
    k = 7.3 # S-N curve inverse slope [/]

    # Spectral data
    sd = FLife.SpectralData(input=x, dt=dt)

    # Rainflow reference fatigue life 
    # (do not be confused here, spectral data object also holds the time domain data)
    rf = FLife.Rainflow(sd)

    # Spectral methods
    dirlik = FLife.Dirlik(sd)
    tb = FLife.TovoBenasciutti(sd)
    print(f'          Rainflow: {rf.get_life(C = C, k=k):4.0f} s')
    print(f'            Dirlik: {dirlik.get_life(C = C, k=k):4.0f} s')
    print(f'Tovo Benasciutti 2: {tb.get_life(C = C, k=k, method="method 2"):4.0f} s')
    
Reference:
Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020, `see Elsevier page. <https://www.elsevier.com/books/Vibration%20Fatigue%20by%20Spectral%20Methods/9780128221907?utm_campaign=ELS%20STBK%20AuthorConnect%20Release&utm_campaignPK=1695759095&utm_term=OP66802&utm_content=1695850484&utm_source=93&BID=1212165450>`_


|Build Status|

.. |Build Status| image:: https://travis-ci.com/ladisk/FLife.svg?branch=master
   :target: https://travis-ci.com/ladisk/FLife
