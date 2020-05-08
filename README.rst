FLife - Vibration Fatigue by Spectral Methods
---------------------------------------------

Obtaining vibration fatigue life in the spectral domain.


Installing this package
-----------------------

Use `pip` to install it by:

.. code-block:: console

    $ pip install FLife


Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

    import FLife
    import numpy as np


    dt = 1e-4
    x = np.random.normal(scale=100, size=10000)

    C = 1.8e+22
    k = 7.3

    # Spectral data
    sd = FLife.SpectralData(input=x, dt=dt)

    # Rainflow reference fatigue life
    rf = FLife.Rainflow(sd)

    # Spectral methods
    dirlik = FLife.Dirlik(sd)
    tb2 = FLife.TovoBenasciutti2(sd)
    print(f'          Rainflow: {rf.get_life(C = C, k=k):4.0f} s')
    print(f'            Dirlik: {dirlik.get_life(C = C, k=k):4.0f} s')
    print(f'Tovo Benasciutti 2: {tb2.get_life(C = C, k=k):4.0f} s')

Reference:
Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020


|codeship|

.. |codeship| image:: https://app.codeship.com/projects/b8713910-6aaf-0138-8527-66d00b8d6fc9/status?branch=master
    :target: https://app.codeship.com/projects/394339