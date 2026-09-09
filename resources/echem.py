"""
echem.py
========
Shared toolkit for CHEM 400: Electrochemistry for Energy Conversion and Storage.

Every lecture notebook imports this file instead of re-typing constants and
equations.  That means an equation is written down **once**, in one place, and
every figure in the course is built from the same one.

Using it
--------
The "Configure Environment" cell of each lecture does this::

    import sys
    if '../Resources' not in sys.path:
        sys.path.append('../Resources')

    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import interact, widgets

    import echem
    from echem import R, T, F, f, WIDE
    echem.use_style()

After that you can call anything below, for example::

    echem.cottrell(t=0.5, conc=1e-6, diff_coeff=1e-5, area=1.0)

Reading it
----------
The file is organised into numbered sections.  Search for the banner (for
example ``SECTION 5``) to jump to one.

    SECTION 1 .... Physical constants
    SECTION 2 .... Figures: style, sizes, colours
    SECTION 3 .... Special functions (erf, erfc, erfcx)
    SECTION 4 .... Thermodynamics: the Nernst equation and reference scales
    SECTION 5 .... Electron transfer kinetics: Butler-Volmer
    SECTION 6 .... Mass transport: Cottrell, diffusion layers, UMEs
    SECTION 7 .... Hydrodynamics: Levich and Koutecky-Levich
    SECTION 8 .... Finite diffusion and thin layers
    SECTION 9 .... Galvanostatic (controlled current) techniques
    SECTION 10 ... Coupled chemical reactions (PCET)
    SECTION 11 ... Ionic transport: transference and junction potentials
    SECTION 12 ... Waveforms
    SECTION 13 ... Helpers for interactive widgets
    SECTION 14 ... Drawing helpers reused by more than one lecture

Every function states its **units** in the docstring.  Electrochemistry is a
field where a missing factor of 1000 is the difference between a result and a
retraction, so please read them.

Sign convention
---------------
This file uses the **modern (IUPAC) convention** as a rule:

    anodic (oxidation) currents are POSITIVE
    cathodic (reduction) currents are NEGATIVE

The older literature often does the opposite (see Lesson 6).  If a figure needs
the historic convention, negate the result at the point of plotting and say so
in the axis label.  Never flip the sign silently inside a calculation.

Two exceptions, both flagged in their own docstrings:

  * Functions documented as returning a MAGNITUDE (cottrell, levich, ...) are
    unsigned by design.  Give them a sign at the point of use.
  * thin_layer_reversible and thin_layer_irreversible return a POSITIVE current
    for a reduction, i.e. cathodic positive, matching the presentation in
    Lesson 28.  Negate them if you need anodic-positive output.

Units used throughout
---------------------
    potential, overpotential   V
    current                    A
    charge                     C
    concentration              mol cm^-3      (NOT mol L^-1)
    area                       cm^2
    distance                   cm
    diffusion coefficient      cm^2 s^-1
    rate constant k0           cm s^-1
    time                       s
    angular velocity           rad s^-1       (NOT rpm; see rpm_to_rad)
    kinematic viscosity        cm^2 s^-1
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc, erfcx, erfinv   # vectorised: they accept arrays


# =============================================================================
# SECTION 1: Physical constants
# =============================================================================

R = 8.314          # ideal gas constant, J mol^-1 K^-1
T = 298.0          # temperature, K (25 C; everything below assumes this)
F = 96485.0        # Faraday constant, C mol^-1
f = F / (R * T)    # 1/(thermal voltage), V^-1.  f = 38.92 V^-1 at 298 K

#: RT/F, the "thermal voltage", 0.02569 V at 298 K.  A one-electron Nernstian
#: wave changes by 2.303*RT/F = 59.2 mV per decade of concentration ratio.
THERMAL_VOLTAGE = R * T / F

#: How far into the tails of the transition voltage distribution a two phase
#: plateau is taken to run, in standard deviations (Lesson 34).  See
#: :func:`two_phase_voltage`.  At 2.5 the plateau and the single phase branches
#: meet with very nearly matching slopes for the default plateau, so dQ/dE joins
#: smoothly.  Widening plateau_slope well beyond its default steepens the ends
#: of the plateau and a small step reappears at each phase boundary.
PLATEAU_SPREAD = 2.5

#: Kinematic viscosity, cm^2 s^-1, for solvents used in hydrodynamic
#: experiments (Lesson 30).
KINEMATIC_VISCOSITY = {
    'H2O': 0.009132,
    'D2O': 0.01028,
    'CH3CN': 0.004536,
    'DMSO': 0.01896,
    'pyridine': 0.009518,
    'DMF': 0.008971,
}


# =============================================================================
# SECTION 2: Figures: style, sizes, colours
# =============================================================================

#: Location of the shared matplotlib style sheet, resolved relative to THIS
#: file so it works no matter which notebook imports it.
STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jupyter.mplstyle')

SINGLE = (3.5, 2.625)   # one panel; this is also the style sheet's default
WIDE = (7.0, 2.625)     # two panels side by side
LARGE = (7.0, 5.0)      # 3D plots and anything that needs room


def use_style():
    """Apply the course matplotlib style sheet.

    Call once per notebook, in the Configure Environment cell.  Feel free to
    edit ``Resources/jupyter.mplstyle``; it only affects your own notebooks.
    """
    plt.style.use(STYLE)


def palette():
    """Return the course colour cycle as a list of colours.

    Use this instead of typing colours out by hand, so that editing the style
    sheet changes every figure in the course at once.
    """
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def color(i):
    """Return colour number ``i`` from the course palette, wrapping around.

    Wrapping means you can never run off the end of the list, no matter how
    many things you plot.
    """
    colors = palette()
    return colors[i % len(colors)]


# =============================================================================
# SECTION 3: Special functions
# =============================================================================
#
# erf, erfc and erfcx are imported from scipy at the top of this file.  They are
# *vectorised*: you can hand them a whole numpy array and get an array back, so
# you never need to write a Python loop over the points of a curve.
#
#     erf(x)    the error function
#     erfc(x)   the complementary error function, 1 - erf(x)
#     erfcx(x)  the SCALED complementary error function, exp(x**2) * erfc(x)
#     erfinv(y) the INVERSE error function: erfinv(erf(x)) == x
#
# erfinv turns a cumulative Gaussian back into the variable underneath it, which
# is how Lesson 34 spreads a phase transition over a distribution of voltages.
#
# erfcx matters in Lesson 21.  Computing exp(x**2)*erfc(x) directly overflows to
# infinity for x greater than about 27, and inf * 0 is a NaN.  erfcx computes
# the same quantity stably and never overflows.


# =============================================================================
# SECTION 4: Thermodynamics
# =============================================================================

def nernst_fraction_oxidized(potential, e_formal=0.0, n=1, temperature=T):
    """Fraction of a redox couple present as O at the electrode surface.

    This is the Nernst equation rearranged into the fraction that is oxidised::

        C_O / (C_O + C_R) = 1 / (1 + exp(-n f (E - E0')))

    It goes to 1 at potentials well positive of E0' and to 0 well negative of
    it, passing through 0.5 exactly at E0'.

    Parameters
    ----------
    potential : float or array, V
    e_formal : float, V
        Formal potential E0', on the same scale as ``potential``.
    n : int
        Number of electrons.
    temperature : float, K
        Defaults to 298 K.  Raising it widens the curve, because the only place
        temperature enters is through f = F/(RT) (Lesson 9).

    Returns
    -------
    float or array, dimensionless, between 0 and 1.
    """
    f_local = F / (R * temperature)
    return 1.0 / (1.0 + np.exp(-n * f_local * (np.asarray(potential) - e_formal)))


def nernst_fraction_reduced(potential, e_formal=0.0, n=1, temperature=T):
    """Fraction of a redox couple present as R at the electrode surface.

    Simply ``1 - nernst_fraction_oxidized(...)``.  See that function.
    """
    return 1.0 - nernst_fraction_oxidized(potential, e_formal, n, temperature)


def nernstian_slope(n=1, temperature=T):
    """Volts per decade of concentration ratio for a Nernstian couple.

    This is 2.303*RT/nF, the number every potentiometric measurement is
    compared against: 0.0592 V (59.2 mV) per decade for n = 1 at 298 K.  A
    sensor that returns this slope is behaving as the Nernst equation demands;
    one that returns less is said to have a "sub-Nernstian" response.

    Parameters
    ----------
    n : int
        Number of electrons.
    temperature : float, K

    Returns
    -------
    float, V per decade (a positive number).
    """
    return 2.303 * R * temperature / (n * F)


def combine_half_cells(e_cathode, e_anode):
    """Cell potential from two standard reduction potentials.

    E_rxn = E_cathode - E_anode, with both quoted as REDUCTIONS on the same
    reference scale (Lessons 4 and 10).  A positive result means the reaction is
    spontaneous as written.

    Both arguments and the result are in V.
    """
    return e_cathode - e_anode


#: Reference electrode potentials, V vs NHE (Lesson 10).
REFERENCE_ELECTRODES = {
    'NHE': 0.0,
    'Hg/HgO': 0.930,
    'Hg/HgSO4/K2SO4': 0.65,
    'Fc/Fc+': 0.624,
    "Ag/AgCl (sat'd KCl)": 0.197,
}

#: Standard reduction potentials for selected couples, V vs NHE (Lesson 10).
REDOX_COUPLES = {
    'H2O/O2': 1.23,
    'Zn/Zn2+': -0.76,
    'Hg/Hg2+': 0.91,
    'Cu/Cu2+': 0.337,
    'Fe/Fe3+': -0.04,
    'Ni/Ni2+': -0.25,
    '2Cu/Cu2O': -0.36,
}


def convert_reference(potential, from_scale, to_scale):
    """Convert a potential between two reference electrode scales.

    >>> convert_reference(0.0, "Ag/AgCl (sat'd KCl)", 'NHE')
    0.197

    Parameters
    ----------
    potential : float or array, V, measured against ``from_scale``.
    from_scale, to_scale : str
        Keys of :data:`REFERENCE_ELECTRODES`.

    Returns
    -------
    The same potential expressed against ``to_scale``, in V.

    Notes
    -----
    This handles the reference electrode only.  Converting to or from RHE also
    requires the pH, because E(RHE) = E(NHE) - 0.0592 * pH at 298 K.
    """
    return (potential
            + REFERENCE_ELECTRODES[from_scale]
            - REFERENCE_ELECTRODES[to_scale])


def rhe_offset(ph, temperature=T):
    """Potential of a reversible hydrogen electrode, V vs NHE.

    The RHE is a hydrogen electrode sitting in the *working* solution rather
    than in 1 M acid, so its potential tracks the pH of that solution::

        E(RHE) = -2.303 (RT/F) pH = -0.0592 pH   at 298 K

    Returns a negative number for any pH above 0.
    """
    return -nernstian_slope(n=1, temperature=temperature) * np.asarray(ph)


def to_rhe(potential, ph, from_scale='NHE', temperature=T):
    """Convert a measured potential onto the RHE scale.

    Combines :func:`convert_reference` with :func:`rhe_offset`::

        E(vs RHE) = E(vs NHE) - E(RHE vs NHE) = E(vs NHE) + 0.0592 pH

    Parameters
    ----------
    potential : float or array, V, measured against ``from_scale``.
    ph : float or array
        pH of the solution the working electrode is in.
    from_scale : str
        Key of :data:`REFERENCE_ELECTRODES`; defaults to NHE.
    temperature : float, K

    Returns
    -------
    The same potential expressed against RHE, in V.

    Notes
    -----
    The RHE scale is the natural one for water electrolysis, because on it the
    hydrogen and oxygen reactions sit at 0 and 1.23 V at *every* pH.  That is
    exactly why plots in the water splitting literature quote V vs RHE.
    """
    e_nhe = convert_reference(potential, from_scale, 'NHE')
    return e_nhe - rhe_offset(ph, temperature)


#: Half-cell reactions used by the battery exercise in Lesson 10.
#:
#: Each entry gives the standard reduction potential (V vs NHE) and the
#: "equivalent weight" of the active material: the grams of that material
#: consumed per mole of electrons passed.  Working per mole of electrons means
#: any two half cells in this table can be paired without rebalancing the
#: stoichiometry by hand.
#:
#: The masses count the named active material ONLY.  A real cell also carries
#: electrolyte, separator, current collectors and a case, so a manufactured
#: battery delivers roughly a third of the number computed here.
BATTERY_COUPLES = {
    #  name                        E0 /V vs NHE   g per mol e-
    'Li+/Li':                     (-3.04,           6.94),   # Li
    'Na+/Na':                     (-2.71,          22.99),   # Na
    'ZnO/Zn (alkaline)':          (-1.28,          32.69),   # Zn
    'S/Li2S':                     (-0.89,          16.03),   # S
    'Cd(OH)2/Cd':                 (-0.81,          56.21),   # Cd
    'Zn2+/Zn (acidic)':           (-0.76,          32.69),   # Zn
    'PbSO4/Pb':                   (-0.36,         103.60),   # Pb
    'H+/H2':                      ( 0.00,           1.01),   # H2
    'MnO2/Mn2O3 (alkaline)':      ( 0.15,          86.94),   # MnO2
    'LiFePO4':                    ( 0.41,         157.76),   # LiFePO4
    'NiOOH/Ni(OH)2':              ( 0.49,          91.70),   # NiOOH
    'LiCoO2':                     ( 0.86,          97.87),   # LiCoO2
    'O2/H2O':                     ( 1.23,           8.00),   # O2
    'PbO2/PbSO4':                 ( 1.69,         119.60),   # PbO2
}


def theoretical_specific_energy(e_cell, mass_anode, mass_cathode):
    """Theoretical specific energy of a cell, W h per kg of active material.

    One mole of electrons carries F coulombs, so a cell running at ``e_cell``
    volts delivers ``F * e_cell`` joules per mole of electrons while consuming
    ``mass_anode + mass_cathode`` grams of active material.  Dividing and
    converting joules per gram to watt-hours per kilogram (a factor of 3.6)
    gives the number below.

    Parameters
    ----------
    e_cell : float, V
        Cell potential, normally from :func:`combine_half_cells`.
    mass_anode, mass_cathode : float, g per mole of electrons
        Equivalent weights, as stored in :data:`BATTERY_COUPLES`.

    Returns
    -------
    float, W h kg^-1.  It is an upper bound and ignores everything that is not
    active material; see the note on :data:`BATTERY_COUPLES`.
    """
    joules_per_gram = F * e_cell / (mass_anode + mass_cathode)
    return joules_per_gram / 3.6


# =============================================================================
# SECTION 5: Electron transfer kinetics (Lessons 11 to 13)
# =============================================================================

def rate_constants(potential, k0, alpha=0.5, e_formal=0.0):
    """Potential-dependent anodic and cathodic rate constants.

        k_c = k0 exp(-alpha f (E - E0'))
        k_a = k0 exp((1 - alpha) f (E - E0'))

    Remember the lesson: rate constants are NOT constant.  The standard rate
    constant k0 is.

    Parameters
    ----------
    potential : float or array, V
    k0 : float, cm s^-1
    alpha : float, dimensionless, 0 to 1
    e_formal : float, V

    Returns
    -------
    (k_anodic, k_cathodic), each cm s^-1.
    """
    over = np.asarray(potential) - e_formal
    k_a = k0 * np.exp((1.0 - alpha) * f * over)
    k_c = k0 * np.exp(-alpha * f * over)
    return k_a, k_c


def exchange_current(k0, conc_o, conc_r, alpha=0.5, area=1.0, n=1):
    """Exchange current i0, the current flowing in each direction at equilibrium.

        i0 = n F A k0 (C_O*)^(1-alpha) (C_R*)^alpha

    Note which concentration carries which exponent: C_O is raised to
    (1 - alpha) and C_R to alpha.  Getting these the wrong way round is
    invisible whenever C_O = C_R, which is exactly the condition used to define
    k0, so it is an easy mistake to keep.

    Parameters
    ----------
    k0 : float, cm s^-1
    conc_o, conc_r : float, mol cm^-3 (bulk concentrations)
    alpha : float, dimensionless
    area : float, cm^2
    n : int

    Returns
    -------
    float, A.  Always positive: it is a magnitude, not a signed current.
    """
    return n * F * area * k0 * np.power(conc_o, 1 - alpha) * np.power(conc_r, alpha)


def butler_volmer_branches(potential, k0, conc_o, conc_r,
                           alpha=0.5, area=1.0, n=1, e_formal=0.0):
    """The anodic and cathodic branches of the Butler-Volmer equation, separately.

        i_a = +n F A k_a C_R      (oxidation, positive)
        i_c = -n F A k_c C_O      (reduction, negative)

    Splitting them is what makes the exchange current visible: at equilibrium
    the two are equal in magnitude and opposite in sign, so the total is zero
    while each branch is i0.

    Concentrations are the values AT THE ELECTRODE SURFACE.  Passing bulk
    concentrations assumes mass transport is fast enough not to matter, which
    is the assumption behind the pure-kinetics figures in Lesson 11.

    Returns
    -------
    (i_anodic, i_cathodic), each in A, signed as above.
    """
    k_a, k_c = rate_constants(potential, k0, alpha, e_formal)
    i_a = n * F * area * k_a * conc_r
    i_c = -n * F * area * k_c * conc_o
    return i_a, i_c


def butler_volmer(potential, k0, conc_o, conc_r,
                  alpha=0.5, area=1.0, n=1, e_formal=0.0):
    """Total Butler-Volmer current: the sum of the two branches.

    Currents are additive, so this is simply ``i_a + i_c``.  Positive means the
    oxidation is winning.  See :func:`butler_volmer_branches` for the pieces
    and for the parameter list.

    Returns
    -------
    float or array, A.
    """
    i_a, i_c = butler_volmer_branches(potential, k0, conc_o, conc_r,
                                      alpha, area, n, e_formal)
    return i_a + i_c


def butler_volmer_from_i0(eta, i0, alpha=0.5):
    """Butler-Volmer current written in terms of the exchange current.

        i = i0 [ exp((1-alpha) f eta) - exp(-alpha f eta) ]

    This is the special case in which mass transport does not matter, so the
    surface concentrations equal the bulk ones and the whole current-potential
    relationship collapses onto the single parameter i0 (Lesson 12).

    Two limits are worth remembering:

        eta much less than 50 mV    i = i0 f eta            (linear)
        eta much more than 50 mV    ln|i| = ln(i0) + (1-alpha) f eta   (Tafel)

    Parameters
    ----------
    eta : float or array, V
    i0 : float, A
    alpha : float, dimensionless

    Returns
    -------
    float or array, A.  Positive is anodic.
    """
    eta = np.asarray(eta)
    return i0 * (np.exp((1 - alpha) * f * eta) - np.exp(-alpha * f * eta))


# =============================================================================
# Marcus theory: what Butler-Volmer throws away (Lesson 11)
# =============================================================================
#
# Butler-Volmer replaces the two potential energy surfaces with straight lines
# and is then forced to treat alpha as an empirical constant.  Marcus keeps the
# parabolas.  Writing the reaction coordinate x so that the reactant well sits
# at x = 0 and the product well at x = 1, with both wells of curvature lambda::
#
#     G_O(x) = lambda x^2
#     G_R(x) = lambda (x - 1)^2 + dG
#
# Setting these equal gives the crossing point, and it is worth staring at:
#
#     x* = (lambda + dG) / (2 lambda) = alpha
#
# The transfer coefficient is not a fudge factor at all.  It is the POSITION OF
# THE TRANSITION STATE along the reaction coordinate, and it moves when you
# change the driving force.  Everything below follows from those three lines.

def marcus_barrier(potential, reorganization_energy, e_formal=0.0):
    """Activation barrier for a reduction, from Marcus theory.

        dG    = (E - E0')                        eV, the driving force
        dG(a) = (lambda + dG)^2 / (4 lambda)     eV, the barrier

    At dG = 0 the barrier is lambda/4.  As the reaction is driven harder the
    barrier shrinks, reaches ZERO at dG = -lambda, and then GROWS AGAIN: the
    Marcus inverted region, where making a reaction more favourable makes it
    slower.  Butler-Volmer cannot produce that behaviour at all, because a
    straight line has no minimum.

    Parameters
    ----------
    potential : float or array, V.
    reorganization_energy : float, eV.  Typically 0.5 to 1.5 eV in water.
    e_formal : float, V.

    Returns
    -------
    float or array, eV.
    """
    driving_force = np.asarray(potential) - e_formal
    return (reorganization_energy + driving_force)**2 / (4.0 * reorganization_energy)


def marcus_alpha(potential, reorganization_energy, e_formal=0.0):
    """Transfer coefficient predicted by Marcus theory.

        alpha = 1/2 + (E - E0') / (2 lambda)

    This is the derivative of :func:`marcus_barrier` with respect to the driving
    force, and it is the same quantity the Butler-Volmer treatment holds fixed
    at 0.5 (Lesson 11).  Three things fall out of it:

    * alpha = 0.5 exactly at E = E0', which is why the assumption works so well
      near equilibrium and why measured values cluster there.
    * alpha drifts linearly away from 0.5 as the reaction is driven, at a rate
      set by 1/(2 lambda).  A large reorganization energy makes alpha look
      constant over a wide window, which is why the approximation survived.
    * The experimentally reported range of roughly 0.3 to 0.7 corresponds to
      driving forces within about +/- 0.4 lambda of the formal potential, which
      is the range most measurements are made over.

    Values are NOT clipped to [0, 1].  In the inverted region alpha genuinely
    goes negative, and that is the whole point: a negative transfer coefficient
    is what "more driving force, slower reaction" looks like in the algebra.
    Butler-Volmer, which fixes alpha at a positive constant, cannot express it.

    Parameters
    ----------
    potential : float or array, V.
    reorganization_energy : float, eV.
    e_formal : float, V.

    Returns
    -------
    float or array, dimensionless.
    """
    driving_force = np.asarray(potential) - e_formal
    return 0.5 + driving_force / (2.0 * reorganization_energy)


def marcus_surfaces(x, reorganization_energy, driving_force):
    """The two parabolic free energy surfaces, for plotting.

    Returns ``(g_oxidized, g_reduced)`` in eV on the reaction coordinate ``x``,
    where x = 0 is the reactant well and x = 1 is the product well::

        G_O(x) = lambda x^2
        G_R(x) = lambda (x - 1)^2 + dG

    Parameters
    ----------
    x : array, dimensionless reaction coordinate.
    reorganization_energy : float, eV.
    driving_force : float, eV.

    Returns
    -------
    (array, array), both eV.
    """
    x = np.asarray(x)
    return (reorganization_energy * x**2,
            reorganization_energy * (x - 1.0)**2 + driving_force)


def charge_transfer_resistance(i0, n=1, temperature=T):
    """Charge transfer resistance, the low-overpotential face of the kinetics.

        R_ct = RT / (n F i0)        ohms

    Very close to equilibrium both exponentials in the Butler-Volmer equation
    can be linearised, and the interface stops behaving like a diode and starts
    behaving like a plain resistor (Lesson 12).  That is a gift: it means the
    exchange current can be measured from a tiny perturbation, without driving
    the reaction hard enough to disturb the surface or run into mass transport.

    Two things follow, and both matter later:

    * A fast couple has a large i0 and therefore a SMALL R_ct, which is what
      makes an interface non-polarizable.  This is the quantitative version of
      the reference electrode requirement from Lesson 10.
    * R_ct is the resistor sitting in parallel with the double layer
      capacitance in the Randles circuit, so it is the quantity an impedance
      measurement actually returns (Lesson 31).

    Take care not to confuse it with the uncompensated solution resistance
    R_u from Lesson 7.  R_u is a property of the electrolyte and the cell
    geometry; R_ct is a property of the reaction.  A measurement sees them in
    series and cannot separate them without help.

    Parameters
    ----------
    i0 : float or array, A
    n : int
    temperature : float, K

    Returns
    -------
    float or array, ohms.
    """
    return R * temperature / (n * F * np.asarray(i0))


def tafel_slope(apparent_alpha, n=1):
    """Tafel slope from an apparent transfer coefficient.

        b = 2.303 RT / (alpha_app n F)      volts per decade of current

    The point of writing it this way is that **alpha_app is not the symmetry
    factor**.  For a single one-electron step the two coincide and b is about
    118 mV/dec at 298 K.  For a multi-step reaction, alpha_app collects the
    electrons transferred before the rate determining step as well:

        alpha_app = n_prior + beta          (electrochemical rate determining step)
        alpha_app = n_prior                 (chemical rate determining step)

    So a measured slope constrains the mechanism, but it does not identify one
    on its own (Lesson 13).

    Parameters
    ----------
    apparent_alpha : float or array, dimensionless
    n : int

    Returns
    -------
    float or array, V per decade.  Multiply by 1000 for mV/dec.
    """
    return 2.303 * THERMAL_VOLTAGE / (np.asarray(apparent_alpha) * n)


#: Mechanisms for the hydrogen evolution reaction in acid, with the apparent
#: transfer coefficient each rate determining step produces at low coverage.
#: beta = 0.5 is assumed for every electrochemical step.
#:
#:     Volmer      H+ + e- + *   ->  H*
#:     Heyrovsky   H* + H+ + e-  ->  H2 + *
#:     Tafel       2 H*          ->  H2 + 2*
HER_MECHANISMS = {
    'Volmer rate determining': 0.5,         # no prior electron transfer, beta = 0.5
    'Heyrovsky rate determining': 1.5,      # one prior electron transfer, plus beta
    'Tafel rate determining': 2.0,          # chemical step, two prior electron transfers
}


def langmuir_coverage(potential, e_adsorption=0.0, n=1):
    """Fractional coverage of an adsorbed intermediate, Langmuir isotherm.

        theta = 1 / (1 + exp(n f (E - E_ads)))

    Coverage saturates at 1: the surface runs out of sites.  That saturation is
    what makes a Tafel plot bend over at high overpotential even when nothing is
    wrong with the mass transport, and it is why a slope measured over too wide a
    window is not a slope at all (Lesson 13).

    Parameters
    ----------
    potential : float or array, V
    e_adsorption : float, V
        Potential at which the surface is half covered.
    n : int

    Returns
    -------
    float or array, dimensionless, between 0 and 1.
    """
    return 1.0 / (1.0 + np.exp(n * f * (np.asarray(potential) - e_adsorption)))


#: Reaction order in H+ for each hydrogen evolution mechanism, measured at a
#: FIXED POTENTIAL ON THE SHE SCALE and at low coverage (Lesson 13):
#:
#:     rho = (d log i / d log a_H+) at constant E
#:
#: Volmer rate determining consumes one proton in the slow step and nothing
#: before it.  Heyrovsky consumes one in the slow step and one in the
#: pre-equilibrium, so it goes as the square.  Tafel consumes none in the slow
#: step, but two protons sit in the two pre-equilibrated Volmer steps.
#:
#: Pair these with :data:`HER_MECHANISMS`: the Tafel slope alone cannot
#: separate Volmer from Heyrovsky-plus-coverage-effects, but the slope and the
#: reaction order together give a two-dimensional fingerprint.
HER_REACTION_ORDERS = {
    'Volmer rate determining': 1.0,
    'Heyrovsky rate determining': 2.0,
    'Tafel rate determining': 2.0,
}


def reaction_order_on_rhe(order_on_she, apparent_alpha):
    """Convert an H+ reaction order from the SHE scale to the RHE scale.

        rho(RHE) = rho(SHE) - alpha_app

    The RHE scale already moves with pH, by 2.303 RT/F = 59 mV per unit, so
    holding the overpotential fixed against RHE is *not* the same experiment as
    holding the potential fixed against SHE.  Some of the proton dependence you
    were trying to measure has been absorbed into the reference (Lesson 13).

    The consequence is worth knowing before you design the experiment: on the
    RHE scale a Volmer-limited and a Heyrovsky-limited mechanism both return
    0.5, and the diagnostic that separated them is gone.

    Parameters
    ----------
    order_on_she : float, from :data:`HER_REACTION_ORDERS`.
    apparent_alpha : float, from :data:`HER_MECHANISMS`.

    Returns
    -------
    float, the order you would measure at fixed overpotential vs RHE.
    """
    return order_on_she - apparent_alpha


def her_microkinetic(eta, delta_g_ads=0.0, log_k_ratio=0.0, beta=0.5,
                     temperature=T):
    """Steady-state Volmer-Heyrovsky current WITHOUT a rate determining step.

    Everything else in this module that touches multi-step kinetics assumes one
    step is slow and the rest are equilibrated.  This function does not.  It
    solves the two-step mechanism ::

        Volmer      H+ + e- + *   <->  H*        k1 forward, km1 reverse
        Heyrovsky   H* + H+ + e-   ->  H2 + *    k2

    for the steady-state coverage, by setting the rate of accumulation of H* to
    zero.  With ``a = exp(-beta f eta)`` and ``b = exp((1-beta) f eta)`` ::

        theta = k1 a / (k1 a + km1 b + k2 a)
        i     = 2 k2 theta a

    The rate constants are tied to the binding energy through a
    Bronsted-Evans-Polanyi relation, which is the "scaling relation" of
    Lesson 13 written for one surface ::

        k1  = exp(-max( delta_g, 0) / kT)     hard to make H* if binding is weak
        km1 = exp(-max(-delta_g, 0) / kT)     hard to remove H* if binding is strong
        k2  = 10**log_k_ratio * km1

    Note that k1/km1 = exp(-delta_g/kT), so the equilibrium coverage is
    thermodynamically correct no matter where the barriers are put.

    Two things fall out of this one function, and they are the same physics:

    * At fixed ``delta_g_ads``, sweeping ``eta`` gives a Tafel plot whose
      LOCAL SLOPE is a curve, not a number.  It runs from about 40 mV/dec where
      the Volmer step is equilibrated and the coverage is low, up towards
      118 mV/dec once the surface saturates and Volmer regains control.  Feed
      the result to :func:`local_tafel_slope` to see it.
    * At fixed ``eta``, sweeping ``delta_g_ads`` gives a volcano peaked at
      ``delta_g_ads = 0``.  The weak-binding branch is Volmer-limited and the
      strong-binding branch is Heyrovsky-limited, which is the Sabatier
      principle arriving as an output rather than an assumption.

    Parameters
    ----------
    eta : float or array, V.  Negative for cathodic HER.
    delta_g_ads : float or array, eV.  Hydrogen binding free energy relative to
        the optimum.  Positive is too weak, negative is too strong.
    log_k_ratio : float.  log10 of the Heyrovsky rate constant relative to the
        reverse Volmer rate constant.  Negative makes Heyrovsky slow, which is
        the condition for a Volmer pre-equilibrium.
    beta : float, symmetry factor of each electrochemical step.
    temperature : float, K.

    Returns
    -------
    (current, coverage) : both float or array, matching the broadcast shape of
        ``eta`` and ``delta_g_ads``.  The current is in arbitrary units; only
        its logarithmic slope is meaningful.
    """
    eta = np.asarray(eta, dtype=float)
    delta_g = np.asarray(delta_g_ads, dtype=float)
    kt_ev = R * temperature / F          # thermal voltage in V is kT/e in eV

    k1 = np.exp(-np.maximum(delta_g, 0.0) / kt_ev)
    km1 = np.exp(-np.maximum(-delta_g, 0.0) / kt_ev)
    k2 = (10.0 ** log_k_ratio) * km1

    f_local = F / (R * temperature)
    a = np.exp(-beta * f_local * eta)
    b = np.exp((1.0 - beta) * f_local * eta)

    coverage = k1 * a / (k1 * a + km1 * b + k2 * a)
    current = 2.0 * k2 * coverage * a
    return current, coverage


def local_tafel_slope(eta, current):
    """Tafel slope evaluated point by point, rather than fitted over a window.

        b(eta) = -1 / ( d log10|i| / d eta )        volts per decade

    This is the honest version of a Tafel slope.  A real analysis fits a
    straight line over a chosen potential window and reports one number; this
    function reports what that number would have been at every potential, which
    makes it obvious when the choice of window is doing the work (Lesson 13).

    Parameters
    ----------
    eta : array, V, monotonic.
    current : array, same shape, any units.  Sign is ignored.

    Returns
    -------
    array, V per decade.  Multiply by 1000 for mV/dec.  Values are positive for
    a cathodic branch swept towards negative potential.
    """
    eta = np.asarray(eta, dtype=float)
    log_current = np.log10(np.abs(np.asarray(current, dtype=float)))
    return -1.0 / np.gradient(log_current, eta)



def current_overpotential(eta, i_lim_over_i0, alpha=0.5, conc_ratio=1.0):
    """Current-overpotential behaviour, normalised by the limiting current.

    This is the equation that carries thermodynamics, kinetics AND mass
    transport at the same time (Lessons 14, 20, 21, 30)::

                    exp((1-a) f eta) - exp(-a f eta)
        i/i_lim = ---------------------------------------------------------
                  (i_lim/i0) + (1+m) exp(-a f eta) + (1+1/m) exp((1-a) f eta)

    The single parameter ``i_lim_over_i0`` decides which regime you are in:

        small (say 1)      kinetics are fast; the wave is Nernstian (reversible)
        large (say 1e5)    kinetics are slow; the wave is drawn out (irreversible)
        in between         quasireversible

    Parameters
    ----------
    eta : float or array, V
        Overpotential, E - E_eq.
    i_lim_over_i0 : float
        Ratio of the mass-transport limited current to the exchange current.
    alpha : float, dimensionless
    conc_ratio : float
        m = C_R* / C_O*.  Must not be zero (the equation contains 1/m).

    Returns
    -------
    float or array, dimensionless, i/i_lim.  Positive is anodic.
    """
    if conc_ratio == 0:
        raise ValueError(
            'conc_ratio (C_R*/C_O*) cannot be zero: the equation contains 1/m. '
            'Use a small non-zero value instead.')
    m = conc_ratio
    eta = np.asarray(eta)
    anodic = np.exp((1 - alpha) * f * eta)
    cathodic = np.exp(-alpha * f * eta)
    return (anodic - cathodic) / (i_lim_over_i0 + (1 + m) * cathodic + (1 + 1 / m) * anodic)


def quasireversible_step_current(eta, k0, conc_o, conc_r, diff_o, diff_r,
                                 t, alpha=0.5, area=1.0, n=1):
    """Current after a potential step when kinetics are NOT fast (Lesson 21).

        i(t) = n F A (k_a C_R* - k_c C_O*) * erfcx(H sqrt(t))

        H = k_c / sqrt(D_O) + k_a / sqrt(D_R)

    The textbook writes the last factor as exp(H^2 t) erfc(H sqrt(t)); that is
    what :func:`scipy.special.erfcx` computes, without overflowing.

    The H sqrt(t) group is the whole story.  When it is small the system looks
    reversible; when it is large the exponential decay of the kinetic term takes
    over and the response is irreversible.

    Parameters
    ----------
    eta : float or array, V (overpotential)
    k0 : float, cm s^-1
    conc_o, conc_r : float, mol cm^-3
    diff_o, diff_r : float, cm^2 s^-1
    t : float, s
    alpha : float
    area : float, cm^2
    n : int

    Returns
    -------
    float or array, A.  Positive is anodic.
    """
    k_a, k_c = rate_constants(eta, k0, alpha, e_formal=0.0)
    H = k_c / np.sqrt(diff_o) + k_a / np.sqrt(diff_r)
    return n * F * area * (k_a * conc_r - k_c * conc_o) * erfcx(H * np.sqrt(t))


# =============================================================================
# SECTION 6: Mass transport (Lessons 14, 16, 19, 20, 23, 24)
# =============================================================================

def mass_transfer_coefficient(diff_coeff, t):
    """Mass transfer coefficient m_j for semi-infinite linear diffusion.

        m_j = sqrt(D_j / (pi t))

    This is the D/delta of Lesson 14 with the diffusion layer thickness filled
    in from the Cottrell result.  It has units of a velocity, cm s^-1, which is
    the giveaway that it is a rate of delivery per unit concentration.

    Note that it DECREASES with time: the diffusion layer grows, so delivery
    slows down.  Limiting currents are not constants.

    Parameters
    ----------
    diff_coeff : float, cm^2 s^-1
    t : float or array, s

    Returns
    -------
    float or array, cm s^-1.
    """
    return np.sqrt(diff_coeff / (np.pi * np.asarray(t)))


def cottrell(t, conc, diff_coeff=1e-5, area=1.0, n=1):
    """Diffusion-limited current after a potential step: the Cottrell equation.

        i_d(t) = n F A C* sqrt(D / (pi t))

    This is the MAGNITUDE of the mass-transport limited current, always
    positive.  Give it a sign at the point of use according to whether the
    reaction being driven is an oxidation or a reduction.

    Parameters
    ----------
    t : float or array, s.  Must be greater than zero.
    conc : float, mol cm^-3 (bulk concentration of the species being consumed)
    diff_coeff : float, cm^2 s^-1
    area : float, cm^2
    n : int

    Returns
    -------
    float or array, A.
    """
    return n * F * area * conc * mass_transfer_coefficient(diff_coeff, t)


def sampled_current(potential, t, conc_total, diff_coeff=1e-5, area=1.0,
                    n=1, e_formal=0.0, starting_species='R'):
    """Current measured at time ``t`` after a step to ``potential``.

    The Cottrell current tells you the biggest current the solution can supply.
    The Nernst equation tells you what fraction of that the applied potential
    actually demands.  Multiply them::

        i(E, t) = fraction_oxidized(E) * i_d(t)

    This is the calculation behind sampled current voltammetry (Lesson 19),
    normal pulse voltammetry (Lesson 23) and the shape of a linear sweep
    (Lesson 25): the wave has the shape of the Nernst equation and the height
    of the Cottrell equation.

    ``starting_species`` selects which experiment you are describing:

        'R' (default)  solution starts entirely as R.  Sweeping positive drives
                       an oxidation; the current is ANODIC (positive) and the
                       wave rises with E.  Used in Lessons 23, 25 and 27.
        'O'            solution starts entirely as O.  Sweeping negative drives
                       a reduction; the current is CATHODIC (negative) and the
                       wave falls with E.  Used in Lesson 19.

    NOTE: the two cases are NOT related by a simple change of sign.  The
    reduction wave is built from the fraction REDUCED, so it mirrors in E as
    well as in sign; negating the 'R' result alone would put the plateau at
    positive potentials, which is backwards.

    Parameters
    ----------
    potential : float or array, V
    t : float or array, s
    conc_total : float, mol cm^-3 (C_O* + C_R*)
    diff_coeff : float, cm^2 s^-1
    area : float, cm^2
    n : int
    e_formal : float, V
    starting_species : {'R', 'O'}

    Returns
    -------
    float or array, A.  Anodic positive, so 'O' returns negative values.
    """
    species = str(starting_species).upper()
    if species not in ('R', 'O'):
        raise ValueError("starting_species must be 'R' or 'O', "
                         f'not {starting_species!r}.')
    ceiling = cottrell(t, conc_total, diff_coeff, area, n)
    if species == 'R':
        return nernst_fraction_oxidized(potential, e_formal, n) * ceiling
    return -nernst_fraction_reduced(potential, e_formal, n) * ceiling


def diffusion_profile(x, diff_coeff, t):
    """Normalised concentration profile of the species being consumed.

        C(x, t) / C* = erf( x / (2 sqrt(D t)) )

    Zero at the electrode surface (the potential step consumed everything
    there) and rising to the bulk value far away.

    Parameters
    ----------
    x : float or array, cm (distance from the electrode surface)
    diff_coeff : float, cm^2 s^-1
    t : float, s

    Returns
    -------
    float or array, dimensionless, between 0 and 1.
    """
    return erf(np.asarray(x) / (2.0 * np.sqrt(diff_coeff * t)))


def diffusion_layer_thickness(diff_coeff, t, multiplier=6):
    """Approximate diffusion layer thickness, as a multiple of sqrt(D t).

    The diffusion layer has no sharp edge, so "thickness" means "the distance
    that captures most of the perturbation":

        2 sqrt(D t)   captures  84 %
        4 sqrt(D t)   captures  95 %
        6 sqrt(D t)   captures  99.998 %

    See page 165 of Bard & Faulkner.

    Returns
    -------
    float or array, cm.
    """
    return multiplier * np.sqrt(diff_coeff * np.asarray(t))


def ume_steady_state_current(conc, diff_coeff, radius, n=1):
    """Steady state current at an ultramicrodisk electrode (Lesson 16).

        i_ss = 4 n F D C* r0

    The current does not decay to zero.  Radial diffusion keeps resupplying the
    electrode, and the diffusion layer stops growing once it is large compared
    with the electrode itself.

    Note that it is proportional to the RADIUS, not the area.  Shrinking the
    electrode costs you less current than you would expect.

    Parameters
    ----------
    conc : float, mol cm^-3
    diff_coeff : float, cm^2 s^-1
    radius : float, cm
    n : int

    Returns
    -------
    float, A.
    """
    return 4 * n * F * diff_coeff * conc * radius


def ume_current_function(tau):
    """Shoup-Szabo current function f(tau) for an ultramicrodisk electrode.

        i(tau) = i_ss * f(tau)
        f(tau) = 0.7854 + 0.8862 tau^(-1/2) + 0.2146 exp(-0.7823 tau^(-1/2))

    with the dimensionless time tau = 4 D t / r0^2.

    Reading the three terms tells you the three time domains of Lesson 16:

        the constant 0.7854            never decays -> the steady state
        the tau^(-1/2) term            the Cottrell-like short time behaviour
        the exponential term           the crossover between them

    f(tau) tends to 0.7854 + 0.2146 = 1 at long times, which is why i_ss is the
    natural unit to normalise by.

    Parameters
    ----------
    tau : float or array, dimensionless.

    Returns
    -------
    float or array, dimensionless, i/i_ss.
    """
    inv_sqrt = 1.0 / np.sqrt(np.asarray(tau))
    return 0.7854 + 0.8862 * inv_sqrt + 0.2146 * np.exp(-0.7823 * inv_sqrt)


def macrodisk_current_function(tau):
    """Cottrell current at a macrodisk, on the same axes as ``ume_current_function``.

    Normalising the Cottrell equation by the ultramicroelectrode steady state
    current, and writing it in terms of the same dimensionless time
    tau = 4 D t / r0^2, gives::

        i / i_ss = sqrt(pi) / (2 sqrt(tau)) = 0.8862 / sqrt(tau)

    which is exactly the second term of the Shoup-Szabo function.  The two
    electrodes behave identically at short times, which is the point of the
    comparison figure in Lesson 16.

    Returns
    -------
    float or array, dimensionless, i/i_ss.
    """
    return np.sqrt(np.pi) / (2.0 * np.sqrt(np.asarray(tau)))


# =============================================================================
# Pulse trains: NPV, staircase, DPV and SWV (Lessons 23 and 24)
# =============================================================================
#
# Everything above treats ONE potential step.  Every technique in Lessons 23
# to 25 applies a whole sequence of them, and the diffusion layer left behind
# by each step is still there when the next one arrives.
#
# The fix is superposition.  Diffusion is linear, so the response to a train of
# steps is the sum of the responses to each step taken on its own, each one
# aged by the time since it happened::
#
#     i(t) = n F A C* sqrt(D/pi) * SUM_j  (x_j - x_j-1) / sqrt(t - T_j)
#
# where step j begins at T_j and takes the surface to a converted fraction x_j
# set by the Nernst equation.  For a single step this collapses back to
# ``sampled_current``: one term, x_1 - x_0, aged by tau.  That is the whole
# content of these functions, and it is why NPV, staircase, DPV and SWV are all
# built from the same three lines of code below.
#
# The model assumes a REVERSIBLE couple and equal diffusion coefficients.


def _converted_fraction(potential, e_formal=0.0, n=1, starting_species='R',
                        temperature=T):
    """Fraction of the couple converted at the surface by a Nernstian electrode.

    'R' means the solution starts as R and the electrode oxidises it, so the
    converted fraction is the fraction oxidised; 'O' is the mirror image.
    """
    species = str(starting_species).upper()
    if species == 'R':
        return nernst_fraction_oxidized(potential, e_formal, n, temperature)
    if species == 'O':
        return nernst_fraction_reduced(potential, e_formal, n, temperature)
    raise ValueError("starting_species must be 'R' or 'O', "
                     f'not {starting_species!r}.')


def _sign_of(starting_species):
    """+1 for an oxidation, -1 for a reduction (anodic-positive convention)."""
    return 1.0 if str(starting_species).upper() == 'R' else -1.0


def step_train_current(potentials, durations, tau, conc_total, diff_coeff=1e-5,
                       area=1.0, n=1, e_formal=0.0, starting_species='R'):
    """Current sampled part-way into each step of an arbitrary staircase.

    This is :func:`sampled_current` generalised from one step to a whole
    programme of them, by the superposition argument in the comment above.
    Hand it any sequence of potentials and it returns the current you would
    measure ``tau`` seconds into each one, with every earlier step's leftover
    diffusion layer accounted for.

    NPV, staircase, DPV and SWV differ only in the programme you hand it.

    Parameters
    ----------
    potentials : array, V
        The potential applied during each step, in the order applied.
    durations : float or array, s
        How long each step is held.  A single number holds them all equally.
    tau : float or array, s
        Sampling time measured from the START of each step.  Must be greater
        than zero and no larger than that step's duration.
    conc_total : float, mol cm^-3 (C_O* + C_R*)
    diff_coeff : float, cm^2 s^-1
    area : float, cm^2
    n : int
    e_formal : float, V
    starting_species : {'R', 'O'}
        See :func:`sampled_current`.

    Returns
    -------
    array, A, one current per step.  Anodic positive.
    """
    potentials = np.atleast_1d(np.asarray(potentials, dtype=float))
    steps = potentials.size
    durations = np.broadcast_to(np.asarray(durations, dtype=float),
                                (steps,)).astype(float)
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (steps,)).astype(float)

    if np.any(durations <= 0):
        raise ValueError('Every step must last longer than zero seconds.')
    if np.any(tau <= 0) or np.any(tau > durations):
        raise ValueError('Each sampling time tau must fall inside its own step: '
                         '0 < tau <= duration.')

    # When each step starts, and when each one is sampled.
    start = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    sample = start + tau

    # The jump in surface composition demanded at each step.  x_0 = 0: before
    # the experiment begins the surface is at its bulk composition.
    fraction = _converted_fraction(potentials, e_formal, n, starting_species)
    jump = np.diff(np.concatenate(([0.0], fraction)))

    # age[k, j] is how long step j has had to age by the time step k is
    # sampled.  Only steps that have already happened (j <= k) contribute.
    age = sample[:, None] - start[None, :]
    happened = np.tril(np.ones((steps, steps), dtype=bool))
    age = np.where(happened, age, 1.0)          # dummy value, masked out next
    terms = np.where(happened, jump[None, :] / np.sqrt(age), 0.0)

    ceiling = n * F * area * conc_total * np.sqrt(diff_coeff / np.pi)
    return _sign_of(starting_species) * ceiling * terms.sum(axis=1)


def step_train_profile(x, potentials, durations, t_eval, diff_coeff=1e-5, n=1,
                       e_formal=0.0, starting_species='R'):
    """Concentration profile of the consumed species part-way through a train.

        C(x, t) / C* = 1 - SUM_j (x_j - x_j-1) erfc( x / (2 sqrt(D (t - T_j))) )

    The same superposition as :func:`step_train_current`, asked for the
    concentration rather than the current.  For a single step to complete
    depletion it collapses to :func:`diffusion_profile`.

    Use it to see what a rest period actually does.  A Nernstian surface
    concentration snaps back the instant the potential returns to E_i, so the
    surface value tells you nothing; the leftover depletion is out in the
    solution, and this is how you look at it.

    Parameters
    ----------
    x : float or array, cm (distance from the electrode surface)
    potentials, durations : see :func:`step_train_current`
    t_eval : float, s, measured from the start of the programme.  Evaluating
        exactly on a step boundary reports the state at the END of the step
        that is finishing.
    diff_coeff : float, cm^2 s^-1
    n : int
    e_formal : float, V
    starting_species : {'R', 'O'}

    Returns
    -------
    float or array, dimensionless, C/C* of the species being consumed.
    """
    x = np.asarray(x, dtype=float)
    potentials = np.atleast_1d(np.asarray(potentials, dtype=float))
    steps = potentials.size
    durations = np.broadcast_to(np.asarray(durations, dtype=float),
                                (steps,)).astype(float)

    start = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    fraction = _converted_fraction(potentials, e_formal, n, starting_species)
    jump = np.diff(np.concatenate(([0.0], fraction)))

    # Rounding in the cumulative start times can leave a step that begins
    # exactly at t_eval looking as though it began a femtosecond earlier.  Such
    # a step has had no time to do anything, but erfc(x/0) would let it wipe out
    # the surface, so require an age that is actually meaningful.
    tolerance = 1e-9 * durations.min()

    profile = np.ones_like(x, dtype=float)
    for step in range(steps):
        age = t_eval - start[step]
        if age <= tolerance or jump[step] == 0.0:
            continue
        profile = profile - jump[step] * erfc(x / (2.0 * np.sqrt(diff_coeff * age)))
    return profile


def pulse_train_program(potentials, t_pulse, t_rest=0.0, e_rest=None):
    """Build the step programme for normal pulse or staircase voltammetry.

    Set ``t_rest`` to zero and you get a staircase: the same pulses with the
    rest periods deleted.  That one argument is the entire difference between
    the two experiments in Lesson 23.

    Parameters
    ----------
    potentials : array, V, the pulse potentials, in the order applied.
    t_pulse : float, s
    t_rest : float, s.  Zero gives a staircase.
    e_rest : float, V.  The potential rested at between pulses; defaults to the
        first pulse potential, which is normally E_i.

    Returns
    -------
    (potentials, durations, sampled) arrays ready to hand to
    :func:`step_train_current`.  ``sampled`` is a boolean mask picking out the
    steps whose current is the analytical signal, so that::

        e, d, sampled = pulse_train_program(...)
        wave = step_train_current(e, d, tau, ...)[sampled]
    """
    potentials = np.atleast_1d(np.asarray(potentials, dtype=float))
    pulses = potentials.size

    if t_rest <= 0:
        return (potentials,
                np.full(pulses, float(t_pulse)),
                np.ones(pulses, dtype=bool))

    if e_rest is None:
        e_rest = potentials[0]

    program = np.empty(2 * pulses)
    program[0::2] = e_rest
    program[1::2] = potentials

    durations = np.empty(2 * pulses)
    durations[0::2] = float(t_rest)
    durations[1::2] = float(t_pulse)

    sampled = np.zeros(2 * pulses, dtype=bool)
    sampled[1::2] = True
    return program, durations, sampled


def dpv_response(potentials, e_pulse=0.05, t_step=0.5, t_pulse=0.05,
                 tau_s=None, tau_p=None, conc_total=1e-6, diff_coeff=1e-5,
                 area=1.0, n=1, e_formal=0.0, starting_species='R'):
    """Differential pulse voltammetry: the wave measured twice and subtracted.

        delta_i = i(tau_p) - i(tau_s)

    Each cycle sits at a staircase potential for ``t_step`` and is sampled at
    ``tau_s``, then pulses by ``e_pulse`` for ``t_pulse`` and is sampled again
    at ``tau_p``.  Both currents come from :func:`step_train_current`, so every
    earlier cycle is still in the answer: that is why the raw currents look so
    bad and the difference looks so good.

    The peak of ``delta_i`` sits at::

        E_p = E_1/2 - e_pulse / 2

    measured against the STAIRCASE potential, and for small ``e_pulse`` it is
    3.52 RT/nF wide at half height (90.4/n mV at 298 K).

    Parameters
    ----------
    potentials : array, V, the staircase (scan) potentials.
    e_pulse : float, V, signed.  Positive for an anodic scan.
    t_step, t_pulse : float, s
    tau_s, tau_p : float, s.  Sampling times measured from the start of the
        step and of the pulse; both default to sampling at the very end.
    conc_total : float, mol cm^-3
    diff_coeff : float, cm^2 s^-1
    area : float, cm^2
    n : int
    e_formal : float, V
    starting_species : {'R', 'O'}

    Returns
    -------
    (delta_i, i_step, i_pulse), each an array in A, one entry per staircase
    potential.
    """
    potentials = np.atleast_1d(np.asarray(potentials, dtype=float))
    cycles = potentials.size
    tau_s = t_step if tau_s is None else tau_s
    tau_p = t_pulse if tau_p is None else tau_p

    program = np.empty(2 * cycles)
    program[0::2] = potentials
    program[1::2] = potentials + e_pulse

    durations = np.empty(2 * cycles)
    durations[0::2] = float(t_step)
    durations[1::2] = float(t_pulse)

    tau = np.empty(2 * cycles)
    tau[0::2] = float(tau_s)
    tau[1::2] = float(tau_p)

    current = step_train_current(program, durations, tau, conc_total,
                                 diff_coeff, area, n, e_formal,
                                 starting_species)
    i_step = current[0::2]
    i_pulse = current[1::2]
    return i_pulse - i_step, i_step, i_pulse


def swv_response(potentials, e_pulse=0.025, t_step=0.02, conc_total=1e-6,
                 diff_coeff=1e-5, area=1.0, n=1, e_formal=0.0,
                 starting_species='R'):
    """Square wave voltammetry, in the dimensionless currents of Lesson 24.

    Each staircase potential gets a forward half cycle at ``+e_pulse`` and a
    reverse half cycle at ``-e_pulse``, each sampled at its end.  Currents are
    returned as the dimensionless psi of Bard & Faulkner::

        psi = i sqrt(pi t_p) / (n F A sqrt(D) C*)

    with t_p the half cycle (pulse) width.  To turn one back into a current,
    multiply by ``n F A C* sqrt(D / (pi t_p))``.

    This is the REVERSIBLE model.  Real delta_psi_p also depends on the
    staircase step, and on kinetics once the couple is not fast; the
    quasireversible calculation needs a numerical solution of the integral
    equation and is beyond the scope of this course.  Use these curves as the
    reversible reference that measured data is judged against.

    Parameters
    ----------
    potentials : array, V, the staircase (scan) potentials.
    e_pulse : float, V.  The square wave AMPLITUDE, E_sw: the waveform swings
        e_pulse above and below the staircase, so the peak-to-peak height is
        twice this.
    t_step : float, s, the full cycle time.  Each half lasts t_step/2, and the
        square wave frequency is 1/t_step.
    conc_total : float, mol cm^-3
    diff_coeff : float, cm^2 s^-1
    area : float, cm^2
    n : int
    e_formal : float, V
    starting_species : {'R', 'O'}

    Returns
    -------
    (psi_forward, psi_reverse, delta_psi), dimensionless arrays, one entry per
    staircase potential.  delta_psi is the analytical signal.
    """
    potentials = np.atleast_1d(np.asarray(potentials, dtype=float))
    cycles = potentials.size
    half = float(t_step) / 2.0

    program = np.empty(2 * cycles)
    program[0::2] = potentials + e_pulse
    program[1::2] = potentials - e_pulse

    current = step_train_current(program, np.full(2 * cycles, half), half,
                                 conc_total, diff_coeff, area, n, e_formal,
                                 starting_species)

    scale = n * F * area * conc_total * np.sqrt(diff_coeff / (np.pi * half))
    psi_forward = current[0::2] / scale
    psi_reverse = current[1::2] / scale
    return psi_forward, psi_reverse, psi_forward - psi_reverse


# -----------------------------------------------------------------------------
# Non-Faradaic current.  Introduced in Lesson 7, and the reason square wave
# voltammetry works at all (Lesson 24).
# -----------------------------------------------------------------------------

def cell_time_constant(r_uncompensated, capacitance):
    """Cell time constant, tau_cell = R_u C_d, in seconds (Lesson 7).

    Charging current has fallen to 1% of its initial value after about
    5 tau_cell.  Sample the Faradaic current before then and you are measuring
    the double layer as much as the reaction.

    Parameters
    ----------
    r_uncompensated : float, ohm
    capacitance : float, F

    Returns
    -------
    float, s.  Check the units: ohm * farad = second.
    """
    return r_uncompensated * capacitance


def charging_current(t, e_step, r_uncompensated, capacitance):
    """Double layer charging current after a potential step (Lesson 7).

        i_c(t) = (delta_E / R_u) exp( -t / (R_u C_d) )

    Compare its shape with the Faradaic current from :func:`cottrell`.  The
    charging current dies EXPONENTIALLY and the Faradaic current only as
    t^(-1/2), which is the entire reason any pulse technique waits before it
    samples.

    Parameters
    ----------
    t : float or array, s
    e_step : float, V, the size of the potential step
    r_uncompensated : float, ohm
    capacitance : float, F

    Returns
    -------
    float or array, A.  Carries the sign of ``e_step``.
    """
    tau_cell = cell_time_constant(r_uncompensated, capacitance)
    return (e_step / r_uncompensated) * np.exp(-np.asarray(t) / tau_cell)


# =============================================================================
# SECTION 7: Hydrodynamics (Lesson 30)
# =============================================================================

def rpm_to_rad(rpm):
    """Convert rotation rate from rpm to angular velocity in rad s^-1.

        omega = 2 pi f,  and  f = rpm / 60

    Rotators are set in rpm.  The Levich equation needs rad s^-1.  Forgetting
    this costs you a factor of sqrt(2 pi / 60) = 0.32 in every current you
    predict, so it is one of the most common errors in the field.
    """
    return 2.0 * np.pi * np.asarray(rpm) / 60.0


def rad_to_rpm(omega):
    """Convert angular velocity in rad s^-1 back to rpm.  See :func:`rpm_to_rad`."""
    return np.asarray(omega) * 60.0 / (2.0 * np.pi)


def levich(omega, conc, diff_coeff, viscosity=KINEMATIC_VISCOSITY['H2O'],
           area=1.0, n=1):
    """Mass transport limited current at a rotating disk electrode.

        i_lim = 0.62 n F A D^(2/3) omega^(1/2) nu^(-1/6) C*

    Rotation fixes the diffusion layer at a steady thickness instead of letting
    it grow, so unlike the Cottrell current this one does NOT decay with time.
    You set it, by choosing the rotation rate.

    Parameters
    ----------
    omega : float or array, rad s^-1.  Use :func:`rpm_to_rad` on instrument
        settings first.
    conc : float, mol cm^-3
    diff_coeff : float, cm^2 s^-1
    viscosity : float, cm^2 s^-1.  Defaults to water; see KINEMATIC_VISCOSITY.
    area : float, cm^2
    n : int

    Returns
    -------
    float or array, A (magnitude).
    """
    return (0.62 * n * F * area * np.power(diff_coeff, 2.0 / 3.0)
            * np.sqrt(np.asarray(omega)) * np.power(viscosity, -1.0 / 6.0) * conc)


def kinetic_current(rate_constant, conc, area=1.0, n=1):
    """Current if electron transfer kinetics were the ONLY limitation.

        i_K = n F A k(E) C*

    Has no dependence on rotation rate, which is precisely what lets a
    Koutecky-Levich plot separate it from the transport term.

    Parameters
    ----------
    rate_constant : float or array, cm s^-1 (potential dependent; see
        :func:`rate_constants`)
    conc : float, mol cm^-3
    area : float, cm^2
    n : int

    Returns
    -------
    float or array, A (magnitude).
    """
    return n * F * area * np.asarray(rate_constant) * conc


def koutecky_levich(i_kinetic, i_levich):
    """Combine kinetic and transport limitations in series.

        1/i = 1/i_K + 1/i_lim

    Resistances in series, in effect: the slower of the two dominates.  Plotting
    1/i against 1/sqrt(omega) gives a straight line whose intercept is 1/i_K,
    which is how kinetics get measured at an RDE.

    Both arguments and the result are in A.
    """
    return 1.0 / (1.0 / np.asarray(i_kinetic) + 1.0 / np.asarray(i_levich))


# =============================================================================
# SECTION 8: Finite diffusion and thin layers (Lesson 28)
# =============================================================================

def thin_layer_reversible(potential, scan_rate, volume, conc, n=1, e_formal=0.0):
    """Sweep voltammetry in a thin layer cell, reversible case.

        i = (n^2 F^2 nu V C*) / (RT) * theta / (1 + theta)^2
        theta = exp(n f (E - E0'))

    NOTE ON SIGN: this returns a POSITIVE current for the reduction, i.e.
    cathodic positive, matching the presentation in Lesson 28.  This is the
    exception to the anodic-positive convention used elsewhere in this file.
    Negate the result if you need anodic-positive output.

    The peak here does NOT come from mass transport.  It comes from running out
    of material: the cell holds a finite amount and the sweep consumes it.  The
    peak current is therefore proportional to nu, not to sqrt(nu), which is the
    diagnostic that distinguishes this case from semi-infinite diffusion.

    Peak current: i_p = n^2 F^2 nu V C* / (4 R T).

    Parameters
    ----------
    potential : float or array, V
    scan_rate : float, V s^-1
    volume : float, cm^3
    conc : float, mol cm^-3
    n : int
    e_formal : float, V

    Returns
    -------
    float or array, A.
    """
    theta = np.exp(n * f * (np.asarray(potential) - e_formal))
    return (n ** 2 * F ** 2 * scan_rate * volume * conc / (R * T)
            * theta / np.power(1 + theta, 2))


def thin_layer_irreversible(potential, scan_rate, volume, area, conc, k0,
                            alpha=0.5, n=1, e_formal=0.0):
    """Sweep voltammetry in a thin layer cell, irreversible case.

        i(E) = F A k0 C* exp(-a f (E-E0')) exp( -(A k0)/(a f V nu) exp(-a f (E-E0')) )

    NOTE ON SIGN: like :func:`thin_layer_reversible`, this returns a POSITIVE
    current for the reduction (cathodic positive), matching Lesson 28.

    The dimensionless group A k0 / (alpha f V nu) sets the behaviour: it
    compares how fast the reaction can go with how fast you are sweeping past
    it.

    Parameters and units follow :func:`thin_layer_reversible`, plus
    ``area`` in cm^2 and ``k0`` in cm s^-1.

    Returns
    -------
    float or array, A.
    """
    drive = np.exp(-alpha * f * (np.asarray(potential) - e_formal))
    return F * area * k0 * conc * drive * np.exp(-(area * k0) / (alpha * f * volume * scan_rate) * drive)


# =============================================================================
# SECTION 9: Galvanostatic techniques (Lessons 33 and 34)
# =============================================================================

def sand_time(current, conc, diff_coeff, area=1.0, n=1):
    """Transition (Sand) time: how long a constant current can be sustained.

        i sqrt(tau) = n F A sqrt(D pi) C* / 2

    rearranged for tau.  Past tau the electrode has exhausted the reactant at
    its surface and the potential runs away to whatever reaction comes next.
    Think of it as a redox titration in which you control the delivery rate.

    Parameters
    ----------
    current : float, A (magnitude)
    conc : float, mol cm^-3
    diff_coeff : float, cm^2 s^-1
    area : float, cm^2
    n : int

    Returns
    -------
    float, s.
    """
    return np.power(n * F * area * conc / (2.0 * current), 2) * diff_coeff * np.pi


def chronopotentiogram(t_over_tau, e_quarter=0.0, n=1):
    """Potential during a constant current experiment, as a function of t/tau.

        E = E_(tau/4) + (RT/nF) ln( sqrt(tau/t) - 1 )

    This is the Nernst equation again, wearing a different hat: the surface
    concentration ratio is being swept by the titration rather than by an
    applied potential.

    E_(tau/4), the potential at one quarter of the transition time, plays the
    same role E_(1/2) does in a voltammogram.  The spacing between t/tau = 0.25
    and t/tau = 0.75 is 47.9/n mV for a reversible couple, which is one of the
    two reversibility tests in Lesson 33.

    Parameters
    ----------
    t_over_tau : float or array, dimensionless, strictly between 0 and 1.
    e_quarter : float, V
    n : int

    Returns
    -------
    float or array, V.
    """
    t_over_tau = np.asarray(t_over_tau, dtype=float)
    return e_quarter + (R * T / (n * F)) * np.log(np.sqrt(1.0 / t_over_tau) - 1.0)


#: Expected potential difference between t/tau = 0.25 and 0.75 for a reversible
#: couple at 298 K, in V.  Divide by n.  See :func:`chronopotentiogram`.
CHRONOPOT_REVERSIBILITY_SPACING = (R * T / F) * (np.log(np.sqrt(1 / 0.25) - 1)
                                                 - np.log(np.sqrt(1 / 0.75) - 1))


def chronopotentiometry_profile(x, current, t, diff_coeff, conc,
                                area=1.0, n=1):
    """Concentration profile during a constant current experiment.

        C_O(x,t)/C_O* = 1 - i [ 2 sqrt(Dt/pi) exp(-x^2/(4Dt)) - x erfc(x/(2 sqrt(Dt))) ]
                                                    / (n F A D C_O*)

    Unlike a potential step, the surface concentration falls steadily rather
    than dropping instantly: you are titrating at a fixed rate.

    Parameters
    ----------
    x : float or array, cm
    current : float, A (magnitude)
    t : float, s
    diff_coeff : float, cm^2 s^-1
    conc : float, mol cm^-3
    area : float, cm^2
    n : int

    Returns
    -------
    float or array, dimensionless C/C*.
    """
    x = np.asarray(x, dtype=float)
    bracket = (2.0 * np.sqrt(diff_coeff * t / np.pi) * np.exp(-(x ** 2) / (4.0 * diff_coeff * t))
               - x * erfc(x / (2.0 * np.sqrt(diff_coeff * t))))
    return 1.0 - current * bracket / (n * F * area * diff_coeff * conc)


def c_rate_current(capacity, c_rate):
    """Current needed to charge or discharge a cell in 1/c_rate hours.

        i = capacity * c_rate

    "1C" means the full capacity in one hour, "2C" in half an hour, "C/10" in
    ten hours.  It is a rate expressed as a reciprocal time, which is why the
    same C-rate means a different current for every cell.

    Parameters
    ----------
    capacity : float, mA h (or mA h g^-1 for a specific capacity)
    c_rate : float, dimensionless

    Returns
    -------
    float, mA (or mA g^-1), matching the units of ``capacity``.
    """
    return capacity * c_rate


def coulombic_efficiency(charge_out, charge_in):
    """Fraction of the charge put in that comes back out, as a percentage.

    The single most diagnostic number in battery testing.  Anything below 100 %
    means charge went somewhere other than the reaction you wanted, and over
    hundreds of cycles a shortfall of a few tenths of a percent compounds into
    a dead cell.

    Both arguments in any consistent unit of charge.  Returns a percentage.
    """
    return 100.0 * np.asarray(charge_out) / np.asarray(charge_in)


def solid_solution_voltage(x, e_formal=3.8, n=1):
    """Cell voltage of an intercalation electrode that stays in ONE phase.

        E = E0' - (RT/nF) ln( x / (1 - x) )

    This is the Nernst equation with x, the fraction of lattice sites occupied,
    playing the part of a concentration ratio.  Because x changes continuously
    as the cell is charged, so does the voltage: a solid solution electrode has
    a **sloping** discharge curve.

    Parameters
    ----------
    x : float or array, fraction of sites occupied, strictly between 0 and 1.
    e_formal : float, V
    n : int

    Returns
    -------
    float or array, V.
    """
    x = np.asarray(x, dtype=float)
    return e_formal - (R * T / (n * F)) * np.log(x / (1.0 - x))


def two_phase_voltage(x, e_formal=3.8, plateau=(0.15, 0.85), n=1,
                      plateau_slope=0.02):
    """Cell voltage of an intercalation electrode that separates into TWO phases.

    While both phases coexist, Gibbs' phase rule fixes the chemical potential,
    so the voltage barely moves: the discharge curve is **flat** to the eye, and
    see ``plateau_slope`` for why it is not exactly flat here.  Outside the
    plateau only one phase is present, and the voltage follows the ordinary
    solid solution curve of :func:`solid_solution_voltage`.

    A flat plateau is therefore not a cosmetic detail. It is a direct readout of
    a first order phase transition inside the electrode, and it is the reason
    LiFePO4 and graphite behave so differently from LiCoO2 at high state of charge.

    The curve is built in three continuous pieces, which is the part worth
    understanding.  The single phase branches are NOT the unshifted solid
    solution curve.  A branch has to arrive at the plateau voltage exactly when
    its phase runs out, so each is offset to land on the plateau at its own end
    of it::

        alpha branch,  x < x1:   E = E(x1) - (RT/nF) [ ln(x/(1-x)) - ln(x1/(1-x1)) ]
        plateau,  x1 < x < x2:   E = e_formal, tilted by plateau_slope
        beta branch,   x > x2:   E = E(x2) - (RT/nF) [ ln(x/(1-x)) - ln(x2/(1-x2)) ]

    Physically the offset is just a statement that the two phases are in
    equilibrium at the boundary: the chemical potential cannot jump there.  Get
    this wrong and the discharge curve acquires two vertical steps at x1 and x2,
    which is a good way to spot the mistake in someone else's figure.

    Parameters
    ----------
    x : float or array, fraction of sites occupied, between 0 and 1.
    e_formal : float, V
        The plateau voltage, at the midpoint of the transition.
    plateau : (float, float), the x range (x1, x2) over which the two phases
        coexist.
    n : int
    plateau_slope : float, V
        How far the plateau falls from one end to the other, centred on
        ``e_formal``.  The default of 20 mV is not cosmetic.  A PERFECTLY flat
        plateau has an infinite dQ/dE, so differentiating one (Lesson 34)
        divides by zero and the curve never appears.  Real plateaus always
        carry a small tilt, from particle size distribution, from strain at the
        phase boundary and from the resistance of the cell, and that tilt is
        what makes a measured dQ/dE peak finite.  Pass 0.0 for the idealised
        flat plateau, but do not then differentiate it.

        The tilt is shaped as the inverse of a cumulative Gaussian, so that the
        plateau represents a normal spread of transition voltages of width
        plateau_slope / (2 * PLATEAU_SPREAD).  dQ/dE then comes back as that
        same Gaussian: a rounded peak of about 0.4 * plateau_slope at half
        height, rather than the needle a sharper tilt would produce.

    Returns
    -------
    float or array, V.
    """
    x = np.asarray(x, dtype=float)
    low, high = plateau
    thermal = R * T / (n * F)

    # Where the plateau begins and ends.  A tilted plateau starts slightly high
    # and finishes slightly low, and the branches must meet those two values.
    e_start = e_formal + plateau_slope / 2.0
    e_end = e_formal - plateau_slope / 2.0

    def _logit(value):
        return np.log(value / (1.0 - value))

    # The single phase branches: the solid solution curve, shifted so that each
    # one arrives exactly on the plateau as its phase is used up.
    alpha = e_start - thermal * (_logit(x) - _logit(low))
    beta = e_end - thermal * (_logit(x) - _logit(high))

    if plateau_slope == 0.0:
        middle = np.full_like(np.asarray(x, dtype=float), e_formal)
    else:
        # The shape of the tilt matters, and it is not free to choose.  Whatever
        # it is, the area under dQ/dE across the plateau is fixed: it is the
        # charge the transition carries.  The tilt decides only how that area is
        # distributed, and a straight tilt spreads it evenly, giving a flat
        # topped block rather than a peak.
        #
        # Real electrodes are not one particle, they are billions, each with its
        # own strain, size and contact resistance, so the transition happens at a
        # SPREAD of voltages rather than at one.  Take that spread to be Gaussian
        # with standard deviation sigma.  The charge converted by voltage E is
        # then the cumulative Gaussian, and inverting it gives the voltage at
        # which a given fraction has converted:
        #
        #     E(u) = e_formal - sigma * sqrt(2) * erfinv(2u - 1)
        #
        # with u the fraction of the way through the plateau.  Differentiate that
        # and dQ/dE comes back as the Gaussian itself: a smooth, rounded, finite
        # peak, which is what a real dQ/dE actually looks like.
        #
        # PLATEAU_SPREAD sets how far into the tails the plateau is taken to run.
        # At 2.5 sigma the slope where the plateau meets each single phase branch
        # happens to match the branch's own slope almost exactly, so dQ/dE comes
        # out continuous there rather than stepping.
        span = erfinv(erf(PLATEAU_SPREAD / np.sqrt(2.0)))
        u = (x - low) / (high - low)
        w = np.clip(2.0*u - 1.0, -1.0, 1.0) * erf(PLATEAU_SPREAD / np.sqrt(2.0))
        middle = e_formal - (plateau_slope / 2.0) * erfinv(w) / span

    # np.select rather than item assignment, so that a single x works too.
    return np.select([x < low, x > high], [alpha, beta], default=middle)


# =============================================================================
# SECTION 10: Coupled chemical reactions (Lesson 22)
# =============================================================================

def pcet_half_wave(pH, e_formal=0.0, m_over_n=1.0):
    """Half wave potential of a proton-coupled electron transfer, versus pH.

        E_(1/2) = E0' - (m/n) (2.303 RT/F) pH

    which at 298 K is a slope of -59.2 (m/n) mV per pH unit.  These are the
    sloping lines of a Pourbaix diagram; the ratio m/n of protons to electrons
    is read straight off the slope.

    Parameters
    ----------
    pH : float or array
    e_formal : float, V
    m_over_n : float, protons transferred per electron transferred.

    Returns
    -------
    float or array, V.
    """
    return e_formal - m_over_n * 2.303 * THERMAL_VOLTAGE * np.asarray(pH)


#: Nernstian pH slope at 298 K, in V per pH unit.  -0.0592 V, the number that
#: appears on every Pourbaix diagram.
PCET_SLOPE = -2.303 * THERMAL_VOLTAGE


# =============================================================================
# SECTION 11: Ionic transport (Lesson 15)
# =============================================================================

#: Cation transference numbers in aqueous solution at 0.1 M, 298 K.
#: See Table 2.3.1 in Bard & Faulkner.
TRANSFERENCE_NUMBERS = {
    'H+': 0.8314,
    'Na+': 0.3854,
    'K+': 0.4898,
}

#: Limiting molar conductivity, cm^2 ohm^-1 equiv^-1, at 298 K (Lesson 15).
MOLAR_CONDUCTIVITY = {
    'H+': 349.82,
    'Na+': 50.11,
    'K+': 73.52,
    'Li+': 38.69,
    'OH-': 198.0,
    'Cl-': 76.34,
}


def junction_potential(t_cation, conc_alpha, conc_beta, z=1):
    """Liquid junction potential for a 1:1 electrolyte, junction of the first type.

        E_j = (t+ - t-) (RT/zF) ln(a_alpha / a_beta)

    with t- = 1 - t+.  The physical content: whichever ion moves faster carries
    more of the charge across the junction, and the resulting charge separation
    is a potential.  If the two ions had equal mobility, E_j would vanish, which
    is exactly why a KCl salt bridge works.

    Concentrations are used in place of activities here, as in the lecture.

    Parameters
    ----------
    t_cation : float, cation transference number (see TRANSFERENCE_NUMBERS)
    conc_alpha, conc_beta : float, any consistent unit (only the ratio matters)
    z : int, ionic charge magnitude

    Returns
    -------
    float, V.
    """
    t_anion = 1.0 - t_cation
    return (t_cation - t_anion) * (R * T / (z * F)) * np.log(conc_alpha / conc_beta)


# =============================================================================
# SECTION 12: Waveforms (Lessons 16, 19, 23, 24, 25, 31)
# =============================================================================
#
# Each of these returns (time, signal) so that a lecture can plot the waveform
# and then annotate it.  Building the arrays is shared; the annotation stays in
# the notebook, because that is the part that teaches.

def step_waveform(e_initial=0.0, e_step=1.0, t_total=1.0, points=100):
    """A single potential step: the simplest perturbation there is.

    Returns
    -------
    (t, e) arrays.  The step occurs at the halfway point.
    """
    t = np.linspace(0, t_total, points)
    e = np.where(t < t_total / 2.0, e_initial, e_step)
    return t, e


def pulse_waveform(e_initial=0.0, e_pulse=0.01, t_rest=0.1, t_pulse=0.1,
                   steps=3, points=100):
    """Normal pulse / sampled current waveform (Lessons 19 and 23).

    A series of pulses of growing amplitude, each followed by a rest at
    ``e_initial`` that lets the diffusion layer relax back to bulk.  Getting
    ``t_rest`` long enough is the whole experimental art of NPV.

    Parameters
    ----------
    e_initial : float, V
    e_pulse : float, V, the increment added on each successive pulse
    t_rest, t_pulse : float, s
    steps : int, number of pulses
    points : int, samples per pulse-plus-rest cycle

    Returns
    -------
    (t, e) arrays.
    """
    ratio = t_rest / (t_rest + t_pulse)
    n_rest = int(ratio * points)
    n_pulse = points - n_rest
    e = np.zeros(0)
    for i in range(steps):
        e = np.append(e, np.full(n_rest, e_initial))
        e = np.append(e, np.full(n_pulse, e_initial + (1 + i) * e_pulse))
    e = np.append(e, [e_initial])
    t = np.linspace(0, (t_rest + t_pulse) * steps, e.size)
    return t, e


def staircase_waveform(e_initial=0.0, e_step=0.01, t_step=1.0, steps=4,
                       points=100):
    """Staircase waveform (Lesson 23): a pulse waveform with the rest removed.

    Faster than NPV, at the cost of never letting the diffusion layer reset.
    Every "linear sweep" on a digital potentiostat is really one of these.

    Returns
    -------
    (t, e) arrays.
    """
    e = np.zeros(0)
    for i in range(steps):
        e = np.append(e, np.full(points, e_initial + i * e_step))
    t = np.linspace(0, t_step * steps, e.size)
    return t, e


def dpv_waveform(e_initial=0.0, e_step=0.01, e_pulse=0.01, t_step=3.0,
                 t_pulse=1.0, steps=4, points=200):
    """Differential pulse waveform (Lesson 24): a pulse train on a staircase.

    The current is sampled once just before each pulse and once at the end of
    it, and the DIFFERENCE is the analytical signal.  Counting the electrons
    twice like this is what turns the Nernstian wave into a peak.

    Returns
    -------
    (t, e_total, e_staircase) arrays.  Plot ``e_staircase`` underneath as a
    dotted line to show what the pulses are riding on.
    """
    cycle = t_step + t_pulse
    n_pulse = int(t_pulse / cycle * points)
    n_step = points - n_pulse
    e = np.zeros(0)
    base = np.zeros(0)
    for i in range(steps):
        e = np.append(e, np.full(n_pulse, e_initial + i * e_step + e_pulse))
        e = np.append(e, np.full(n_step, e_initial + i * e_step))
        base = np.append(base, np.full(points, e_initial + i * e_step))
    t = np.linspace(0, cycle * steps, e.size)
    return t, e, base


def swv_waveform(e_initial=0.0, e_step=0.01, e_pulse=0.01, t_step=2.0,
                 steps=4, points=100):
    """Square wave waveform (Lesson 24): a square wave summed onto a staircase.

    The forward and reverse halves of each square are sampled and subtracted.
    Because the two halves are equal and opposite, the double layer charging
    current cancels, which is what allows the very high effective scan rates.

    Returns
    -------
    (t, e_total, e_staircase) arrays.
    """
    square = np.zeros(0)
    base = np.zeros(0)
    half = points // 2
    for i in range(steps):
        square = np.append(square, np.full(half, e_pulse))
        square = np.append(square, np.full(points - half, -e_pulse))
        base = np.append(base, np.full(points, e_initial + i * e_step))
    t = np.linspace(0, t_step * steps, base.size)
    return t, base + square, base


def cv_waveform(e_initial=-0.2, e_switch=0.2, scan_rate=0.1, cycles=1,
                points_per_cycle=500):
    """Cyclic voltammetry waveform: sweep out to E_switch and back (Lesson 25).

    Parameters
    ----------
    e_initial, e_switch : float, V
    scan_rate : float, V s^-1
    cycles : int
    points_per_cycle : int

    Returns
    -------
    (t, e) arrays.
    """
    sweep_time = abs(e_initial - e_switch) / scan_rate
    half = points_per_cycle // 2
    one_cycle = np.append(
        e_initial + scan_rate * np.linspace(0, sweep_time, half),
        e_switch - scan_rate * np.linspace(0, sweep_time, points_per_cycle - half))
    e = np.tile(one_cycle, cycles)
    t = np.linspace(0, 2 * cycles * sweep_time, e.size)
    return t, e


def sine_waveform(frequency, amplitude=0.005, cycles=1, phase_deg=0.0,
                  points_per_cycle=201):
    """Sinusoidal perturbation for impedance spectroscopy (Lesson 31).

        e(t) = amplitude * sin(2 pi frequency t + phase)

    Note that ``frequency`` is the ordinary frequency in Hz.  The angular
    frequency that appears in the impedance equations is omega = 2 pi f.

    Returns
    -------
    (t, e) arrays.
    """
    t = np.linspace(0, cycles / frequency, points_per_cycle * cycles)
    return t, amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase_deg))


# =============================================================================
# SECTION 13: Helpers for interactive widgets
# =============================================================================

def parse_numbers(text, expected=None, name='value'):
    """Read a comma separated list of numbers typed into a text widget.

    Written to fail loudly.  ``np.fromstring`` silently returns -1.0 for
    anything it cannot parse and silently truncates at the first bad entry,
    which produces a plausible looking but wrong figure.

    >>> parse_numbers('0.3, 0.5, 0.7')
    array([0.3, 0.5, 0.7])

    Parameters
    ----------
    text : str
    expected : int or None
        If given, require exactly this many numbers.
    name : str
        Used in the error message.

    Returns
    -------
    numpy array of floats.

    Raises
    ------
    ValueError with a message that says what went wrong.
    """
    pieces = [p.strip() for p in str(text).replace(';', ',').split(',')]
    pieces = [p for p in pieces if p]
    if not pieces:
        raise ValueError(f'No {name}s given. Type numbers separated by commas.')
    try:
        values = np.array([float(p) for p in pieces])
    except ValueError:
        bad = [p for p in pieces if not _is_number(p)]
        raise ValueError(
            f'Could not read {bad} as {name}s. '
            'Type numbers separated by commas, for example: 0.3, 0.5, 0.7') from None
    if expected is not None and values.size != expected:
        raise ValueError(
            f'Expected {expected} {name}s but got {values.size}: {list(values)}')
    return values


def _is_number(text):
    """True if ``text`` can be read as a float.  Used by :func:`parse_numbers`."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def nearest_index(array, value):
    """Index of the entry in ``array`` closest to ``value``.

    The idiom ``(np.abs(a - v)).argmin()`` appears constantly when a figure
    needs to mark a particular time or potential on a computed curve.  Naming
    it makes the intent obvious and stops the index being reused for something
    it does not describe.
    """
    return int(np.abs(np.asarray(array) - value).argmin())


# =============================================================================
# SECTION 14: Drawing helpers reused by more than one lecture
# =============================================================================

def draw_solution_step(ax, high, low, left_cations, right_cations,
                       left_anions, right_anions):
    """Draw a two compartment cell with counted ions in each half (Lesson 15).

    Used to walk through what happens to the ions when charge crosses a liquid
    junction.  Changing the four counts is the whole demonstration.

    Parameters
    ----------
    ax : matplotlib axes
    high, low : float, the top and bottom of the solution band in data units
    left_cations, right_cations, left_anions, right_anions : int

    Returns
    -------
    The same axes, so calls can be chained.
    """
    ax.plot([0, 0], [low, high], 'black')
    ax.plot([2, 2], [low, high], 'black')
    for level in [high, high - 0.1]:
        ax.plot([-0.1, 0], [level - 0.1, level], 'black')
        ax.plot([2.0, 2.1], [level, level - 0.1], 'black')
    ax.plot([1, 1], [low, high], linestyle=':', c='gray')

    for i in range(left_cations):
        ax.scatter(0.1 * i + 0.05, high - 0.05, facecolors='blue',
                   edgecolors='blue', marker=r'$\bigoplus$', s=60)
    for i in range(right_cations):
        ax.scatter(1.95 - 0.1 * i, high - 0.05, facecolors='blue',
                   edgecolors='blue', marker=r'$\bigoplus$', s=60)
    for i in range(left_anions):
        x = 0.1 * i + 0.05
        _draw_anion(ax, x, low + 0.05)
    for i in range(right_anions):
        x = 1.95 - 0.1 * i
        _draw_anion(ax, x, low + 0.05)
    return ax


def _draw_anion(ax, x, y):
    """Draw one anion: an open circle with a minus sign through it."""
    ax.scatter(x, y, facecolors='none', edgecolors='red', s=60)
    ax.plot([x - 0.02, x + 0.02], [y] * 2, 'red')


def annotate_demo_cv(ax, potential, current, marked_potentials=None,
                     n_forward=500):
    """Plot the shared demonstration voltammogram with its landmarks labelled.

    Used by Lessons 25 and 28 so that both show the identical figure.

    Marks the anodic peak E_p,a and the half peak potential E_p/2,a, and
    optionally scatters points at chosen potentials for a sketching activity.

    Parameters
    ----------
    ax : matplotlib axes
    potential, current : arrays from ``Resources/data/L21/demo_cv.csv``
    marked_potentials : list of float or None
        Potentials at which to place markers for students to sketch
        concentration profiles.
    n_forward : int
        Number of points making up the forward sweep.

    Returns
    -------
    dict with the located landmark potentials, in V.
    """
    forward_e = potential[:n_forward]
    forward_i = current[:n_forward]

    peak_index = int(np.argmax(forward_i))
    e_peak_anodic = forward_e[peak_index]
    e_half_peak = forward_e[nearest_index(forward_i[:peak_index], forward_i.max() / 2)]

    ax.set_xlabel(r"E /V vs E$^{0'}$")
    ax.set_ylabel(r'i/i$_p$')
    ax.plot(forward_e, forward_i, 'b')

    if marked_potentials:
        for e in marked_potentials:
            idx = nearest_index(forward_e, e)
            ax.scatter(x=forward_e[idx], y=forward_i[idx], s=20)

    ax.axvline(e_peak_anodic, color='orange', linestyle=':')
    ax.annotate(r'E$_{p,a}$', (e_peak_anodic, 0.3), color='orange')
    ax.axvline(e_half_peak, color='black', linestyle=':')
    ax.annotate(r'E$_{p/2,a}$', (e_half_peak - 0.07, 0.75), color='black')
    ax.set_xlim(-0.22, 0.22)

    return {'e_peak_anodic': e_peak_anodic, 'e_half_peak': e_half_peak}
