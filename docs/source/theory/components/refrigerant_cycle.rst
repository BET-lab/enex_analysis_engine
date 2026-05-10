Refrigerant Cycle
=================

The core of the heat pump models relies on evaluating the thermodynamic state of the vapor-compression cycle. The mathematical resolution assumes steady-state balances at each time step and uses the `CoolProp` library for accurate refrigerant property evaluation.

Cycle Visualization
-------------------

The interactive widget below visualizes the thermodynamic cycle on Temperature-Enthalpy (T-h), Pressure-Enthalpy (P-h), and Temperature-Entropy (T-s) diagrams. You can select the refrigerant and adjust the source and sink temperatures to see how the cycle states change.

.. raw:: html
   :file: ../../_static/html/cycle_widget.html

Modeling Assumptions
--------------------

- **Negligible Heat Exchanger Thermal Resistance:** The thermal resistance of the metallic coil walls in the heat exchangers is negligible. The entire surface temperature of the condenser and evaporator is assumed to equal the condensation (:math:`T_\text{ref,cond,sat}`) and evaporation (:math:`T_\text{ref,evap,sat}`) saturation temperatures, respectively.
- **Negligible Single-Phase Region:** The surface area dedicated to the superheated and subcooled single-phase regions within the heat exchangers is assumed to be small relative to the two-phase region.
- **Isenthalpic Expansion:** The expansion process through the expansion valve is considered perfectly isenthalpic.

Mathematical Modeling
---------------------

Thermodynamic States
^^^^^^^^^^^^^^^^^^^^

The thermodynamic states of the refrigerant cycle are anchored by the evaporating and condensing saturation temperatures. These are derived from the heat source temperature (e.g., outdoor air or BHE fluid) and the heat sink temperature (e.g., tank water):

.. math::

   T_\text{ref,evap,sat} &= T_\text{source} - \Delta T_\text{ref,evap} \\
   T_\text{ref,cond,sat} &= T_\text{sink} + \Delta T_\text{ref,cond}

The condenser approach temperature (:math:`\Delta T_\text{ref,cond}`) is determined by the target heating capacity (:math:`Q_\text{ref,cond}`) and the overall heat transfer coefficient (:math:`UA_\text{cond}`) of the condenser:

.. math::

   \Delta T_\text{ref,cond} = \frac{Q_\text{ref,cond}}{UA_\text{cond}}

The treatment of the overall heat transfer coefficient (:math:`UA`) depends on the physical placement and type of the heat exchanger:

- **Immersed / Hydronic Heat Exchangers (e.g., Tank Condensers or BHE Evaporators):** When the heat exchanger interacts with a liquid body under simplified mixing assumptions (like a lumped-capacitance water tank) or fixed ground loop flows, the water-side or brine-side thermal resistance is often assumed constant. Thus, :math:`UA` is treated as a static design parameter (:math:`UA_\text{design}`).
- **Air-Coupled Heat Exchangers (e.g., Outdoor Evaporators or Air-Cooled Condensers):** When the heat exchanger is coupled with an external fan, the :math:`UA` dynamically scales with the fan's volumetric airflow rate (:math:`\dot{V}_\text{a}`). Based on the Colburn :math:`j`-factor analogy for plain fin-and-tube configurations, the convective heat transfer coefficient varies non-linearly with fluid velocity. This is modeled using an empirical exponent :math:`n`:

  .. math::

     UA = UA_\text{design} \left(\frac{\dot{V}_\text{a}}{\dot{V}_\text{a,design}}\right)^{n}

  where the exponent :math:`n` typically ranges from 0.58 to 0.83 depending on the number of tube rows, and is nominally set to 0.65 to represent standard operating performance.

The compressor inlet (cmp,in) and expansion valve inlet (exp,in) temperatures are defined by user-specified superheat (:math:`\Delta T_\text{superheat}`) and subcool (:math:`\Delta T_\text{subcool}`) margins to ensure safe operation:

.. math::

   T_\text{ref,cmp,in} &= T_\text{ref,evap,sat} + \Delta T_\text{superheat} \\
   T_\text{ref,exp,in} &= T_\text{ref,cond,sat} - \Delta T_\text{subcool}

Based on the isentropic efficiency (:math:`\eta_\text{cmp,isen}`) of the compressor, the actual discharge enthalpy (:math:`h_\text{ref,cmp,out}`) is calculated from the ideal isentropic discharge enthalpy (:math:`h_\text{2,isen}`):

.. math::

   h_\text{ref,cmp,out} = h_\text{ref,cmp,in} + \frac{h_\text{2,isen} - h_\text{ref,cmp,in}}{\eta_\text{cmp,isen}}

Energy Balance and Mass Flow Rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The required refrigerant mass flow rate (:math:`\dot{m}_\text{ref}`) is derived by applying the steady-state energy balance across the condenser to meet the target heat load:

.. math::

   \dot{m}_\text{ref} = \frac{Q_\text{ref,cond}}{h_\text{ref,cmp,out} - h_\text{ref,exp,in}}

With the mass flow rate established, the evaporator heat absorption rate (:math:`Q_\text{ref,evap}`) and the compressor electrical power input (:math:`E_\text{cmp}`) are computed:

.. math::

   Q_\text{ref,evap} &= \dot{m}_\text{ref} (h_\text{ref,cmp,in} - h_\text{ref,exp,out}) \\
   E_\text{cmp} &= \dot{m}_\text{ref} (h_\text{ref,cmp,out} - h_\text{ref,cmp,in})

The compressor speed (:math:`N_\text{cmp}` in RPM) is determined by the required mass flow rate, the compressor displacement volume (:math:`V_\text{disp,cmp}`), and the suction density (:math:`\rho_\text{ref,cmp,in}`):

.. math::

   N_\text{cmp} = \frac{\dot{m}_\text{ref}}{V_\text{disp,cmp} \cdot \rho_\text{ref,cmp,in}} \times 60
