Air Source Heat Pump Boiler (ASHPB)
===================================

.. seealso::
   
   API 레퍼런스: :mod:`enex_analysis.air_source_heat_pump_boiler`

System Overview
---------------

The Air Source Heat Pump Boiler (ASHPB) dynamically evaluates the thermodynamic refrigerant cycle by interfacing an outdoor-air evaporator with a thermal storage tank. The model resolves the coupled energy balance equations at each time step using CoolProp for refrigerant property evaluation.

.. figure:: ../_static/images/fig1_system_diagram.png
   :align: center
   :alt: ASHPB System Diagram
   :width: 80%

   Schematic overview of the proposed air-source heat pump boiler (ASHPB) system.

Modeling Assumptions
--------------------

The physics-based model simplifies the highly complex spatial and transient thermal phenomena under the following assumptions:

- **Lumped Thermal Capacitance:** The thermal storage tank water is fully mixed and represented by a single uniform temperature. Thermal stratification is not considered.
- **Negligible Heat Exchanger Thermal Resistance:** The thermal resistance of the metallic coil walls in the heat exchangers is negligible. The entire surface temperature of the condenser and evaporator is assumed to equal the condensation (:math:`T_\text{ref,cond,sat}`) and evaporation (:math:`T_\text{ref,evap,sat}`) saturation temperatures, respectively.
- **Negligible Single-Phase Region:** The surface area dedicated to the superheated and subcooled single-phase regions within the heat exchangers is assumed to be small relative to the two-phase region.
- **Isenthalpic Expansion:** The expansion process through the expansion valve is considered perfectly isenthalpic.
- **Full Tank Water Level:** The storage tank maintains a full water level, meaning the inlet makeup water flow rate perfectly matches the domestic hot water (DHW) draw rate.

Mathematical Modeling
---------------------

Refrigerant Cycle Resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The thermodynamic states of the refrigerant cycle are fundamentally anchored by the evaporating and condensing saturation temperatures. These are derived from the outdoor air temperature (:math:`T_0`) and the tank water temperature (:math:`T_\text{w,tank}`):

.. math::

   T_\text{ref,evap,sat} &= T_0 - \Delta T_\text{ref,evap} \\
   T_\text{ref,cond,sat} &= T_\text{w,tank} + \Delta T_\text{ref,cond}

The condenser approach temperature (:math:`\Delta T_\text{ref,cond}`) is directly determined by the target heating capacity (:math:`Q_\text{ref,cond}`) and the overall heat transfer coefficient (:math:`UA_\text{cond}`) of the condenser:

.. math::

   \Delta T_\text{ref,cond} = \frac{Q_\text{ref,cond}}{UA_\text{cond}}

The compressor inlet (cmp,in) and expansion valve inlet (exp,in) temperatures are defined by user-specified superheat (:math:`\Delta T_\text{superheat}`) and subcool (:math:`\Delta T_\text{subcool}`) margins to ensure safe and stable cycle operation:

.. math::

   T_\text{ref,cmp,in} &= T_\text{ref,evap,sat} + \Delta T_\text{superheat} \\
   T_\text{ref,exp,in} &= T_\text{ref,cond,sat} - \Delta T_\text{subcool}

Based on the isentropic efficiency (:math:`\eta_\text{cmp,isen}`) of the compressor, the actual discharge enthalpy (:math:`h_\text{ref,cmp,out}`) is calculated from the ideal isentropic discharge enthalpy (:math:`h_\text{2,isen}`):

.. math::

   h_\text{ref,cmp,out} = h_\text{ref,cmp,in} + \frac{h_\text{2,isen} - h_\text{ref,cmp,in}}{\eta_\text{cmp,isen}}

Energy Balance and Mass Flow Rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The required refrigerant mass flow rate (:math:`\dot{m}_\text{ref}`) is derived by applying the steady-state energy balance across the condenser to meet the target heat load (:math:`Q_\text{ref,cond}`):

.. math::

   \dot{m}_\text{ref} = \frac{Q_\text{ref,cond}}{h_\text{ref,cmp,out} - h_\text{ref,exp,in}}

With the mass flow rate established, the evaporator heat absorption rate (:math:`Q_\text{ref,evap}`) and the compressor electrical power input (:math:`E_\text{cmp}`) are computed:

.. math::

   Q_\text{ref,evap} &= \dot{m}_\text{ref} (h_\text{ref,cmp,in} - h_\text{ref,exp,out}) \\
   E_\text{cmp} &= \dot{m}_\text{ref} (h_\text{ref,cmp,out} - h_\text{ref,cmp,in})

The compressor speed (:math:`N_\text{cmp}` in RPM) is determined by the required mass flow rate, the compressor displacement volume (:math:`V_\text{disp,cmp}`), and the suction density (:math:`\rho_\text{ref,cmp,in}`):

.. math::

   N_\text{cmp} = \frac{\dot{m}_\text{ref}}{V_\text{disp,cmp} \cdot \rho_\text{ref,cmp,in}} \times 60

Optimal Operating Point
^^^^^^^^^^^^^^^^^^^^^^^

At each time step, the model identifies the optimal evaporation approach temperature (:math:`\Delta T_\text{ref,evap}`) that minimizes the total electrical power input (:math:`E_\text{tot}`), comprising the compressor and the outdoor unit fan power (:math:`E_\text{ou,fan}`). This bounded 1-D optimization is solved using Brent's method:

.. math::

   \min_{\Delta T_\text{ref,evap}} \quad E_\text{tot} = E_\text{cmp} + E_\text{ou,fan}

Simulation Results
------------------

The dynamic resolution of the refrigerant cycle and system energy balances allows for high-fidelity, 1-minute temporal resolution simulations. The model captures the transient variations in the system Coefficient of Performance (COP) as the storage tank temperature fluctuates due to domestic hot water draws and the resulting cold-water mixing.

.. figure:: ../_static/images/fig3_performance.png
   :align: center
   :alt: Seasonal daily dynamics of the ASHPB system
   :width: 100%

   Seasonal daily dynamics of the ASHPB system operating at 1-minute time-steps. The panels sequentially illustrate (a) thermal storage tank temperature, (b) electrical power input breakdown, (c) system COP, and (d) cumulative total energy use.
