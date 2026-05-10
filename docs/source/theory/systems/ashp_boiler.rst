Air Source Heat Pump Boiler (ASHPB)
===================================

.. seealso::
   
   API 레퍼런스: :mod:`enex_analysis.air_source_heat_pump_boiler`

System Overview
---------------

The Air Source Heat Pump Boiler (ASHPB) dynamically evaluates the thermodynamic refrigerant cycle by interfacing an outdoor-air evaporator with a thermal storage tank. 

.. figure:: ../../_static/images/fig1_system_diagram.png
   :align: center
   :alt: ASHPB System Diagram
   :width: 80%

   Schematic overview of the proposed air-source heat pump boiler (ASHPB) system.

System Integration
------------------

The ASHPB orchestrates two primary sub-components:

1. **Refrigerant Cycle**: Resolves the vapor-compression cycle dynamics. See :doc:`../components/refrigerant_cycle` for full modeling details.
2. **Thermal Storage Tank**: Captures the temporal evolution of the bulk water temperature based on heat pump supply and DHW draws. See :doc:`../components/thermal_storage`.

Optimal Operating Point
-----------------------

At each time step, the system model integrates the external boundary conditions (outdoor air temperature) and identifies the optimal evaporation approach temperature (:math:`\Delta T_\text{ref,evap}`) that minimizes the total electrical power input (:math:`E_\text{tot}`), comprising both the compressor and the outdoor unit fan power (:math:`E_\text{ou,fan}`). 

This bounded 1-D optimization is solved using Brent's method:

.. math::

   \min_{\Delta T_\text{ref,evap}} \quad E_\text{tot} = E_\text{cmp} + E_\text{ou,fan}

Simulation Results
------------------

The dynamic resolution of the integrated sub-components allows for high-fidelity, 1-minute temporal resolution simulations. The model captures the transient variations in the system Coefficient of Performance (COP) as the storage tank temperature fluctuates due to domestic hot water draws and the resulting cold-water mixing.

.. figure:: ../../_static/images/fig3_performance.png
   :align: center
   :alt: Seasonal daily dynamics of the ASHPB system
   :width: 100%

   Seasonal daily dynamics of the ASHPB system operating at 1-minute time-steps. The panels sequentially illustrate (a) thermal storage tank temperature, (b) electrical power input breakdown, (c) system COP, and (d) cumulative total energy use.
