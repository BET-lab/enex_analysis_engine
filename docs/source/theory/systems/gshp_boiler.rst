Ground Source Heat Pump Boiler (GSHPB)
======================================

.. seealso::
   
   API 레퍼런스: :mod:`enex_analysis.ground_source_heat_pump_boiler`

System Overview
---------------

The Ground Source Heat Pump Boiler (GSHPB) couples a vapour-compression refrigerant cycle with a Borehole Heat Exchanger (BHE) on the evaporator side and a thermal storage tank on the condenser side. 

System Integration
------------------

The GSHPB integrates three major sub-components:

1. **Refrigerant Cycle**: The core thermodynamic cycle. See :doc:`../components/refrigerant_cycle`.
2. **Borehole Heat Exchanger (BHE)**: The ground thermal response model and circulation pump physics. See :doc:`../components/borehole_he`.
3. **Thermal Storage Tank**: The DHW buffering and mixing tank. See :doc:`../components/thermal_storage`.

Unlike air-source systems, the GSHPB evaporator is coupled to the fluid circulating through the BHE, ensuring stable evaporator temperatures independent of ambient air conditions.

Optimal Operating Point
-----------------------

At each simulation time step, the model identifies the optimal evaporation approach temperature (:math:`\Delta T_\text{ref,evap}`) that satisfies the physical NTU limits of the evaporator while minimizing the total electrical power input (:math:`E_\text{tot}`). 

The total power includes both the compressor and the BHE circulation pump:

.. math::

   \min_{\Delta T_\text{ref,evap}} \quad E_\text{tot} = E_\text{cmp} + E_\text{pmp}

This optimization is solved via 1-D Brent's method.
