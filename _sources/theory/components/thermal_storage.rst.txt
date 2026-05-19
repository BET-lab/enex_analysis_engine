Thermal Storage Tank
====================

The Thermal Storage Tank models the dynamic buffering of thermal energy. It acts as the thermal capacitor between the heat pump system and the domestic hot water (DHW) demand.

Modeling Assumptions
--------------------

- **Lumped Thermal Capacitance:** The tank water is fully mixed and represented by a single uniform temperature. Thermal stratification is not considered.
- **Full Tank Water Level:** The storage tank maintains a full water level, meaning the inlet cold makeup water flow rate perfectly matches the DHW draw rate.

Mathematical Modeling
---------------------

The temporal evolution of the tank's bulk water temperature (:math:`T_\text{w,tank}`) is governed by a standard lumped-capacitance energy balance equation evaluated at each 1-minute time step:

.. math::

   C_\text{tank} \frac{dT_\text{w,tank}}{dt} = Q_\text{HP} + Q_\text{STC} + Q_\text{flow} - Q_\text{loss}

Where:

- :math:`C_\text{tank}` is the total thermal capacitance of the water mass inside the tank.
- :math:`Q_\text{HP}` is the thermal energy supplied by the heat pump condenser.
- :math:`Q_\text{STC}` is the thermal energy supplied by the Solar Thermal Collector (if connected in a tank-circuit configuration).
- :math:`Q_\text{flow}` represents the net sensible heat change due to water draw events.
- :math:`Q_\text{loss}` is the ambient thermal loss through the tank's insulation shell.

Water Draw and Mixing
^^^^^^^^^^^^^^^^^^^^^

During a domestic hot water draw event, hot water leaves the tank, and an equivalent volume of cold mains water (:math:`T_\text{w,mains}`) enters the tank to replace it. The net enthalpy change rate is:

.. math::

   Q_\text{flow} = \dot{m}_\text{DHW} \cdot c_{p,\text{w}} \cdot (T_\text{w,mains} - T_\text{w,tank})

If a mains-preheat STC system is installed, the incoming water temperature may be elevated above the standard mains temperature before mixing into the tank.

Ambient Heat Loss
^^^^^^^^^^^^^^^^^

The ambient heat loss is calculated using an overall thermal conductance (:math:`UA_\text{tank}`):

.. math::

   Q_\text{loss} = UA_\text{tank} \cdot (T_\text{w,tank} - T_\text{amb})
