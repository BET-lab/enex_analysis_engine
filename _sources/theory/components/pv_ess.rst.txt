Photovoltaic and Energy Storage Systems (PV + ESS)
==================================================

Photovoltaic (PV) systems generate electricity from solar irradiance, while Energy Storage Systems (ESS) allow for temporal shifting of this generated power. The integration of PV and ESS enables high-efficiency heat pump operation by maximizing self-consumption.

Irradiance Decomposition and Transposition
------------------------------------------

Raw weather data typically provides Global Horizontal Irradiance (GHI). However, solar panels and collectors are usually tilted and oriented towards a specific azimuth. To compute the actual irradiance falling onto the Plane of Array (POA), a multi-step preprocessing logic is executed:

1. **Solar Position Calculation**:
   Using the site's latitude, longitude, altitude, and timezone, the sun's apparent zenith and azimuth are calculated at each time step via ``pvlib.location.Location.get_solarposition()``.

2. **GHI Decomposition**:
   The Erbs decomposition model (`pvlib.irradiance.erbs <https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.erbs.html>`_) is applied to separate the GHI into Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DHI).

3. **POA Transposition**:
   Finally, the Perez transposition model (`pvlib.irradiance.get_total_irradiance <https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html>`_) projects the direct, diffuse, and ground-reflected components onto the tilted surface defined by the user's ``tilt`` and ``azimuth`` parameters.

.. tip::
   
   **Computational Efficiency**
   
   Executing the full ``pvlib`` position and transposition algorithms inside the 1-minute dynamic simulation loop would be computationally prohibitive. Therefore, this process is executed as a **preprocessing step** (e.g., via ``enex_analysis.weather.decompose_ghi_to_poa``). The resulting POA components (:math:`I_\text{DN}` and :math:`I_\text{dH}`) are directly fed into the simulation's ``StepContext``, ensuring high-speed dynamic solving.

Mathematical Modeling
---------------------

PV Generation
^^^^^^^^^^^^^

The PV array converts the incident Plane of Array (POA) solar irradiance (:math:`I_\text{sol} = I_\text{DN} + I_\text{dH}`) into DC electricity. The cell temperature (:math:`T_\text{pv}`) is dynamically approximated based on the ambient temperature (:math:`T_0`), irradiance, absorptivity (:math:`\alpha_\text{pv}`), and the external convective coefficient (:math:`h_o`).

The nominal efficiency of the PV panel is derated based on this cell temperature using the temperature coefficient (:math:`\beta_\text{pv}`):

.. math::

   \eta_\text{pv,actual} = \eta_\text{pv,ref} \left[1 - \beta_\text{pv} (T_\text{pv} - T_\text{ref})\right]

The resulting DC power output is then regulated by the charge controller.

Energy Storage System (ESS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ESS (Battery) model accepts charging and discharging requests subject to the physical capacity limits (:math:`C_\text{ess,max}`) and the allowable State of Charge (SOC) boundaries (:math:`SOC_\text{min}` to :math:`SOC_\text{max}`). Internal losses and exergy destruction are computed from the round-trip efficiencies (:math:`\eta_\text{ess,chg}` and :math:`\eta_\text{ess,dis}`).

Dynamic Energy Routing
^^^^^^^^^^^^^^^^^^^^^^

The energy routing between the PV array, ESS, Heat Pump (HP), and the Grid is resolved synchronously at each timestep *after* the total HVAC electrical load (:math:`E_\text{hp,load}`) is evaluated. The 3-way DC energy routing logic functions as follows:

1. **HP Off**: 
   When there is no building demand, all PV generation is routed to charge the ESS. Any overflow beyond the battery's maximum charge rate or capacity is curtailed (dumped).
   
2. **HP On + PV Surplus**: 
   When PV generation exceeds the HP electrical demand, the PV array directly supplies the HP through the inverter. The remaining surplus DC power is used to charge the ESS.

3. **HP On + PV Deficit**: 
   When PV generation is insufficient to meet the HP demand, the PV array provides its maximum output to the inverter. The ESS is discharged to cover the deficit. If the combined PV and ESS output is still insufficient, the remaining shortfall is imported from the external power grid.

The final electrical balance ensures that the grid import (:math:`E_\text{grid}`) strictly covers any unresolved deficit:

.. math::

   E_\text{grid} = \max\left(0, E_\text{hp,load} - \eta_\text{inv} (E_\text{dc,pv} + E_\text{dc,ess,dis})\right)
