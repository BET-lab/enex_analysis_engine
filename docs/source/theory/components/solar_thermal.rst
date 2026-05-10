Solar Thermal Collector (STC)
=============================

Solar Thermal Collectors (STCs) directly capture solar irradiance to provide thermal energy. 

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

The STC energy generation model evaluates the solar irradiance converted into useful heat. 

Energy Capture
^^^^^^^^^^^^^^

The usable thermal energy (:math:`Q_\text{STC}`) produced by the collector is determined by its efficiency (:math:`\eta_\text{STC}`), the incident solar irradiance (:math:`I_\text{solar}`), and the collector surface area (:math:`A_\text{STC}`):

.. math::

   Q_\text{STC} = \eta_\text{STC} \cdot I_\text{solar} \cdot A_\text{STC}

The efficiency (:math:`\eta_\text{STC}`) depends on optical properties, ambient temperature, and the collector operating temperature.

Thermodynamic Outlet Temperature Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To accurately evaluate the dynamic thermal performance of the collector across changing environmental conditions, an analytical exponential model is employed. This model prevents numerical instability and allows precise calculation of the outlet temperature (:math:`T_\text{out}`) based on the mass flow rate and heat loss coefficient.

Given the effective heat capacity rate :math:`G = \dot{m} c_w` and the overall heat transfer coefficient area product :math:`AU`, the thermal response factor :math:`\xi` is defined as:

.. math::

   \xi = \exp\left(-\frac{AU}{G}\right)

The outlet temperature is then solved analytically as:

.. math::

   T_\text{out} = \xi \cdot T_\text{in} + (1 - \xi) \cdot T_0 + (1 - \xi) \cdot \frac{Q_\text{sol}}{AU}

where:

* :math:`T_\text{in}` is the collector inlet fluid temperature.
* :math:`T_0` is the ambient outdoor temperature.
* :math:`Q_\text{sol} = I_\text{solar} \cdot A_\text{STC} \cdot \alpha` is the total absorbed solar energy.

System Configurations
---------------------

The generated heat is supplied either directly to the main thermal storage tank (Tank-Circuit configuration) or to preheat the incoming mains water (Mains-Preheat configuration). The activation logic for the STC circulation pump depends heavily on the chosen integration topology and current thermal boundary conditions.

Interactive Performance Widget
------------------------------

To build an intuition for how environmental parameters and system configurations impact the outlet temperature and overall useful heat, utilize the interactive simulator below.

.. raw:: html

   <iframe src="../../_static/widgets/stc_widget.html" width="100%" height="550px" style="border:none;"></iframe>
