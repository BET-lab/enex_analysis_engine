import numpy as np
import math
import constant as c
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

rho_w = 1000
c_w   = 4186 # Water specific heat [J/kgK]

@dataclass
class ElectricBoiler:
   def __post_init__(self):
      # Parameters
      self.COP = 2.5
      self.eta = 0.3        # Energy conversion efficiency [-]

      # Temperature Parameters
      self.T_w_tank  = c.C2K(60) # Hot water temperature [K]
      self.T_w_sup  = c.C2K(10) # Supply water temperature [K]
      self.T_w_tap = c.C2K(45) # Tap water temperature [K]
      self.T0    = c.C2K(0)  # Ambient temperature [K]
      
      # Tank Volume Parameters
      self.water_use_in_a_day = 0.2 # Usable volume [m³/day]
      self.V_tank             = 0.1 # Total tank volume [m³]
      self.n                  = 3   # Height to diameter ratio [-] (height = n * diameter)

      # Heat Transfer Parameters
      self.h_co   = 2    # Convective heat transfer coefficient [W/m²K]
      self.h_ro   = 5    # Radiative heat transfer coefficient [W/m²K]
      self.x_shell = 0.01 # tank shell thickness [m]
      self.k_shell = 50   # tank shell thermal conductivity [W/mK]
      self.x_ins  = 0.10 # Insulation thickness [m]
      self.k_ins  = 0.03 # Insulation thermal conductivity [W/mK]

        # Tank Dimension Calculations
   def system_update(self):
      self.T_tank_is     = self.T_w_tank # inner surface temperature of tank [K]
      self.hour_w_use    = 3 # water use hours [h]
      self.dV_w_tap      = self.water_use_in_a_day/(self.hour_w_use*c.h2s) # mean volume variation [m³/s]
      self.alp           = (self.T_w_tap - self.T_w_sup)/(self.T_w_tank - self.T_w_sup) # if 문으로 음수 제한
      self.alp           = print("alp is negative") if self.alp < 0 else self.alp
      self.dV_w_sup_tank = self.alp * self.dV_w_tap
      self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_tap
   
      # Surface areas
      self.r0 = (self.V_tank / (2 * math.pi * self.n)) ** (1 / 3)  # Tank inner radius [m]
      self.r1 = self.r0 + self.x_shell  # Tank outer radius [m]
      self.r2 = self.r1 + self.x_ins  # Insulation outer radius [m]

      self.h = self.n * (2 * self.r0)  # Tank height [m]

      self.A_top_bottom = 2 * math.pi * self.r0 ** 2  # Total top and bottom area [m²]

      # Thermal resistances
      self.h_o = self.h_co + self.h_ro  # Overall heat transfer coefficient [W/m²K]
      # Cylindrical coordinates ========================================
      self.R_side_shell = math.log((self.r1) / self.r0) / (2 * math.pi * self.k_shell) # Shell thermal resistance [mK/W]
      self.R_side_ins   = math.log((self.r2) / (self.r1)) / (2 * math.pi * self.k_ins) # Insulation thermal resistance [mK/W]
      self.R_side_ext   = 1 / (2 * math.pi * self.r2 * self.h_o)  # External thermal resistance [mK/W]
      self.R_side_tot = self.R_side_shell + self.R_side_ins + self.R_side_ext # Total side thermal resistance [mK/W]
      self.U_side       = 1 / self.R_side_tot # Overall heat transfer coefficient [W/mK]
      # Cartesian coordinates ==========================================
      self.R_top_bottom_shell = (self.x_shell) / (self.k_shell) # Insulation thermal resistance [m2K/W]
      self.R_top_bottom_ins   = (self.x_ins) / (self.k_ins) # Insulation thermal resistance [m2K/W]
      self.R_top_bottom_ext   = 1/self.h_o  # Combined external thermal resistance [m2K/W]
      self.R_top_bottom_tot   = self.R_top_bottom_shell + self.R_top_bottom_ins + self.R_top_bottom_ext # Total top and bottom thermal resistance [m2K/W]
      self.U_top_bottom       = 1 / self.R_top_bottom_tot # Overall heat transfer coefficient [W/m2K]

      # Total heat transfer coefficient
      self.U_tank = self.U_side*self.h + self.U_top_bottom*self.A_top_bottom # Overall heat transfer coefficient [W/K]

      # Heat loss calculation
      self.Q_l_tank = self.U_tank * (self.T_w_tank - self.T0)

      # Heat Transfer Rates
      self.Q_w_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)
      self.Q_w_sup  = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)
      self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)
      self.E_heater = self.Q_w_tank + self.Q_l_tank - self.Q_w_sup # Electric Power input [W]
      self.E_NG     = self.E_heater/self.eta # Natural gas Power input [W]

      self.energy_balance = {
      'hot water tank': {
         "$E_{heater}$" : self.E_heater,                                                # input
         "$X_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0),  # input
         "$X_{w,tank}$"   : c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0), # output
         "$X_{l,tank}$"   : self.Q_l_tank # output
         },
      'mixing': {
         "$X_{w,tank}$"      : c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0), # input
         "$X_{w,sup,tap}$": c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0),   # input
         "$X_{w,tap}$"             : c_w * rho_w * self.dV_w_tap * (self.T_w_tap - self.T0),       # output
         }
      }

      s_g_tank = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0) + (1/self.T_tank_is)*self.Q_l_tank - c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup/self.T0)
      s_g_mix  = c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap/self.T0) - (c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0)+c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup/self.T0))

      self.entropy_balance = {
      'hot water tank': {
         "$E_{heater}$"    : (1/float('inf'))*self.E_heater,                                     # input
         "$X_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup/self.T0),  # input
         "$s_{g,tank}$"    : s_g_tank,                                                           # output
         "$X_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0), # output
         "$X_{l,tank}$"    : (1/self.T_tank_is)*self.Q_l_tank # output
         },
      'mixing': {
         "$X_{w,tank}$"   : c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0), # input
         "$X_{w,sup,tap}$": c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup/self.T0),   # input
         "$s_{g,mix}$"    : s_g_mix,
         "$X_{w,tap}$"    : c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap/self.T0),       # output
         }
      }

      self.exergy_balance = {
      'hot water tank': { 
         "$E_{heater}$"    : self.E_heater,                                                                                          # input
         "$X_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup-self.T0)-self.T0 * math.log(self.T_w_sup/self.T0)),   # input
         "$X_{c,,tank}$"   : s_g_tank * self.T0,
         "$X_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank-self.T0)-self.T0 * math.log(self.T_w_tank/self.T0)), # output
         "$X_{l,tank}$"    : (1-self.T0/self.T_tank_is)*self.Q_l_tank # output
         },
      'mixing': {
         "$X_{w,tank}$"   : c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank-self.T0)-self.T0 * math.log(self.T_w_tank/self.T0)), # input
         "$X_{w,sup,tap}$": c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup-self.T0) - self.T0 * math.log (self.T_w_sup/self.T0)), # input
         "$X_{c,mix}$"    : s_g_mix * self.T0,                                                                                      # output
         "$X_{w,tap}$"    : c_w * rho_w * self.dV_w_tap * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap/self.T0)),    # output
         }
      }  
      


@dataclass
class GasBoiler:
    def __post_init__(self):
        # Parameters
        self.COP = 2.5
        self.eta = 0.9  # Boiler efficiency [-]\
        self.eta_NG = 0.93 # Net efficiency [-]

        # Temperature Parameters
        self.T_w_tank = c.C2K(60)  # Hot water temperature [K]
        self.T_w_sup  = c.C2K(10)  # Supply water temperature [K]
        self.T_w_tap  = c.C2K(45)  # Tap water temperature [K]
        self.T0       = c.C2K(0)  # Surrounding temperature [K]
        self.T_flame  = c.C2K(1200)  # Flame temperature [K]
        self.T_exh    = c.C2K(70)  # Exhaust gas temperature [K]

        # Tank Volume Parameters
        self.water_use_in_a_day = 0.2  # [m³/day]
        self.hour_w_use    = 3 # water use hours [h]
        self.V_tank = 0.02  # Total tank volume [m³]
        self.n = 3  # Height to diameter ratio [-] (height = n * diameter)

        # Heat Transfer Parameters
        self.h_co = 2  # Convective heat transfer coefficient [W/m²K]
        self.h_ro = 5  # Radiative heat transfer coefficient [W/m²K]
        self.x_shell = 0.01  # Tank shell thickness [m]
        self.k_shell = 50  # Tank shell thermal conductivity [W/mK]
        self.x_ins = 0.10  # Insulation thickness [m]
        self.k_ins = 0.03  # Insulation thermal conductivity [W/mK]

    def system_update(self):
        c_w = 4186  # Specific heat capacity of water [J/kgK]
        rho_w = 1000  # Density of water [kg/m³]

        # Temperature
        self.T_tank_is = self.T_w_tank

        # Surface areas
        self.r0 = (self.V_tank / (2 * math.pi * self.n)) ** (1 / 3)  # Tank inner radius [m]
        self.r1 = self.r0 + self.x_shell  # Tank outer radius [m]
        self.r2 = self.r1 + self.x_ins  # Insulation outer radius [m]

        self.h = self.n * (2 * self.r0)  # Tank height [m]

        self.A_top_bottom = 2 * math.pi * self.r0 ** 2  # Total top and bottom area [m²]

        # Thermal resistances
        self.h_o = self.h_co + self.h_ro  # Overall heat transfer coefficient [W/m²K]
        # Cylindrical coordinates ========================================
        self.R_side_shell = math.log((self.r1) / self.r0) / (2 * math.pi * self.k_shell) # Shell thermal resistance [mK/W]
        self.R_side_ins   = math.log((self.r2) / (self.r1)) / (2 * math.pi * self.k_ins) # Insulation thermal resistance [mK/W]
        self.R_side_ext   = 1 / (2 * math.pi * self.r2 * self.h_o)  # External thermal resistance [mK/W]
        self.R_side_tot = self.R_side_shell + self.R_side_ins + self.R_side_ext # Total side thermal resistance [mK/W]
        self.U_side       = 1 / self.R_side_tot # Overall heat transfer coefficient [W/mK]
        # Cartesian coordinates ==========================================
        self.R_top_bottom_shell = (self.x_shell) / (self.k_shell) # Insulation thermal resistance [m2K/W]
        self.R_top_bottom_ins   = (self.x_ins) / (self.k_ins) # Insulation thermal resistance [m2K/W]
        self.R_top_bottom_ext   = 1/self.h_o  # Combined external thermal resistance [m2K/W]
        self.R_top_bottom_tot   = self.R_top_bottom_shell + self.R_top_bottom_ins + self.R_top_bottom_ext # Total top and bottom thermal resistance [m2K/W]
        self.U_top_bottom       = 1 / self.R_top_bottom_tot # Overall heat transfer coefficient [W/m2K]

        # Total heat transfer coefficient
        self.U_tank = self.U_side*self.h + self.U_top_bottom*self.A_top_bottom # Overall heat transfer coefficient [W/K]

        # Water flow rates and temperatures
        self.dV_w_tap = self.water_use_in_a_day / (self.hour_w_use*c.h2s)  # Average tap water flow rate [m³/s]
        self.alp = (self.T_w_tap - self.T_w_sup) / (self.T_w_tank - self.T_w_sup)
        if self.alp < 0:
            raise ValueError("Alpha (mixing ratio) is negative, check temperature inputs.")
        self.dV_w_sup_boiler = self.alp * self.dV_w_tap
        self.dV_w_sup_mix = (1 - self.alp) * self.dV_w_tap

        # Energy balance for boiler
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank
        self.T_w_boiler = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_boiler)
        self.E_NG = (c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_boiler - self.T_w_sup)) / self.eta

        # Heat losses
        self.Q_l_exh = (1 - self.eta) * self.E_NG  # Heat loss from exhaust gases

        # Energy balances
        self.energy_balance = {
            "boiler": {
                "$E_{NG}$": self.E_NG,
                "$X_{w,sup}$":c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_sup - self.T0),
                "$X_{w,boiler,out}$": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_boiler -self.T0),
                "$X_{a,exh}$": self.Q_l_exh
            },
            "hot water tank": {
                "$X_{w,boiler,out}$": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_boiler - self.T0),
                "$X_{w,tank}$": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_tank - self.T0),
                "$X_{l,tank}$": self.Q_l_tank,
            },
            "mixing": {
                "$X_{w,tank}$": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_tank - self.T0),
                "$X_{w,sup,tap}$": c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0),
                "$X_{w,tap}$": c_w * rho_w * self.dV_w_tap * (self.T_w_tap - self.T0)
            }
        }

        #Entropy generation
        self.T_NG = self.T0 / (1 - self.eta_NG)
        s_g_boiler = c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_boiler/self.T0) + (1/self.T_exh) * self.Q_l_exh - (c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_sup/self.T0)+(1/self.T_NG) * self.E_NG)
        s_g_tank = c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_tank/self.T0) + (1/self.T_tank_is) * self.Q_l_tank - c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_boiler/self.T0)
        s_g_mix = c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap/self.T0) - (c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_tank/self.T0) + c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup/self.T0))
        
        # Entropy balances
        self.entropy_balance = {
            "boiler": {
                "$E_{NG}$": (1/self.T_NG) * self.E_NG,
                "$X_{w,sup}$":c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_sup/self.T0),
                "$s_{g,boiler}$": s_g_boiler,
                "$X_{w,boiler,out}$": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_boiler/self.T0),
                "$X_{a,exh}$": (1/self.T_exh) * self.Q_l_exh,
            },
            "hot water tank": {
                "$X_{w,boiler,out}$" : c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_boiler/self.T0),
                "$s_{g,tank}$": s_g_tank,
                "$X_{w,tank}$": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_tank/self.T0),
                "$X_{l,tank}$": (1/self.T_tank_is) * self.Q_l_tank,
            },
            "mixing": {
                "$X_{w,tank}$": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_tank/self.T0),
                "$X_{w,sup,tap}$": c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup/self.T0),
                "$s_{g,mix}$": s_g_mix,
                "$X_{w,tap}$": c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap/self.T0)
            }
        }

        # Exergy balances
        self.exergy_balance = {
            "boiler": {
                "$X_{NG}$"          : self.eta_NG * self.E_NG,
                "$X_{w,sup}$"       : c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_sup-self.T0) - self.T0*math.log(self.T_w_sup/self.T0)),
                "$X_{c,boiler}$"    : s_g_boiler * self.T0,
                "$X_{w,boiler,out}$": c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_boiler-self.T0) - self.T0*math.log(self.T_w_boiler/self.T0)),
                "$X_{a,exh}$"       : (1-self.T0/self.T_exh) * self.Q_l_exh
            },
            "hot water tank": {
                "$X_{w,boiler,out}$": c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_boiler-self.T0) - self.T0*math.log(self.T_w_boiler/self.T0)),
                "$X_{c,tank}$"      : s_g_tank*self.T0,
                "$X_{w,tank}$"      : c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_tank-self.T0) - self.T0*math.log(self.T_w_tank/self.T0)),
                "$X_{l,tank}$"      : (1-self.T0/self.T_tank_is) * self.Q_l_tank,
            },
            "mixing": {
                "$X_{w,tank}$"   : c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_tank-self.T0)-self.T0 * math.log(self.T_w_tank/self.T0)),
                "$X_{w,sup,tap}$": c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup-self.T0) - self.T0 * math.log(self.T_w_sup/self.T0)),
                "$X_{c,mix}$"    : s_g_mix * self.T0,
                "$X_{w,tap}$"    : c_w * rho_w * self.dV_w_tap * ((self.T_w_tap-self.T0)-self.T0 * math.log(self.T_w_tap/self.T0))
            }
        }




@dataclass
class HeatPumpBoiler: 
    def __post_init__(self): 
        # Parameters
        self.eta_cmp = 0.9  # Compressor efficiency [-]
        self.eta_fan = 0.6  # Fan efficiency [-]
        self.kappa_fan = 0.7 # Fan heat absorption coefficient [-]
        self.r_ext = 0.2 # External unit radius [m]
        self.COP_hp   = 2.5  # Coefficient of performance [-]
        self.dP = 200   # Pressure difference [Pa]

        # Temperature Parameters
        self.T0          = c.C2K(0)  # Environment temperature [K]
        self.T_a_ext_out = c.C2K(-5)  # External unit outlet air temperature [K]
        self.T_w_tank    = c.C2K(60)  # Hot water temperature [K]
        self.T_w_sup     = c.C2K(10)  # Supply water temperature [K]
        self.T_w_tap     = c.C2K(45)  # Tap water temperature [K]
        self.T_r_ext     = c.C2K(-10)  # Refrigerant temperature at external unit [K]
        self.T_r_tank    = c.C2K(65)  # Refrigerant temperature at tank [K]

        # Tank Volume Parameters
        self.water_use_in_a_day = 0.2 # Usable volume [m³/day]
        self.V_tank             = 0.1 # Total tank volume [m³]
        self.n                  = 3   # Height to diameter ratio [-] (height = n * diameter)

        # Heat Transfer Parameters
        self.h_co    = 2    # Convective heat transfer coefficient [W/m²K]
        self.h_ro    = 5    # Radiative heat transfer coefficient [W/m²K]
        self.x_shell = 0.01 # tank shell thickness [m]
        self.k_shell = 50   # tank shell thermal conductivity [W/mK]
        self.x_ins   = 0.10 # Insulation thickness [m]
        self.k_ins   = 0.03 # Insulation thermal conductivity [W/mK]


    def system_update(self): 
        c_w   = 4186  # Specific heat capacity of water [J/kgK]
        rho_w = 1000  # Density of water [kg/m³]
        c_a   = 1005  # Specific heat capacity of air [J/kgK]
        rho_a = 1.225  # Density of air [kg/m³]

        # Water flow rates
        self.dV_w_tap      = self.water_use_in_a_day / (3 * c.h2s)  # Average tap water flow rate [m³/s]
        self.alpha         = (self.T_w_tap - self.T_w_sup) / (self.T_w_tank - self.T_w_sup)  # Mixing ratio
        self.dV_w_sup_tank = self.alpha * self.dV_w_tap  # Supply flow rate to tank [m³/s]
        self.dV_w_sup_mix  = (1 - self.alpha) * self.dV_w_tap  # Supply flow rate to mixing [m³/s]

        # Surface areas
        self.r0 = (self.V_tank / (2 * math.pi * self.n)) ** (1 / 3)  # Tank inner radius [m]
        self.r1 = self.r0 + self.x_shell  # Tank outer radius [m]
        self.r2 = self.r1 + self.x_ins  # Insulation outer radius [m]

        self.h = self.n * (2 * self.r0)  # Tank height [m]

        self.A_top_bottom = 2 * math.pi * self.r0 ** 2  # Total top and bottom area [m²]

        # Thermal resistances
        self.h_o = self.h_co + self.h_ro  # Overall heat transfer coefficient [W/m²K]
        # Cylindrical coordinates ========================================
        self.R_side_shell = math.log((self.r1) / self.r0) / (2 * math.pi * self.k_shell) # Shell thermal resistance [mK/W]
        self.R_side_ins   = math.log((self.r2) / (self.r1)) / (2 * math.pi * self.k_ins) # Insulation thermal resistance [mK/W]
        self.R_side_ext   = 1 / (2 * math.pi * self.r2 * self.h_o)  # External thermal resistance [mK/W]
        self.R_side_tot   = self.R_side_shell + self.R_side_ins + self.R_side_ext # Total side thermal resistance [mK/W]
        self.U_side       = 1 / self.R_side_tot # Overall heat transfer coefficient [W/mK]
        # Cartesian coordinates ==========================================
        self.R_top_bottom_shell = (self.x_shell) / (self.k_shell) # Insulation thermal resistance [m2K/W]
        self.R_top_bottom_ins   = (self.x_ins) / (self.k_ins) # Insulation thermal resistance [m2K/W]
        self.R_top_bottom_ext   = 1/self.h_o  # Combined external thermal resistance [m2K/W]
        self.R_top_bottom_tot   = self.R_top_bottom_shell + self.R_top_bottom_ins + self.R_top_bottom_ext # Total top and bottom thermal resistance [m2K/W]
        self.U_top_bottom       = 1 / self.R_top_bottom_tot # Overall heat transfer coefficient [W/m2K]

        # Total heat transfer coefficient
        self.U_tank = self.U_side*self.h + self.U_top_bottom*self.A_top_bottom # Overall heat transfer coefficient [W/K]

        # Temperature
        self.T_a_ext_in = self.T0  # External unit inlet air temperature [K]
        self.T_tank_is  = self.T_w_tank # inner surface temperature of the tank [K]

        # Fan and Compressor Parameters
        self.A_ext = math.pi * self.r_ext**2  # External unit area [m²] 20 cm x 20 cm assumption

        # Heat transfer
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0) # Tank heat losses
        Q_w_tank      = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0) # Heat transfer from tank water to mixing valve
        Q_w_sup_tank  = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0) # Heat transfer from supply water to tank water

        self.Q_r_tank = self.Q_l_tank + (Q_w_tank - Q_w_sup_tank) # Heat transfer from refrigerant to tank water
        self.E_cmp    = self.Q_r_tank/self.COP_hp  # E_cmp [W]
        self.Q_r_ext  = self.Q_r_tank - self.E_cmp # Heat transfer from external unit to refrigerant

        def fan_equation(V_a_ext): 
            term1 = self.dP * V_a_ext / self.eta_fan # E_fan [W]
            term2 = c_a * rho_a * V_a_ext * (self.T_a_ext_in - self.T_a_ext_out) 
            return term1 + term2 - self.Q_r_ext
        
        def fan_equation_detail(V_a_ext):
            term1 = (1/2 * rho_a * V_a_ext**3) / (self.eta_fan * self.A_ext**2) # Fan power input [W]
            term2 = c_a * rho_a * V_a_ext * (self.T_a_ext_in - self.T0) # Air heat absorption [W] -> outlet air temperature decreases by this
            term3 = (1 - self.eta_fan) * (1 - self.kappa_fan) * term1 # Fan heat absorption [W] -> outlet air temperature increases by this
            term4 = c_a * rho_a * V_a_ext * ((self.T_a_ext_in - self.Q_r_ext / (c_a * rho_a * V_a_ext)) + ((1 - self.eta_fan) * self.kappa_fan * term1) / (c_a * rho_a * V_a_ext) - self.T0)
            return term1 + term2 - term1 / self.A_ext**2 - term3 - term4 
        
        V_a_ext_initial_guess = 1.0

        from scipy.optimize import fsolve
        self.dV_a_ext = fsolve(fan_equation, V_a_ext_initial_guess)[0]
        if self.dV_a_ext < 0: 
            print("Negative air flow rate, check the input temperatures and heat transfer values.")
        self.E_fan   = self.dP * self.dV_a_ext/self.eta_fan  # Power input to external fan [W] (\Delta P = 0.5 * rho * V^2)
        self.v_a_ext = self.dV_a_ext / self.A_ext  # Air velocity [m/s]

        # Energy balances
        self.energy_balance = {
            "external unit": {
                "$E_{fan}$"      : self.E_fan,
                "$E_{r,ext}$"    : - self.Q_r_ext, # out
                "$E_{a,ext,in}$" : c_a * rho_a * self.dV_a_ext * (self.T_a_ext_in - self.T0), # in
                "$E_{a,ext,out}$": c_a * rho_a * self.dV_a_ext * (self.T_a_ext_out - self.T0), # out
            },
            "refrigerant": {
                "$E_{cmp}$"   : self.E_cmp,
                "$E_{r,tank}$": self.Q_r_tank,
                "$E_{r,ext}$" : -self.Q_r_ext,
            },
            "hot water tank": {
                "$X_{r,tank}$"    : self.Q_r_tank,
                "$X_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0),
                "$X_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0),
                "$X_{l,tank}$"    : self.Q_l_tank
            },
            "mixing": {
                "$X_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0),
                "$X_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0),
                "$X_{w,tap}$"     : c_w * rho_w * self.dV_w_tap * (self.T_w_tap - self.T0)
            }
        }

        # Entropy balances
        s_g_ext  = c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_out / self.T0) + (1/self.T_r_ext) * self.Q_r_ext - c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_in / self.T0)
        s_g_r    = (1 / self.T_r_tank) * self.Q_r_tank - (1 / self.T_r_ext) * self.Q_r_ext
        s_g_tank = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0) + (1/self.T_tank_is)*self.Q_l_tank - ((1/self.T_r_tank) * self.Q_r_tank + c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup/self.T0))
        s_g_mix  = c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap / self.T0) - (c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank / self.T0) + c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0))

        self.entropy_balance = {
            "external unit": {
                "$s_{fan}$"      : (1 / float('inf')) * self.E_fan, # in
                "$s_{a,ext,in}$" : c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_in / self.T0), # in
                "$s_{g,ext}$"    : s_g_ext, # gen
                "$s_{r,ext}$"    : (1/self.T_r_ext) * self.Q_r_ext, # out
                "$s_{a,ext,out}$": c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_out / self.T0), # out
            },
            "refrigerant": {
                "$s_{cmp}$"   : (1 / float('inf')) * self.E_cmp,
                "$s_{r,tank}$": -(1 / self.T_r_tank) * self.Q_r_tank,
                "$s_{r,ext}$" : (1 / self.T_r_ext) * self.Q_r_ext,
                "$s_{g,r}$"   : s_g_r,
            },
            "hot water tank": {
                "$s_{r,tank}$"    : (1/self.T_r_tank) * self.Q_r_tank,
                "$s_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup/self.T0),
                "$s_{g,tank}$"    : s_g_tank,
                "$s_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0),
                "$s_{l,tank}$"    : (1/self.T_tank_is)*self.Q_l_tank,
            },
            "mixing": {
                "$s_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank / self.T0),
                "$s_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0),
                "$s_{g,mix}$"     : s_g_mix,
                "$s_{w,tap}$"     : c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap / self.T0),
            }
        }

        # Exergy balances
        self.exergy_balance = {
            "external unit": {
                "$E_{fan}$"      : self.E_fan, # in
                "$X_{r,ext}$"    : -(1-self.T0/self.T_r_ext) * self.Q_r_ext, # out -> in (negative sign)
                "$X_{a,ext,in}$" : c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0)), # in
                "$X_{c,ext}$"    : s_g_ext * self.T0,
                "$X_{a,ext,out}$": c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0)), # out
            },
            "refrigerant": {
                "$E_{cmp}$"   : self.E_cmp,
                "$X_{c,r}$"   : s_g_r * self.T0,
                "$X_{r,tank}$": (1 - self.T0 / self.T_r_tank) * self.Q_r_tank,
                "$X_{r,ext}$" : -(1 - self.T0/self.T_r_ext) * self.Q_r_ext,
            },
            "hot water tank": {
                "$X_{r,tank}$"    : (1-self.T0/self.T_r_tank) * self.Q_r_tank,
                "$X_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup-self.T0) - self.T0 * math.log(self.T_w_sup/self.T0)),
                "$X_{c,tank}$"    : s_g_tank * self.T0,
                "$X_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank-self.T0) - self.T0 * math.log(self.T_w_tank/self.T0)),
                "$X_{l,tank}$"    : (1-self.T0/self.T_tank_is)*self.Q_l_tank,
            },
            "mixing": {
                "$X_{w,tank}$"    : c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank / self.T0)),
                "$X_{w,sup,tank}$": c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0)),
                "$X_{c,mix}$"     : s_g_mix * self.T0,
                "$X_{w,tap}$"     : c_w * rho_w * self.dV_w_tap * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap / self.T0)),
            }
        }