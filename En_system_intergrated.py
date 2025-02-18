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

# function
def print_balance(balance, balance_type, decimal=2):
    '''
    📌 Function: print_balance

    이 함수는 주어진 balance 딕셔너리를 이용하여 Energy, Entropy, Exergy balance를 출력합니다.

    🔹 Parameters:
        - balance (dict): Energy, Entropy, Exergy balance 딕셔너리
        - balance_type (str): Balance의 종류 ("energy", "entropy", "exergy")
        - decimal (int, optional): 소수점 이하 출력 자릿수 (기본값: 2)

    🔹 Returns:
        - None (출력만 수행)

    🔹 출력 형식:
        - 서브시스템 별 balance 정보 출력
        - in, out, consumed, generated 등의 카테고리를 구분하여 출력
        - 각 값은 지정된 소수점 자릿수까지 반올림하여 표시

    🔹 Example:
        ```
        print_balance(exergy_balance, "exergy", decimal=2)
        ```

    🔹 실행 예시:
        ```
        HOT WATER TANK EXERGY BALANCE: =====================

        IN ENTRIES:
        $X_{w,comb,out}$: 5000.00 [W]

        OUT ENTRIES:
        $X_{w,tank}$: 4500.00 [W]
        $X_{l,tank}$: 400.00 [W]

        CONSUMED ENTRIES:
        $X_{c,tank}$: 100.00 [W]

        GENERATED ENTRIES:
        $s_{g,tank}$: 50.00 [W/K]
        ```
    '''
    total_length = 50
    dc = decimal
    unit = "[W]" if balance_type in ["energy", "exergy"] else "[W/K]"
    for subsystem, category in balance.items():
        text = f"{subsystem.upper()} {balance_type.upper()} BALANCE:"
        print(f'\n\n{text}'+'='*(total_length-len(text)))
        for category, items in category.items():
            print(f"\n{category.upper()} ENTRIES:")
            for entry in items:
                print(f"{entry['symbol']}: {round(entry['value'],dc)} {unit}")

def calculate_total_exergy_consumption(exergy_balance):
    """
    📌 Function: calculate_total_exergy_consumption

    모든 서브시스템의 엑서지 소비값(consumed exergy)의 전체 합을 계산하는 함수입니다.

    🔹 Parameters:
        - exergy_balance (dict): 각 서브시스템의 엑서지 밸런스 딕셔너리

    🔹 Returns:
        - total_exergy_consumption (float): 전체 엑서지 소비량 [W]

    🔹 Example:
        ```python
        total_exergy = calculate_total_exergy_consumption(exergy_balance)
        print(f"Total Exergy Consumption: {total_exergy} W")
        ```
    """
    total_exergy_consumption = 0

    for subsystem, categories in exergy_balance.items():
        total_exergy_consumption += sum(entry["value"] for entry in categories["consumed"])

    return total_exergy_consumption

@dataclass
class ElectricBoiler:
    def __post_init__(self):
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

        # Energy Balance ========================================
        self.energy_balance = {}
        # hot water tank
        self.energy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$E_{heater}$", "value": self.E_heater},
                {"symbol": "$E_{w,sup,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)},
                {"symbol": "$E_{l,tank}$", "value": self.Q_l_tank},
            ]
        }

        # mixing valve
        self.energy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$E_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)},
                {"symbol": "$E_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * (self.T_w_tap - self.T0)},
            ]
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {}
        # hot water tank
        self.entropy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$s_{heater}$", "value": (1 / float('inf')) * self.E_heater},
                {"symbol": "$s_{w,sup,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup/self.T0)},
            ],
            "out": [
                {"symbol": "$s_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0)},
                {"symbol": "$s_{l,tank}$", "value": (1/self.T_tank_is) * self.Q_l_tank},
            ],
        }
        self.entropy_balance["hot water tank"]["generated"] = [{"symbol": "$s_{g,tank}$", 
                                                                "value": sum(entry["value"] for entry in self.entropy_balance["hot water tank"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["hot water tank"]["in"])}]
        ## mixing valve
        self.entropy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$s_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0)},
                {"symbol": "$s_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup/self.T0)},
            ],
            "out": [
                {"symbol": "$s_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap/self.T0)},
            ],
        }
        self.entropy_balance["mixing valve"]["generated"] = [{"symbol": "$s_{g,mix}$", 
                                                              "value": sum(entry["value"] for entry in self.entropy_balance["mixing valve"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["mixing valve"]["in"])}]


        ## Exergy Balance ========================================
        self.exergy_balance = {}
        # hot water tank
        self.exergy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$E_{heater}$", "value": self.E_heater},
                {"symbol": "$X_{w,sup,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup/self.T0))},
            ],
            "out": [
                {"symbol": "$X_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank/self.T0))},
                {"symbol": "$X_{l,tank}$", "value": (1 - self.T0 / self.T_tank_is) * self.Q_l_tank},
            ],
            "consumed": [
                {"symbol": "$X_{c,tank}$", "value": self.entropy_balance["hot water tank"]["generated"][0]["value"] * self.T0},
            ],
        }
        # mixing valve
        self.exergy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$X_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank/self.T0))},
                {"symbol": "$X_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup/self.T0))},
            ],
            "out": [
                {"symbol": "$X_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap/self.T0))},
            ],
            "consumed": [
                {"symbol": "$X_{c,mix}$", "value": self.entropy_balance["mixing valve"]["generated"][0]["value"] * self.T0},
            ],
        }

@dataclass
class GasBoiler:
    def __post_init__(self):
        # Parameters
        self.eta_boiler = 0.9  # Combustion chamber efficiency [-]
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
        self.E_NG = (c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_boiler - self.T_w_sup)) / self.eta_boiler

        # Heat losses
        self.Q_l_exh = (1 - self.eta_boiler) * self.E_NG  # Heat loss from exhaust gases

        # Temperature calculations
        self.T_NG = self.T0 / (1 - self.eta_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Energy Balance ========================================
        self.energy_balance = {}
        # Combustion chamber
        self.energy_balance["combustion chamber"] = {
            "in": [
                {"symbol": "$E_{NG}$", "value": self.E_NG},
                {"symbol": "$E_{w,sup}$", "value": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_sup - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{w,comb,out}$", "value": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_boiler - self.T0)},
                {"symbol": "$E_{a,exh}$", "value": self.Q_l_exh},
            ]
        }

        # Hot Water Tank
        self.energy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$E_{w,comb,out}$", "value": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_boiler - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_tank - self.T0)},
                {"symbol": "$E_{l,tank}$", "value": self.Q_l_tank},
            ]
        }

        # Mixing Valve
        self.energy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$E_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_boiler * (self.T_w_tank - self.T0)},
                {"symbol": "$E_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * (self.T_w_tap - self.T0)},
            ]
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {}
        # Combustion chamber
        self.entropy_balance["combustion chamber"] = {
            "in": [
                {"symbol": "$s_{NG}$", "value": (1/self.T_NG) * self.E_NG},
                {"symbol": "$s_{w,sup}$", "value": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_sup/self.T0)},
            ],
            "out": [
                {"symbol": "$s_{w,comb,out}$", "value": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_boiler/self.T0)},
                {"symbol": "$s_{a,exh}$", "value": (1/self.T_exh) * self.Q_l_exh},
            ]
        }
        self.entropy_balance["combustion chamber"]["generated"] = [
            {"symbol": "$s_{g,boiler}$", "value": sum(entry["value"] for entry in self.entropy_balance["combustion chamber"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["combustion chamber"]["in"])}
        ]

        # Hot Water Tank
        self.entropy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$s_{w,comb,out}$", "value": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_boiler/self.T0)},
            ],
            "out": [
                {"symbol": "$s_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_tank/self.T0)},
                {"symbol": "$s_{l,tank}$", "value": (1/self.T_tank_is) * self.Q_l_tank},
            ]
        }
        self.entropy_balance["hot water tank"]["generated"] = [
            {"symbol": "$s_{g,tank}$", "value": sum(entry["value"] for entry in self.entropy_balance["hot water tank"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["hot water tank"]["in"])}
        ]

        # Mixing Valve
        self.entropy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$s_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_boiler * math.log(self.T_w_tank/self.T0)},
                {"symbol": "$s_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup/self.T0)},
            ],
            "out": [
                {"symbol": "$s_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap/self.T0)},
            ]
        }
        self.entropy_balance["mixing valve"]["generated"] = [
            {"symbol": "$s_{g,mix}$", "value": sum(entry["value"] for entry in self.entropy_balance["mixing valve"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["mixing valve"]["in"])}
        ]

        ## Exergy Balance ========================================
        self.exergy_balance = {}
        # Combustion chamber
        self.exergy_balance["combustion chamber"] = {
            "in": [
                {"symbol": "$X_{NG}$", "value": self.eta_NG * self.E_NG},
                {"symbol": "$X_{w,sup}$", "value": c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup/self.T0))},
            ],
            "out": [
                {"symbol": "$X_{w,comb,out}$", "value": c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_boiler - self.T0) - self.T0 * math.log(self.T_w_boiler/self.T0))},
                {"symbol": "$X_{a,exh}$", "value": (1 - self.T0/self.T_exh) * self.Q_l_exh},
            ],
            "consumed": [
                {"symbol": "$X_{c,boiler}$", "value": self.entropy_balance["combustion chamber"]["generated"][0]["value"] * self.T0},
            ],
        }

        # Hot Water Tank
        self.exergy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$X_{w,comb,out}$", "value": c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_boiler - self.T0) - self.T0 * math.log(self.T_w_boiler/self.T0))},
            ],
            "out": [
                {"symbol": "$X_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank/self.T0))},
                {"symbol": "$X_{l,tank}$", "value": (1 - self.T0/self.T_tank_is) * self.Q_l_tank},
            ],
            "consumed": [
                {"symbol": "$X_{c,tank}$", "value": self.entropy_balance["hot water tank"]["generated"][0]["value"] * self.T0},
            ],
        }

        # Mixing Valve
        self.exergy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$X_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_boiler * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank/self.T0))},
                {"symbol": "$X_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup/self.T0))},
            ],
            "out": [
                {"symbol": "$X_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap/self.T0))},
            ],
            "consumed": [
                {"symbol": "$X_{c,mix}$", "value": self.entropy_balance["mixing valve"]["generated"][0]["value"] * self.T0},
            ],
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
        
        # External fan air flow rate
        V_a_ext_initial_guess = 1.0
        from scipy.optimize import fsolve
        self.dV_a_ext = fsolve(fan_equation, V_a_ext_initial_guess)[0]
        if self.dV_a_ext < 0: 
            print("Negative air flow rate, check the input temperatures and heat transfer values.")
        self.E_fan   = self.dP * self.dV_a_ext/self.eta_fan  # Power input to external fan [W] (\Delta P = 0.5 * rho * V^2)
        self.v_a_ext = self.dV_a_ext / self.A_ext  # Air velocity [m/s]

        # Energy Balance ========================================
        self.energy_balance = {}
        # External Unit
        self.energy_balance["external unit"] = {
            "in": [
                {"symbol": "$E_{fan}$", "value": self.E_fan},
                {"symbol": "$E_{a,ext,in}$", "value": c_a * rho_a * self.dV_a_ext * (self.T_a_ext_in - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{a,ext,out}$", "value": c_a * rho_a * self.dV_a_ext * (self.T_a_ext_out - self.T0)},
                {"symbol": "$E_{r,ext}$", "value": self.Q_r_ext},
            ]
        }

        # Refrigerant loop
        self.energy_balance["refrigerant loop"] = {
            "in": [
                {"symbol": "$E_{cmp}$", "value": self.E_cmp},
                {"symbol": "$E_{r,ext}$", "value": self.Q_r_ext},
            ],
            "out": [
                {"symbol": "$E_{r,tank}$", "value": self.Q_r_tank},
            ]
        }

        # Hot Water Tank
        self.energy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$E_{r,tank}$", "value": self.Q_r_tank},
                {"symbol": "$E_{w,sup,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)},
                {"symbol": "$E_{l,tank}$", "value": self.Q_l_tank},
            ]
        }

        # Mixing Valve
        self.energy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$E_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)},
                {"symbol": "$E_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)},
            ],
            "out": [
                {"symbol": "$E_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * (self.T_w_tap - self.T0)},
            ]
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {}
        # External Unit
        self.entropy_balance["external unit"] = {
            "in": [
                {"symbol": "$s_{fan}$", "value": (1 / float('inf')) * self.E_fan},
                {"symbol": "$s_{a,ext,in}$", "value": c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_in / self.T0)},
            ],
            "out": [
                {"symbol": "$s_{a,ext,out}$", "value": c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_out / self.T0)},
                {"symbol": "$s_{r,ext}$", "value": (1/self.T_r_ext) * self.Q_r_ext},
            ]
        }
        self.entropy_balance["external unit"]["generated"] = [
            {"symbol": "$s_{g,ext}$", "value": sum(entry["value"] for entry in self.entropy_balance["external unit"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["external unit"]["in"])}
        ]

        # Refrigerant
        self.entropy_balance["refrigerant loop"] = {
            "in": [
                {"symbol": "$s_{cmp}$", "value": (1 / float('inf')) * self.E_cmp},
                {"symbol": "$s_{r,ext}$", "value": (1/self.T_r_ext) * self.Q_r_ext}, # negative sign (in -> out) cool exergy
            ],
            "out": [
                {"symbol": "$s_{r,tank}$", "value": (1/self.T_r_tank) * self.Q_r_tank},
            ]
        }
        self.entropy_balance["refrigerant loop"]["generated"] = [
            {"symbol": "$s_{g,r}$", "value": sum(entry["value"] for entry in self.entropy_balance["refrigerant loop"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["refrigerant loop"]["in"])}
        ]

        # Hot Water Tank
        self.entropy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$s_{r,tank}$", "value": (1/self.T_r_tank) * self.Q_r_tank},
                {"symbol": "$s_{w,sup,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup/self.T0)},
            ],
            "out": [
                {"symbol": "$s_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0)},
                {"symbol": "$s_{l,tank}$", "value": (1/self.T_tank_is) * self.Q_l_tank},
            ]
        }
        self.entropy_balance["hot water tank"]["generated"] = [
            {"symbol": "$s_{g,tank}$", "value": sum(entry["value"] for entry in self.entropy_balance["hot water tank"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["hot water tank"]["in"])}
        ]

        # Mixing Valve
        self.entropy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$s_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank/self.T0)},
                {"symbol": "$s_{w,sup,serv}$", "value": c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup/self.T0)},
            ],
            "out": [
                {"symbol": "$s_{w,serv}$", "value": c_w * rho_w * self.dV_w_tap * math.log(self.T_w_tap/self.T0)},
            ]
        }
        self.entropy_balance["mixing valve"]["generated"] = [
            {"symbol": "$s_{g,mix}$", "value": sum(entry["value"] for entry in self.entropy_balance["mixing valve"]["out"]) - sum(entry["value"] for entry in self.entropy_balance["mixing valve"]["in"])}
        ]

        ## Exergy Balance ========================================
        self.exergy_balance = {}
        # External Unit
        self.exergy_balance["external unit"] = {
            "in": [
                {"symbol": "$E_{fan}$", "value": self.E_fan},
                {"symbol": "$X_{r,ext}$", "value": -(1-self.T0/self.T_r_ext) * self.Q_r_ext}, # negative sign (in -> out) cool exergy
                {"symbol": "$X_{a,ext,in}$", "value": c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))},
            ],
            "consumed": [
                {"symbol": "$X_{c,ext}$", "value": self.entropy_balance["external unit"]["generated"][0]["value"] * self.T0},
            ],
            "out": [
                {"symbol": "$X_{a,ext,out}$", "value": c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))},
            ],
        }
        

        # Refrigerant
        self.exergy_balance["refrigerant loop"] = {
            "in": [
                {"symbol": "$X_{cmp}$", "value": self.E_cmp},
            ],
            "consumed": [
                {"symbol": "$X_{c,r}$", "value": self.entropy_balance["refrigerant loop"]["generated"][0]["value"] * self.T0},
            ],
            "out": [
                {"symbol": "$X_{r,tank}$", "value": (1-self.T0/self.T_r_tank) * self.Q_r_tank},
                {"symbol": "$X_{r,ext}$", "value": -(1-self.T0/self.T_r_ext) * self.Q_r_ext}, # negative sign (in -> out) cool exergy
            ],
        }
        
        # hot water tank
        self.exergy_balance["hot water tank"] = {
            "in": [
                {"symbol": "$X_{r,tank}$", "value": (1 - self.T0 / self.T_r_tank) * self.Q_r_tank},
                {"symbol": "$X_{w,sup,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup/self.T0))},
            ],
            "consumed": [
                {"symbol": "$X_{c,tank}$", "value": self.entropy_balance["hot water tank"]["generated"][0]["value"] * self.T0},
            ],
            "out": [
                {"symbol": "$X_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank/self.T0))},
                {"symbol": "$X_{l,tank}$", "value": (1 - self.T0 / self.T_tank_is) * self.Q_l_tank},
            ],
        }

        # Mixing Valve
        self.exergy_balance["mixing valve"] = {
            "in": [
                {"symbol": "$X_{w,tank}$", "value": c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank/self.T0))},
                {"symbol": "$X_{w,sup,tap}$", "value": c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup/self.T0))},
            ],
            "consumed": [
                {"symbol": "$X_{c,mix}$", "value": self.entropy_balance["mixing valve"]["generated"][0]["value"] * self.T0},
            ],
            "out": [
                {"symbol": "$X_{w,tap}$", "value": c_w * rho_w * self.dV_w_tap * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap/self.T0))},
            ],
        }