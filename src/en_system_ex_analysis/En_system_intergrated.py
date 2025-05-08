import numpy as np
import math
from . import calc_util as cu
# import calc_util as cu
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]

c_w   = 4186 # Water specific heat [J/kgK]
rho_w = 1000
mu_w = 0.001 # Water dynamic viscosity [Pa.s]

sigma = 5.67*10**-8 # Stefan-Boltzmann constant [W/m²K⁴]

# https://www.notion.so/betlab/Scattering-of-photon-particles-coming-from-the-sun-and-their-energy-entropy-exergy-b781821ae9a24227bbf1a943ba9df51a?pvs=4#1ea6947d125d80ddb0a5caec50031ae3
k_D = 0.000462
k_d = 0.0014
#%%
# function
def darcy_friction_factor(Re, e_d):
    '''
    Calculate the Darcy friction factor for given Reynolds number and relative roughness.
    
    Parameters:
    Re (float): Reynolds number
    e_d (float): Relative roughness (e/D)
    
    Returns:
    float: Darcy friction factor
    '''
    # Laminar flow
    if Re < 2300:
        return 64 / Re
    # Turbulent flow
    else:
        return 0.25 / (math.log10(e_d / 3.7 + 5.74 / Re ** 0.9)) ** 2

def calc_h_vertical_plate(T_s, T_inf, L):
    '''
    📌 Function: compute_natural_convection_h_cp
    이 함수는 자연 대류에 의한 열전달 계수를 계산합니다.
    🔹 Parameters
        - T_s (float): 표면 온도 [K]
        - T_inf (float): 유체 온도 [K]
        - L (float): 특성 길이 [m]
    🔹 Return
        - h_cp (float): 열전달 계수 [W/m²K]
    🔹 Example
        ```
        h_cp = compute_natural_convection_h_cp(T_s, T_inf, L)
        ```
    🔹 Note
        - 이 함수는 자연 대류에 의한 열전달 계수를 계산하는 데 사용됩니다.
        - L은 특성 길이로, 일반적으로 물체의 길이나 직경을 사용합니다.
        - 이 함수는 Churchill & Chu 식을 사용하여 열전달 계수를 계산합니다.
    '''
    # 공기 물성치 @ 40°C
    nu = 1.6e-5  # 0.000016 m²/s
    k_air = 0.027 # W/m·K
    Pr = 0.7
    beta = 1 / ((T_s + T_inf)/2) # 1/K
    g = 9.81 # m/s²

    # Rayleigh 수 계산
    delta_T = T_s - T_inf
    Ra_L = g * beta * delta_T * L**3 / (nu**2) * Pr

    # Churchill & Chu 식 https://doi.org/10.1016/0017-9310(75)90243-4
    Nu_L = (0.825 + (0.387 * Ra_L**(1/6)) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
    h_cp = Nu_L * k_air / L  # [W/m²K]
    
    return h_cp

def linear_function(x, a, b):
    return a * x + b

def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

def cubic_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def quartic_function(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def print_balance(balance, decimal=2):
    '''
    📌 Function: print_balance

    이 함수는 주어진 balance 딕셔너리를 이용하여 Energy, Entropy, Exergy balance를 출력합니다.

    🔹 Parameters:
        - balance (dict): Energy, Entropy, Exergy balance 딕셔너리
        - decimal (int, optional): 소수점 이하 출력 자릿수 (기본값: 2)

    🔹 Returns:
        - None (출력만 수행)

    🔹 출력 형식:
        - 서브시스템 별 balance 정보 출력
        - in, out, consumed, generated 등의 카테고리를 구분하여 출력
        - 각 값은 지정된 소수점 자릿수까지 반올림하여 표시

    🔹 Example:
        ```
        print_balance(exergy_balance, decimal=2)
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
        $S_{g,tank}$: 50.00 [W/K]
        ```
    '''
    total_length = 50
    
    balance_type = "energy"
    unit = "[W]"
    
    for subsystem, category_dict in balance.items(): 
        for category, terms in category_dict.items():
            # category: in, out, consumed, generated
            if "gen" in category:
                balance_type = "entropy"
                unit = "[W/K]"
            elif "con" in category:
                balance_type = "exergy"
    
    for subsystem, category_dict in balance.items(): 
        # subsystem: hot water tank, mixing valve...
        # category_dict: {in: {a,b}, out: {a,b}...} 
        text = f"{subsystem.upper()} {balance_type.upper()} BALANCE:"
        print(f'\n\n{text}'+'='*(total_length-len(text)))
        
        for category, terms in category_dict.items():
            # category: in, out, consumed, generated
            # terms: {a,b}
            # a,b..: symbol: value
            print(f"\n{category.upper()} ENTRIES:")
            
            for symbol, value in terms.items():
                print(f"{symbol}: {round(value, decimal)} {unit}")

def calculate_ASHP_cooling_COP(T_a_int_out, T_a_ext_in, Q_r_int, Q_r_max, COP_ref):
    PLR = Q_r_int / Q_r_max
    EIR_by_T = 0.38 + 0.02 * cu.K2C(T_a_int_out) + 0.01 * cu.K2C(T_a_ext_in)
    EIR_by_PLR = 0.22 + 0.50 * PLR + 0.26 * PLR**2
    COP = PLR * COP_ref / (EIR_by_T * EIR_by_PLR)
    return COP

def calculate_ASHP_heating_COP(T0, Q_r_int, Q_r_max):
    PLR = Q_r_int / Q_r_max
    COP = -7.46 * (PLR - 0.0047 * cu.K2C(T0) - 0.477)**2 + 0.0941 * cu.K2C(T0) + 4.34
    return COP

#%%
# class - Fan & Pump
@dataclass
class Fan:
    def __post_init__(self): 
        # Parameters
        self.fan1 = {
            'flow rate'  : [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0], # [m3/s]
            'pressure'   : [140, 136, 137, 147, 163, 178, 182, 190, 198, 181], # [Pa]
            'efficiency' : [0.43, 0.48, 0.52, 0.55, 0.60, 0.65, 0.68, 0.66, 0.63, 0.52], # [-]
        }
        self.fan2 = {
            'flow rate'  : [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5], # [m3/s]
            'pressure'   : [137, 138, 143, 168, 182, 191, 198, 200, 201, 170], # [Pa]
            'efficiency' : [0.45, 0.49, 0.57, 0.62, 0.67, 0.69, 0.68, 0.67, 0.63, 0.40], # [-]
        }
        self.fan_list = [self.fan1, self.fan2]

    def get_effieciency(self, fan, dV_fan):
        self.efficiency_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['efficiency'])
        eff = cubic_function(dV_fan, *self.efficiency_coeffs)
        return eff
    
    def get_pressure(self, fan, dV_fan):
        self.pressure_coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan['pressure'])
        pressure = cubic_function(dV_fan, *self.pressure_coeffs)
        return pressure
    
    def get_power(self, fan, dV_fan):
        eff = self.get_effieciency(fan, dV_fan)
        pressure = self.get_pressure(fan, dV_fan)
        power = pressure * dV_fan / eff
        return power

    def show_graph(self):
        """
        유량(flow rate) 대비 압력(pressure) 및 효율(efficiency) 그래프를 출력.
        - 원본 데이터는 점(dot)으로 표시.
        - 커브 피팅된 곡선은 선(line)으로 표시.
        """
        fig, axes = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ['dm.red3', 'dm.blue3', 'dm.green3', 'dm.orange3']
        plot_colors = ['dm.red6', 'dm.blue6', 'dm.green6', 'dm.orange6']

        data_pairs = [
            ("pressure", "Pressure [Pa]", "Flow Rate vs Pressure"),
            ("efficiency", "Efficiency [-]", "Flow Rate vs Efficiency"),
        ]

        for ax, (key, ylabel, title) in zip(axes, data_pairs):
            print(f"\n{'='*10} {title} {'='*10}")
            for i, fan in enumerate(self.fan_list):
                # 원본 데이터 (dot 형태)
                ax.scatter(fan['flow rate'], fan[key], label=f'Fan {i+1} Data', color=scatter_colors[i], s=2)

                # 곡선 피팅 수행
                coeffs, _ = curve_fit(cubic_function, fan['flow rate'], fan[key])
                flow_range = np.linspace(min(fan['flow rate']), max(fan['flow rate']), 100)
                fitted_values = cubic_function(flow_range, *coeffs)

                # 피팅된 곡선 (line 형태)
                ax.plot(flow_range, fitted_values, label=f'Fan {i+1} Fit', color=plot_colors[i], linestyle='-')
                a,b,c,d = coeffs
                print(f"fan {i+1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

            ax.set_xlabel('Flow Rate [m$^3$/s]', fontsize=dm.fs(0.5))
            ax.set_ylabel(ylabel, fontsize=dm.fs(0.5))
            ax.set_title(title, fontsize=dm.fs(0.5))
            ax.legend()

        plt.subplots_adjust(wspace=0.3)
        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
        dm.save_and_show(fig)

@dataclass
class Pump:
    """
    Pump 클래스: 펌프의 성능 데이터를 저장하고 분석하는 클래스.
    
    - 유량(flow rate)과 효율(efficiency) 데이터를 보유.
    - 효율 데이터를 기반으로 곡선 피팅(curve fitting)을 수행하여 예측 값 계산.
    - 주어진 압력 차이(dP_pmp)와 유량(V_pmp)을 이용하여 펌프의 전력 소비량 계산.
    """

    def __post_init__(self):
        """
        클래스 초기화 후 자동 실행되는 메서드.
        두 개의 펌프의 유량 및 효율 데이터를 저장.
        """
        self.pump1 = {
            'flow rate'  : np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])/cu.h2s, # m3/s
            'efficiency' : [0.255, 0.27, 0.3, 0.33, 0.34, 0.33, 0.32, 0.3, 0.26], # [-]
        }
        self.pump2 = {
            'flow rate'  : np.array([1.8, 2.2, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8])/cu.h2s, # m3/s
            'efficiency' : [0.23, 0.26, 0.29, 0.32, 0.35, 0.34, 0.33, 0.31, 0.28], # [-]
        }
        self.pump_list = [self.pump1, self.pump2]
        
    def get_efficiency(self, pump, dV_pmp):
        """
        주어진 유량(V_pmp)에 대해 3차 곡선 피팅을 통해 펌프 효율을 예측.
        
        :param pump: 선택한 펌프 (self.pump1 또는 self.pump2)
        :param V_pmp: 유량 (m3/h)
        :return: 예측된 펌프 효율
        """
        self.efficiency_coeffs, _ = curve_fit(cubic_function, pump['flow rate'], pump['efficiency'])
        eff = cubic_function(dV_pmp, *self.efficiency_coeffs)
        return eff

    def get_power(self, pump, V_pmp, dP_pmp):
        """
        주어진 유량(V_pmp)과 압력 차이(dP_pmp)를 이용하여 펌프의 전력 소비량을 계산.
        
        :param pump: 선택한 펌프 (self.pump1 또는 self.pump2)
        :param V_pmp: 유량 (m3/h)
        :param dP_pmp: 펌프 압력 차이 (Pa)
        :return: 펌프의 소비 전력 (W)
        """
        efficiency = self.get_efficiency(pump, V_pmp)
        power = (V_pmp * dP_pmp) / efficiency
        return power

    def show_graph(self):
        """
        유량(flow rate) 대비 효율(efficiency) 그래프를 출력.
        - 원본 데이터는 점(dot)으로 표시.
        - 커브 피팅된 곡선은 선(line)으로 표시.
        """
        fig, ax = plt.subplots(figsize=(dm.cm2in(10), dm.cm2in(5)))

        # 그래프 색상 설정
        scatter_colors = ['dm.red3', 'dm.blue3', 'dm.green3', 'dm.orange3']
        plot_colors = ['dm.red6', 'dm.blue6', 'dm.green6', 'dm.orange6']

        for i, pump in enumerate(self.pump_list):
            # 원본 데이터 (dot 형태)
            ax.scatter(pump['flow rate']*cu.h2s, pump['efficiency'], label=f'Pump {i+1} Data', color=scatter_colors[i], s=2)

            # 곡선 피팅 수행
            coeffs, _ = curve_fit(cubic_function, pump['flow rate']*cu.h2s, pump['efficiency'])
            flow_range = np.linspace(min(pump['flow rate']), max(pump['flow rate']), 100)*cu.h2s
            fitted_values = cubic_function(flow_range, *coeffs)

            # 피팅된 곡선 (line 형태)
            a,b,c,d = coeffs
            ax.plot(flow_range, fitted_values, label=f'Pump {i+1} Fit', color=plot_colors[i], linestyle='-')
            print(f"fan {i+1}: {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")

        ax.set_xlabel('Flow Rate [m$^3$/h]', fontsize=dm.fs(0.5))
        ax.set_ylabel('Efficiency [-]', fontsize=dm.fs(0.5))
        ax.legend()

        dm.simple_layout(fig, margins=(0.05, 0.05, 0.05, 0.05), bbox=(0, 1, 0, 1), verbose=False)
        dm.save_and_show(fig)

#%%
# class - Domestic Hot Water System
@dataclass
class ElectricBoiler:
    def __post_init__(self):
        
        # Temperature [K]
        self.T_w_tank = 60
        self.T_w_sup  = 10
        self.T_w_tap  = 45
        self.T0       = 0 

        # Tank water use [m3/s]
        self.dV_w_serv  = 0.0002

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 50   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        
        # Celcius to Kelvin
        self.T_w_tank = cu.C2K(self.T_w_tank) # tank water temperature [K]
        self.T_w_sup  = cu.C2K(self.T_w_sup)  # supply water temperature [K]
        self.T_w_tap  = cu.C2K(self.T_w_tap)  # tap water temperature [K]
        self.T0       = cu.C2K(self.T0)       # reference temperature [K]
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_tap - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_tank = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot

        # Heat Transfer Rates
        self.Q_w_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)
        self.Q_w_sup  = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)
        self.E_heater = self.Q_w_tank + self.Q_l_tank - self.Q_w_sup # Electric Power input [W]

        # Pre-calculate Energy values
        self.Q_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_tap - self.T0)

        # Pre-calculate Entropy values
        self.S_heater = (1 / float('inf')) * self.E_heater
        self.S_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup / self.T0)
        self.S_w_tank = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank / self.T0)
        self.S_l_tank = (1 / self.T_tank_is) * self.Q_l_tank
        self.S_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0)
        self.S_w_serv = c_w * rho_w * self.dV_w_serv * math.log(self.T_w_tap / self.T0)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - (self.S_heater + self.S_w_sup_tank)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)

        # Pre-calculate Exergy values for hot water tank
        self.X_heater = self.E_heater - self.S_heater * self.T0
        self.X_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank / self.T0))
        self.X_l_tank = (1 - self.T0 / self.T_tank_is) * self.Q_l_tank
        self.X_c_tank = self.S_g_tank * self.T0

        # Pre-calculate Exergy values for mixing valve
        self.X_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_serv = c_w * rho_w * self.dV_w_serv * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap / self.T0))
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_heater

        # Energy Balance ========================================
        self.energy_balance = {}
        # hot water tank energy balance (without using lists)
        self.energy_balance["hot water tank"] = {
            "in": {
            "E_heater": self.E_heater,
            "Q_w_sup_tank": self.Q_w_sup_tank
            },
            "out": {
            "Q_w_tank": self.Q_w_tank,
            "Q_l_tank": self.Q_l_tank
            }
        }

        # Mixing valve energy balance (without using lists)
        self.energy_balance["mixing valve"] = {
            "in": {
            "Q_w_tank": self.Q_w_tank,
            "Q_w_sup_mix": self.Q_w_sup_mix
            },
            "out": {
            "Q_w_serv": self.Q_w_serv
            }
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {
            "hot water tank": {
            "in": {
                "S_heater": self.S_heater,
                "S_w_sup_tank": self.S_w_sup_tank
            },
            "out": {
                "S_w_tank": self.S_w_tank,
                "S_l_tank": self.S_l_tank
            },
            "gen": {
                "S_g_tank": self.S_g_tank
            }
            },
            "mixing valve": {
            "in": {
                "S_w_tank": self.S_w_tank,
                "S_w_sup_mix": self.S_w_sup_mix
            },
            "out": {
                "S_w_serv": self.S_w_serv
            },
            "gen": {
                "S_g_mix": self.S_g_mix
            }
            }
        }

        ## Exergy Balance ========================================
        self.exergy_balance = {}
        # Hot water tank exergy balance (without using lists)
        self.exergy_balance["hot water tank"] = {
            "in": {
            "E_heater": self.E_heater,
            "X_w_sup_tank": self.X_w_sup_tank
            },
            "out": {
            "X_w_tank": self.X_w_tank,
            "X_l_tank": self.X_l_tank
            },
            "con": {
            "X_c_tank": self.X_c_tank
            }
        }
        # Mixing valve exergy balance (without using lists)
        self.exergy_balance["mixing valve"] = {
            "in": {
            "X_w_tank": self.X_w_tank,
            "X_w_sup_mix": self.X_w_sup_mix
            },
            "out": {
            "X_w_serv": self.X_w_serv
            },
            "con": {
            "X_c_mix": self.X_c_mix
            }
        }

@dataclass
class GasBoiler:
    def __post_init__(self):
        
        # Efficiency [-]
        self.eta_comb = 0.9
        self.eta_NG   = 0.93

        # Temperature [°C]
        self.T_w_tank = 60 
        self.T_w_sup  = 10
        self.T_w_tap  = 45 
        self.T0       = 0
        self.T_exh    = 70 
        self.T_NG     = 1400

        # Tank water use [m3/s]
        self.dV_w_serv  = 0.0002

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 50   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        
        # Celcius to Kelvin
        self.T_w_tank = cu.C2K(self.T_w_tank) # tank water temperature [K]
        self.T_w_sup  = cu.C2K(self.T_w_sup)  # supply water temperature [K]
        self.T_w_tap  = cu.C2K(self.T_w_tap)  # tap water temperature [K]
        self.T0       = cu.C2K(self.T0)       # reference temperature [K]
        self.T_exh    = cu.C2K(self.T_exh)    # exhaust gas temperature [K]
        self.T_NG     = cu.C2K(self.T_NG)     # natural gas temperature [K]
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank # inner surface temperature of tank [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_tap - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_comb = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base # [K/W]
        self.R_side = self.R_side_unit / self.H # [K/W]
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0)  # Heat loss from tank

        # Temperature [K]
        self.T_w_comb = self.T_w_tank + self.Q_l_tank / (c_w * rho_w * self.dV_w_sup_comb)
        self.T_NG = self.T0 / (1 - self.eta_NG) # eta_NG = 1 - T0/T_NG => T_NG = T0/(1-eta_NG) [K]
        
        # Pre-define variables for balance dictionaries
        self.E_NG     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T_w_sup) / self.eta_comb
        self.Q_w_sup      = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_sup - self.T0)
        self.Q_exh        = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_comb - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_comb * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_tap - self.T0)

        # Pre-calculate Entropy values for boiler
        self.S_NG         = (1 / self.T_NG) * self.E_NG
        self.S_w_sup      = c_w * rho_w * self.dV_w_sup_comb * math.log(self.T_w_sup / self.T0)
        self.S_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * math.log(self.T_w_comb / self.T0)
        self.S_exh        = (1 / self.T_exh) * self.Q_exh
        self.S_g_comb     = (self.S_w_comb_out + self.S_exh) - (self.S_NG + self.S_w_sup)

        self.S_w_tank = c_w * rho_w * self.dV_w_sup_comb * math.log(self.T_w_tank / self.T0)
        self.S_l_tank = (1 / self.T_tank_is) * self.Q_l_tank
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - self.S_w_comb_out

        self.S_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0)
        self.S_w_serv = c_w * rho_w * self.dV_w_serv * math.log(self.T_w_tap / self.T0)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)

        # Pre-calculate Exergy values for boiler
        self.X_NG = self.eta_NG * self.E_NG
        self.X_w_sup = c_w * rho_w * self.dV_w_sup_comb * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_comb_out = c_w * rho_w * self.dV_w_sup_comb * ((self.T_w_comb - self.T0) - self.T0 * math.log(self.T_w_comb / self.T0))
        self.X_exh = (1 - self.T0 / self.T_exh) * self.Q_exh
        self.X_c_comb = self.S_g_comb * self.T0

        self.X_w_tank = c_w * rho_w * self.dV_w_sup_comb * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank / self.T0))
        self.X_l_tank = (1 - self.T0 / self.T_tank_is) * self.Q_l_tank
        self.X_c_tank = self.S_g_tank * self.T0

        self.X_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_serv = c_w * rho_w * self.dV_w_serv * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap / self.T0))
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_comb + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / self.X_NG

        self.energy_balance = {}
        self.energy_balance["combustion chamber"] = {
            "in": {
            "E_NG": self.E_NG,
            "Q_w_sup": self.Q_w_sup
            },
            "out": {
            "Q_w_comb_out": self.Q_w_comb_out,
            "Q_exh": self.Q_exh
            }
        }

        self.energy_balance["hot water tank"] = {
            "in": {
            "Q_w_comb_out": self.Q_w_comb_out
            },
            "out": {
            "Q_w_tank": self.Q_w_tank,
            "Q_l_tank": self.Q_l_tank
            }
        }

        self.energy_balance["mixing valve"] = {
            "in": {
            "Q_w_tank": self.Q_w_tank,
            "Q_w_sup_mix": self.Q_w_sup_mix
            },
            "out": {
            "Q_w_serv": self.Q_w_serv
            }
        }

        self.entropy_balance = {}
        self.entropy_balance["combustion chamber"] = {
            "in": {
            "S_NG": self.S_NG,
            "S_w_sup": self.S_w_sup
            },
            "out": {
            "S_w_comb_out": self.S_w_comb_out,
            "S_exh": self.S_exh
            },
            "gen": {
            "S_g_comb": self.S_g_comb
            }
        }

        self.entropy_balance["hot water tank"] = {
            "in": {
            "S_w_comb_out": self.S_w_comb_out
            },
            "out": {
            "S_w_tank": self.S_w_tank,
            "S_l_tank": self.S_l_tank
            },
            "gen": {
            "S_g_tank": self.S_g_tank
            }
        }
        
        self.entropy_balance["mixing valve"] = {
            "in": {
            "S_w_tank": self.S_w_tank,
            "S_w_sup_mix": self.S_w_sup_mix
            },
            "out": {
            "S_w_serv": self.S_w_serv
            },
            "gen": {
            "S_g_mix": self.S_g_mix
            }
        }

        self.exergy_balance = {}
        self.exergy_balance["combustion chamber"] = {
            "in": {
            "X_NG": self.X_NG,
            "X_w_sup": self.X_w_sup
            },
            "out": {
            "X_w_comb_out": self.X_w_comb_out,
            "X_exh": self.X_exh
            },
            "con": {
            "X_c_comb": self.X_c_comb
            }
        }

        self.exergy_balance["hot water tank"] = {
            "in": {
            "X_w_comb_out": self.X_w_comb_out
            },
            "out": {
            "X_w_tank": self.X_w_tank,
            "X_l_tank": self.X_l_tank
            },
            "con": {
            "X_c_tank": self.X_c_tank
            }
        }

        self.exergy_balance["mixing valve"] = {
            "in": {
            "X_w_tank": self.X_w_tank,
            "X_w_sup_mix": self.X_w_sup_mix
            },
            "out": {
            "X_w_serv": self.X_w_serv
            },
            "con": {
            "X_c_mix": self.X_c_mix
            }
        }

@dataclass
class HeatPumpBoiler: 
    def __post_init__(self): 
        
        # Efficiency [-]
        self.eta_fan = 0.6
        self.COP_hp   = 2.5
        
        # Fan diameter [m]
        self.r_ext = 0.2 
        
        # Pressure [Pa]
        self.dP = 200 

        # Temperature [K]
        self.T0          = 0
        self.T_a_ext_out = -5
        self.T_r_ext     = -10
        self.T_r_tank    = 65
        self.T_w_tank    = 60
        self.T_w_tap     = 45
        self.T_w_sup     = 10

        # Tank water use [m3/s]
        self.dV_w_serv  = 0.0002

        # Tank size [m]
        self.r0 = 0.2
        self.H = 0.8
        
        # Tank layer thickness [m]
        self.x_shell = 0.01 
        self.x_ins   = 0.10 
        
        # Tank thermal conductivity [W/mK]
        self.k_shell = 50   
        self.k_ins   = 0.03 

        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15 
        
    def system_update(self):
        
        # Celcius to Kelvin
        self.T0          = cu.C2K(self.T0)
        self.T_a_ext_out = cu.C2K(self.T_a_ext_out)
        self.T_r_ext     = cu.C2K(self.T_r_ext)
        self.T_r_tank    = cu.C2K(self.T_r_tank)
        self.T_w_tank    = cu.C2K(self.T_w_tank)
        self.T_w_tap     = cu.C2K(self.T_w_tap)
        self.T_w_sup     = cu.C2K(self.T_w_sup)
        
        # Temperature [K]
        self.T_tank_is = self.T_w_tank 
        self.T_a_ext_in = self.T0  # External unit inlet air temperature [K]

        # Surface areas
        self.r1 = self.r0 + self.x_shell
        self.r2 = self.r1 + self.x_ins
        
        # Tank surface areas [m²]
        self.A_side = 2 * math.pi * self.r2 * self.H
        self.A_base = math.pi * self.r0**2
        
        # Total tank volume [m³]
        self.V_tank = self.A_base * self.H

        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_tap - self.T_w_sup)/(self.T_w_tank - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup_tank = self.alp * self.dV_w_serv
        self.dV_w_sup_mix  = (1-self.alp)*self.dV_w_serv

        # Thermal resistances per unit area/legnth
        self.R_base_unit = self.x_shell / self.k_shell + self.x_ins / self.k_ins # [m2K/W]
        self.R_side_unit = math.log(self.r1 / self.r0) / (2 * math.pi * self.k_shell) + math.log(self.r2 / self.r1) / (2 * math.pi * self.k_ins) # [mK/W]
        
        # Thermal resistances [K/W]
        self.R_base = self.R_base_unit / self.A_base
        self.R_side = self.R_side_unit / self.H 
        
        # Thermal resistances [K/W]
        self.R_base_ext = 1 / (self.h_o * self.A_base)
        self.R_side_ext = 1 / (self.h_o * self.A_side)

        # Total thermal resistances [K/W]
        self.R_base_tot = self.R_base + self.R_base_ext
        self.R_side_tot = self.R_side + self.R_side_ext

        # U-value [W/K]
        self.U_tank = 2/self.R_base_tot + 1/self.R_side_tot

        # Fan and Compressor Parameters
        self.A_ext = math.pi * self.r_ext**2  # External unit area [m²] 20 cm x 20 cm assumption

        # Heat transfer
        self.Q_l_tank = self.U_tank * (self.T_tank_is - self.T0) # Tank heat losses
        self.Q_w_tank      = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0) # Heat transfer from tank water to mixing valve
        self.Q_w_sup_tank  = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0) # Heat transfer from supply water to tank water

        self.Q_r_tank = self.Q_l_tank + (self.Q_w_tank - self.Q_w_sup_tank) # Heat transfer from refrigerant to tank water
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

        self.Q_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_sup - self.T0)
        self.Q_w_tank     = c_w * rho_w * self.dV_w_sup_tank * (self.T_w_tank - self.T0)
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_tap - self.T0)
        self.Q_a_ext_in   = c_a * rho_a * self.dV_a_ext * (self.T_a_ext_in - self.T0)
        self.Q_a_ext_out  = c_a * rho_a * self.dV_a_ext * (self.T_a_ext_out - self.T0)

        self.S_fan       = (1 / float('inf')) * self.E_fan
        self.S_a_ext_in  = c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_in / self.T0)
        self.S_a_ext_out = c_a * rho_a * self.dV_a_ext * math.log(self.T_a_ext_out / self.T0)
        self.S_r_ext     = (1 / self.T_r_ext) * self.Q_r_ext
        self.S_cmp       = (1 / float('inf')) * self.E_cmp
        self.S_r_ext_cmp = (1 / self.T_r_ext) * self.Q_r_ext
        self.S_r_tank    = (1 / self.T_r_tank) * self.Q_r_tank
        self.S_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_sup / self.T0)
        self.S_w_tank     = c_w * rho_w * self.dV_w_sup_tank * math.log(self.T_w_tank / self.T0)
        self.S_l_tank     = (1 / self.T_tank_is) * self.Q_l_tank
        self.S_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0)
        self.S_w_serv      = c_w * rho_w * self.dV_w_serv * math.log(self.T_w_tap / self.T0)

        self.S_g_ext = self.S_a_ext_out + self.S_r_ext - (self.S_fan + self.S_a_ext_in)
        self.S_g_r = self.S_r_tank - (self.S_cmp + self.S_r_ext_cmp)
        self.S_g_tank = (self.S_w_tank + self.S_l_tank) - (self.S_r_tank + self.S_w_sup_tank)
        self.S_g_mix = self.S_w_serv - (self.S_w_tank + self.S_w_sup_mix)

        self.X_fan = self.E_fan - self.S_fan * self.T0
        self.X_cmp = self.E_cmp - self.S_cmp * self.T0
        self.X_r_ext = -(1 - self.T0 / self.T_r_ext) * self.Q_r_ext
        self.X_r_tank = (1 - self.T0 / self.T_r_tank) * self.Q_r_tank
        self.X_w_sup_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_tank = c_w * rho_w * self.dV_w_sup_tank * ((self.T_w_tank - self.T0) - self.T0 * math.log(self.T_w_tank / self.T0))
        self.X_l_tank = (1 - self.T0 / self.T_tank_is) * self.Q_l_tank
        self.X_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * ((self.T_w_sup - self.T0) - self.T0 * math.log(self.T_w_sup / self.T0))
        self.X_w_serv = c_w * rho_w * self.dV_w_serv * ((self.T_w_tap - self.T0) - self.T0 * math.log(self.T_w_tap / self.T0))
        self.X_a_ext_in = c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        self.X_a_ext_out = c_a * rho_a * self.dV_a_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))
        
        self.X_c_ext = self.S_g_ext * self.T0
        self.X_c_r = self.S_g_r * self.T0
        self.X_c_tank = self.S_g_tank * self.T0
        self.X_c_mix = self.S_g_mix * self.T0
        
        # total
        self.X_c_tot = self.X_c_ext + self.X_c_r + self.X_c_tank + self.X_c_mix
        self.X_eff = self.X_w_serv / (self.X_fan + self.X_cmp)

        self.energy_balance = {}
        self.energy_balance["external unit"] = {
            "in": {
            "E_fan": self.E_fan,
            "Q_a_ext_in": self.Q_a_ext_in,
            },
            "out": {
            "Q_a_ext_out": self.Q_a_ext_out,
            "Q_r_ext": self.Q_r_ext,
            }
        }

        self.energy_balance["refrigerant loop"] = {
            "in": {
            "E_cmp": self.E_cmp,
            "Q_r_ext": self.Q_r_ext
            },
            "out": {
            "Q_r_tank": self.Q_r_tank
            }
        }

        self.energy_balance["hot water tank"] = {
            "in": {
            "Q_r_tank": self.Q_r_tank,
            "Q_w_sup_tank": self.Q_w_sup_tank
            },
            "out": {
            "Q_w_tank": self.Q_w_tank,
            "Q_l_tank": self.Q_l_tank
            }
        }

        self.energy_balance["mixing valve"] = {
            "in": {
            "Q_w_tank": self.Q_w_tank,
            "Q_w_sup_mix": self.Q_w_sup_mix
            },
            "out": {
            "Q_w_serv": self.Q_w_serv
            }
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {}

        self.entropy_balance["external unit"] = {
            "in": {
            "S_fan": self.S_fan,
            "S_a_ext_in": self.S_a_ext_in
            },
            "out": {
            "S_a_ext_out": self.S_a_ext_out,
            "S_r_ext": self.S_r_ext
            },
            "gen": {
            "S_g_ext": self.S_g_ext
            }
        }

        self.entropy_balance["refrigerant loop"] = {
            "in": {
            "S_cmp": self.S_cmp,
            "S_r_ext": self.S_r_ext_cmp
            },
            "out": {
            "S_r_tank": self.S_r_tank
            },
            "gen": {
            "S_g_r": self.S_g_r
            }
        }

        self.entropy_balance["hot water tank"] = {
            "in": {
            "S_r_tank": self.S_r_tank,
            "S_w_sup_tank": self.S_w_sup_tank
            },
            "out": {
            "S_w_tank": self.S_w_tank,
            "S_l_tank": self.S_l_tank
            },
            "gen": {
            "S_g_tank": self.S_g_tank
            }
        }

        self.entropy_balance["mixing valve"] = {
            "in": {
            "S_w_tank": self.S_w_tank,
            "S_w_sup_mix": self.S_w_sup_mix
            },
            "out": {
            "S_w_serv": self.S_w_serv
            },
            "gen": {
            "S_g_mix": self.S_g_mix
            }
        }

        ## Exergy Balance ========================================
        self.exergy_balance = {}

        self.exergy_balance["external unit"] = {
            "in": {
            "E_fan": self.E_fan,
            "X_r_ext": self.X_r_ext,
            "X_a_ext_in": self.X_a_ext_in
            },
            "con": {
            "X_c_ext": self.X_c_ext
            },
            "out": {
            "X_a_ext_out": self.X_a_ext_out
            }
        }

        self.exergy_balance["refrigerant loop"] = {
            "in": {
            "E_cmp": self.E_cmp
            },
            "con": {
            "X_c_r": self.X_c_r
            },
            "out": {
            "X_r_tank": self.X_r_tank,
            "X_r_ext": self.X_r_ext
            }
        }

        self.exergy_balance["hot water tank"] = {
            "in": {
            "X_r_tank": self.X_r_tank,
            "X_w_sup_tank": self.X_w_sup_tank
            },
            "con": {
            "X_c_tank": self.X_c_tank
            },
            "out": {
            "X_w_tank": self.X_w_tank,
            "X_l_tank": self.X_l_tank
            }
        }

        self.exergy_balance["mixing valve"] = {
            "in": {
            "X_w_tank": self.X_w_tank,
            "X_w_sup_mix": self.X_w_sup_mix
            },
            "con": {
            "X_c_mix": self.X_c_mix
            },
            "out": {
            "X_w_serv": self.X_w_serv
            }
        }

@dataclass
class SolarHotWater:
    def __post_init__(self):
        # Constants [-]
        self.alpha    = 0.95 # Absorptivity of collector
        self.eta_comb = 0.9 # Efficiency of combustion chamber
        self.eta_NG   = 0.93 # Efficiency of natural gas

        # Solar radiation [W/m²]  
        self.I_DN = 800
        self.I_dH = 200
        
        # solar thermal panel
        self.A_stp = 2 # Solar thermal panel area [m²]
        
        # Temperature [°C]
        self.T0       = 0
        self.T_w_comb = 60
        self.T_w_tap  = 45
        self.T_w_sup  = 10
        self.T_exh    = 70
        
        # Tank water use [m3/s]
        self.dV_w_serv = 0.0002
        
        # Overall heat transfer coefficient [W/m²K]
        self.h_o = 15
        self.h_r = 2 # radiative heat transfer coefficient in air layer [W/m²K]
        
        # Thermal conductivity [W/mK]
        self.k_air = 0.025 # Air thermal conductivity [W/mK]
        self.k_ins = 0.03 # Insulation thermal conductivity [W/mK]
        
        # Thickness [m]
        self.x_air = 0.01 # Air layer thickness [m]
        self.x_ins = 0.05 # insulation layer thickness [m]
        
    def system_update(self): 
        # Iradiance [W/m²]
        self.I_sol = self.I_DN + self.I_dH
        
        # Resistance [m2K/W] (conduction)
        self.R_air = self.x_air / self.k_air # [m2K/W]
        self.R_ins = self.x_ins / self.k_ins # [m2K/W]
        self.R_o   = 1/self.h_o
        self.R_r   = 1/self.h_r
        
        self.R1 = (self.R_r * self.R_air)/(self.R_r + self.R_air)
        self.R2 = self.R_ins
        
        # U-value [W/m²K]
        self.U1 = 1 / self.R1
        self.U2 = 1 / self.R2
        self.U  = self.U1 + self.U2 # 병렬
        
        # Demensionless numbers
        self.ksi_sp = np.exp(-self.A_stp * self.U/(c_w * rho_w * self.dV_w_sup))
        
        # Celcius to Kelvin
        self.T0       = cu.C2K(self.T0)
        self.T_w_comb = cu.C2K(self.T_w_comb)
        self.T_w_tap  = cu.C2K(self.T_w_tap)
        self.T_w_sup  = cu.C2K(self.T_w_sup)
        self.T_NG     = cu.C2K(self.T_NG)
        self.T_exh    = cu.C2K(self.T_exh)
        self.T_sp     = cu.C2K(self.T_sp)
        self.T_NG     = self.T0 / (1 - self.eta_NG)
        
        # Volumetric flow rate ratio [-]
        self.alp = (self.T_w_tap - self.T_w_sup)/(self.T_w_comb - self.T_w_sup)
        self.alp = print("alp is negative") if self.alp < 0 else self.alp
        
        # Volumetric flow rates [m³/s]
        self.dV_w_sup     = self.alp * self.dV_w_serv
        self.dV_w_sup_mix = (1-self.alp)*self.dV_w_serv
        
        # Energy balance
        self.Q_w_sup     = c_w * rho_w * self.dV_w_sup * (self.T_w_sup - self.T0)
        self.Q_sol       = self.I_sol * self.A_stp * self.alpha
        
        T_w_stp_out_numerator = self.T0 + (
        self.Q_sol + self.Q_w_sup
        + self.A_stp * self.U * (self.ksi_sp * self.T_w_sup / (1 - self.ksi_sp))
        + self.A_stp * self.U * self.T0
        ) / (c_w * rho_w * self.dV_w_sup)

        T_w_stp_out_denominator = 1 + (self.A_stp * self.U) / ((1 - self.ksi_sp) * c_w * rho_w * self.dV_w_sup)

        self.T_w_stp_out = T_w_stp_out_numerator / T_w_stp_out_denominator
        self.T_sp = 1/(1-self.ksi_sp)*self.T_w_stp_out - self.ksi_sp/(1-self.ksi_sp)*self.T_w_sup

        self.Q_w_stp_out = c_w * rho_w * self.dV_w_sup * (self.T_w_stp_out - self.T0)
        self.Q_l         = self.A_stp * self.U * (self.T_sp - self.T0)
        
        self.E_NG     = c_w * rho_w * self.dV_w_sup * (self.T_w_comb - self.T_w_stp_out) / self.eta_comb
        self.Q_exh    = (1 - self.eta_comb) * self.E_NG  # Heat loss from exhaust gases
        self.Q_w_comb = c_w * rho_w * self.dV_w_sup * (self.T_w_comb - self.T0)
        
        self.Q_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * (self.T_w_sup - self.T0)
        self.Q_w_serv     = c_w * rho_w * self.dV_w_serv * (self.T_w_tap - self.T0)
        
        # Entropy balance
        self.S_w_sup = c_w * rho_w * self.dV_w_sup * math.log(self.T_w_sup / self.T0)
        self.S_DN = k_D * self.I_DN**(0.9)
        self.S_dH = k_d * self.I_dH**(0.9)
        self.S_sol = self.S_DN + self.S_dH
        self.S_w_stp_out = c_w * rho_w * self.dV_w_sup * math.log(self.T_w_stp_out / self.T0)       
        self.S_l = (1 / self.T_sp) * self.A_stp * self.U * (self.T_sp - self.T0)
        self.S_g_stp = self.S_w_stp_out + self.S_l - (self.S_sol + self.S_w_sup)
        
        self.S_NG = (1 / self.T_NG) * self.E_NG
        self.S_exh = (1 / self.T_exh) * self.Q_exh
        self.S_w_comb = c_w * rho_w * self.dV_w_sup * math.log(self.T_w_comb / self.T0)
        self.S_g_comb = (self.S_w_comb + self.S_exh) - (self.S_NG + self.S_w_stp_out)
        
        self.S_w_sup_mix = c_w * rho_w * self.dV_w_sup_mix * math.log(self.T_w_sup / self.T0)
        self.S_w_serv = c_w * rho_w * self.dV_w_serv * math.log(self.T_w_tap / self.T0)
        self.S_g_mix = self.S_w_serv - (self.S_w_sup + self.S_w_sup_mix)
        
        # Exergy balance
        self.X_w_sup = self.Q_w_sup - self.S_w_sup * self.T0
        self.X_sol = self.Q_sol - self.S_sol * self.T0
        self.X_w_stp_out = self.Q_w_stp_out - self.S_w_stp_out * self.T0
        self.X_l = self.Q_l - self.S_l * self.T0
        self.X_c_stp = self.S_g_stp * self.T0

        self.X_NG = self.eta_NG * self.E_NG
        self.X_exh = (1 - self.T0 / self.T_exh) * self.Q_exh
        self.X_w_comb = self.Q_w_comb - self.S_w_comb * self.T0
        self.X_c_comb = self.S_g_comb * self.T0

        self.X_w_sup_mix = self.Q_w_sup_mix - self.S_w_sup_mix * self.T0
        self.X_w_serv = self.Q_w_serv - self.S_w_serv * self.T0 
        self.X_c_mix = self.S_g_mix * self.T0

        self.energy_balance = {}
        self.energy_balance["solar thermal panel"] = {
            "in": {
            "Q_sol": self.Q_sol,
            "Q_w_sup": self.Q_w_sup
            },
            "out": {
            "Q_w_stp_out": self.Q_w_stp_out,
            "Q_l": self.Q_l
            }
        }

        self.energy_balance["combustion chamber"] = {
            "in": {
            "Q_w_stp_out": self.Q_w_stp_out,
            "E_NG": self.E_NG,
            },
            "out": {
            "Q_exh": self.Q_exh,
            "Q_w_comb": self.Q_w_comb
            }
        }

        self.energy_balance["mixing valve"] = {
            "in": {
            "Q_w_comb": self.Q_w_comb,
            "Q_w_sup_mix": self.Q_w_sup_mix
            },
            "out": {
            "Q_w_serv": self.Q_w_serv
            }
        }

        ## Entropy Balance ========================================
        self.entropy_balance = {}

        self.entropy_balance["solar thermal panel"] = {
            "in": {
            "S_sol": self.S_sol,
            "S_w_sup": self.S_w_sup
            },
            "gen": {
            "S_g_stp": self.S_g_stp
            },
            "out": {
            "S_w_stp_out": self.S_w_stp_out,
            "S_l": self.S_l
            }
        }

        self.entropy_balance["combustion chamber"] = {
            "in": {
            "S_w_stp_out": self.S_w_stp_out,
            "S_NG": self.S_NG,
            },
            "gen": {
            "S_g_comb": self.S_g_comb
            },
            "out": {
            "S_exh": self.S_exh,
            "S_w_comb": self.S_w_comb
            }
        }

        self.entropy_balance["mixing valve"] = {
            "in": {
            "S_w_comb": self.S_w_comb,
            "S_w_sup_mix": self.S_w_sup_mix
            },
            "gen": {
            "S_g_mix": self.S_g_mix
            },
            "out": {
            "S_w_serv": self.S_w_serv
            }
        }


        ## Exergy Balance ========================================
        self.exergy_balance = {}

        self.exergy_balance["solar thermal panel"] = {
            "in": {
            "X_sol": self.X_sol,
            "X_w_sup": self.X_w_sup
            },
            "con": {
            "X_c_stp": self.X_c_stp
            },
            "out": {
            "X_w_stp_out": self.X_w_stp_out,
            "X_l": self.X_l
            }
        }

        self.exergy_balance["combustion chamber"] = {
            "in": {
            "X_w_stp_out": self.X_w_stp_out,
            "X_NG": self.X_NG,
            },
            "con": {
            "X_c_comb": self.X_c_comb
            },
            "out": {
            "X_exh": self.X_exh,
            "X_w_comb": self.X_w_comb
            }
        }

        self.exergy_balance["mixing valve"] = {
            "in": {
            "X_w_comb": self.X_w_comb,
            "X_w_sup_mix": self.X_w_sup_mix
            },
            "con": {
            "X_c_mix": self.X_c_mix
            },
            "out": {
            "X_w_serv": self.X_w_serv
            }
        }
              

#%%
# class - AirSourceHeatPump
@dataclass
class AirSourceHeatPump_cooling:
    def __post_init__(self):

        # fan
        self.fan_int = Fan().fan1
        self.fan_ext = Fan().fan2

        # COP
        self.Q_r_max = 10000 # [W]

        # temperature
        self.T0      = cu.C2K(30) # environmental temperature [K]
        self.T_a_room = cu.C2K(20) # room air temperature [K]
        
        self.T_r_int     = cu.C2K(5) # internal unit refrigerant temperature [K]
        self.T_a_int_out = cu.C2K(10) # internal unit air outlet temperature [K]
        
        self.T_a_ext_out = cu.C2K(40) # external unit air outlet temperature [K]
        self.T_r_ext     = cu.C2K(45) # external unit refrigerant temperature [K]
        
        
        # load
        self.Q_r_int = 10000 # [W]
        
        # 
        self.COP_ref = 4

    def system_update(self):

        # temperature
        self.T_a_int_in  = self.T_a_room # internal unit air inlet temperature [K]
        self.T_a_ext_in  = self.T0 # external unit air inlet temperature [K]

        # others
        self.COP     = calculate_ASHP_cooling_COP(self.T_a_int_out, self.T_a_ext_in, self.Q_r_int, self.Q_r_max, self.COP_ref) # COP [-]
        self.E_cmp   = self.Q_r_int / self.COP # compressor power input [W]
        self.Q_r_ext = self.Q_r_int + self.E_cmp # heat transfer from external unit to refrigerant [W]

        # internal, external unit
        self.dV_int = self.Q_r_int / (c_a * rho_a * (abs(self.T_a_int_out - self.T_a_int_in))) # volumetric flow rate of internal unit [m3/s]
        self.dV_ext = self.Q_r_ext / (c_a * rho_a * (abs(self.T_a_ext_out - self.T_a_ext_in))) # volumetric flow rate of external unit [m3/s]

        # fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int) # power input of internal unit fan [W]
        self.E_fan_ext = Fan().get_power(self.fan_ext, self.dV_ext) # power input of external unit fan [W]

        # exergy result 
        self.X_a_int_in  = c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        self.X_a_int_out = c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        self.X_a_ext_in  = c_a * rho_a * self.dV_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        self.X_a_ext_out = c_a * rho_a * self.dV_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))

        self.X_r_int   = - self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_ext   = self.Q_r_ext * (1 - self.T0 / self.T_r_ext)

        # Internal unit of ASHP
        self.Xin_int  = self.E_fan_int + self.X_r_int
        self.Xout_int = self.X_a_int_out - self.X_a_int_in
        self.Xc_int   = self.Xin_int - self.Xout_int

        # Closed refrigerant loop system of ASHP
        self.Xin_r  = self.E_cmp
        self.Xout_r = self.X_r_int + self.X_r_ext
        self.Xc_r   = self.Xin_r - self.Xout_r

        # External unit of ASHP
        self.Xin_ext  = self.E_fan_ext + self.X_r_ext
        self.Xout_ext = self.X_a_ext_out - self.X_a_ext_in
        self.Xc_ext   = self.Xin_ext - self.Xout_ext

        # Total exergy of ASHP
        self.Xin  = self.E_fan_int + self.E_cmp + self.E_fan_ext
        self.Xout = self.X_a_int_out - self.X_a_int_in
        self.Xc   = self.Xin - self.Xout
        
        ## Exergy Balance ========================================
        self.exergy_balance = {}
        # Internal Unit
        self.exergy_balance["internal unit"] = {
            "in": {
            "$E_{f,int}$": self.E_fan_int,
            "$X_{r,int}$": self.X_r_int,
            },
            "con": {
            "$X_{c,int}$": self.Xc_int,
            },
            "out": {
            "$X_{a,int,out}$": self.X_a_int_out,
            "$X_{a,int,in}$": self.X_a_int_in,
            }
        }
        
        # Refrigerant
        self.exergy_balance["refrigerant loop"] = {
            "in": {
            "$E_{cmp}$": self.E_cmp,
            },
            "con": {
            "$X_{c,r}$": self.Xc_r,
            },
            "out": {
            "$X_{r,int}$": self.X_r_int,
            "$X_{r,ext}$": self.X_r_ext,
            }
        }

        # External Unit
        self.exergy_balance["external unit"] = {
            "in": {
            "$E_{f,ext}$": self.E_fan_ext,
            "$X_{r,ext}$": self.X_r_ext,
            },
            "con": {
            "$X_{c,ext}$": self.Xc_ext,
            },
            "out": {
            "$X_{a,ext,out}$": self.X_a_ext_out,
            "$X_{a,ext,in}$": self.X_a_ext_in,
            }
        }

        ## Exergy Balance ========================================
        self.exergy_balance = {}
        # Internal Unit
        self.exergy_balance["internal unit"] = {
            "in": [
                {"symbol": "$E_{f,int}$", "value": self.E_fan_int},
                {"symbol": "$X_{r,int}$", "value": self.X_r_int},
            ],
            "con": [
                {"symbol": "$X_{c,int}$", "value": self.Xc_int},
            ],
            "out": [
                {"symbol": "$X_{a,int,out}$", "value": self.X_a_int_out},
                {"symbol": "$X_{a,int,in}$", "value": self.X_a_int_in},
            ],
            "total": [
                {"symbol": "$X_{in,int}$", "value": self.Xin_int},
                {"symbol": "$X_{c,int}$", "value": self.Xc_int},
                {"symbol": "$X_{out,int}$", "value": self.Xout_int},
            ]
        }
        
        # Refrigerant
        self.exergy_balance["refrigerant loop"] = {
            "in": [
                {"symbol": "$E_{cmp}$", "value": self.E_cmp},
            ],
            "con": [
                {"symbol": "$X_{c,r}$", "value": self.Xc_r},
            ],
            "out": [
                {"symbol": "$X_{r,int}$", "value": self.X_r_int},
                {"symbol": "$X_{r,ext}$", "value": self.X_r_ext},
            ],
            "total": [
                {"symbol": "$X_{in,r}$", "value": self.Xin_r},
                {"symbol": "$X_{c,r}$", "value": self.Xc_r},
                {"symbol": "$X_{out,r}$", "value": self.Xout_r},
            ]
        }

        # External Unit
        self.exergy_balance["external unit"] = {
            "in": [
                {"symbol": "$E_{f,ext}$", "value": self.E_fan_ext},
                {"symbol": "$X_{r,ext}$", "value": self.X_r_ext},
            ],
            "con": [
                {"symbol": "$X_{c,ext}$", "value": self.Xc_ext},
            ],
            "out": [
                {"symbol": "$X_{a,ext,out}$", "value": self.X_a_ext_out},
                {"symbol": "$X_{a,ext,in}$", "value": self.X_a_ext_in},
            ],
            "total": [
                {"symbol": "$X_{in,ext}$", "value": self.Xin_ext},
                {"symbol": "$X_{c,ext}$", "value": self.Xc_ext},
                {"symbol": "$X_{out,ext}$", "value": self.Xout_ext},
            ]
        }

@dataclass
class AirSourceHeatPump_heating:
    def __post_init__(self):

        # fan
        self.fan_int = Fan().fan1
        self.fan_ext = Fan().fan2

        # COP
        self.Q_r_max = 10000 # maximum heating capacity [W]

        # temperature
        self.T0      = cu.C2K(0) # environmental temperature [K]
        self.T_a_room = cu.C2K(20) # room air temperature [K]
        
        self.T_r_int = cu.C2K(35) # internal unit refrigerant temperature [K]
        self.T_a_int_out = cu.C2K(30) # internal unit air outlet temperature [K]
        self.T_a_ext_out = cu.C2K(-10) # external unit air outlet temperature [K]
        self.T_r_ext = cu.C2K(-15) # external unit refrigerant temperature [K]
        
        
        # load
        self.Q_r_int = 10000 # [W]

    def system_update(self):
        # temperature
        self.T_a_int_in  = self.T_a_room
        self.T_a_ext_in  = self.T0 # external unit air inlet temperature [K]

        # others
        self.COP     = calculate_ASHP_heating_COP(T0 = self.T0, Q_r_int = self.Q_r_int, Q_r_max = self.Q_r_max) # COP [-]
        self.E_cmp   = self.Q_r_int / self.COP # compressor power input [W]
        self.Q_r_ext = self.Q_r_int - self.E_cmp # heat transfer from external unit to refrigerant [W]

        # internal, external unit
        self.dV_int = self.Q_r_int / (c_a * rho_a * abs(self.T_a_int_out - self.T_a_int_in)) # volumetric flow rate of internal unit [m3/s]
        self.dV_ext = self.Q_r_ext / (c_a * rho_a * abs(self.T_a_ext_out - self.T_a_ext_in)) # volumetric flow rate of external unit [m3/s]

        # fan power
        self.E_fan_int = Fan().get_power(self.fan_int, self.dV_int) # power input of internal unit fan [W]
        self.E_fan_ext = Fan().get_power(self.fan_ext, self.dV_ext) # power input of external unit fan [W]

        # exergy result 
        self.X_a_int_in  = c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        self.X_a_int_out = c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        self.X_a_ext_in  = c_a * rho_a * self.dV_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        self.X_a_ext_out = c_a * rho_a * self.dV_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))

        self.X_r_int   = self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_ext   = - self.Q_r_ext * (1 - self.T0 / self.T_r_ext)

        # Internal unit of ASHP
        self.Xin_int = self.E_fan_int + self.X_r_int
        self.Xout_int = self.X_a_int_out - self.X_a_int_in
        self.Xc_int = self.E_fan_int + self.X_r_int - (self.X_a_int_out - self.X_a_int_in)

        # Refrigerant loop of ASHP
        self.Xin_r = self.E_cmp
        self.Xout_r = self.X_r_int + self.X_r_ext
        self.Xc_r = self.E_cmp - (self.X_r_int + self.X_r_ext)

        # External unit of ASHP
        self.Xin_ext = self.E_fan_ext + self.X_r_ext
        self.Xout_ext = self.X_a_ext_out - self.X_a_ext_in
        
        self.Xc_ext = self.E_fan_ext + self.X_r_ext - (self.X_a_ext_out - self.X_a_ext_in)

        ## Exergy Balance ========================================
        self.exergy_balance = {}

        # Internal Unit of ASHP
        self.exergy_balance["internal unit"] = {
            "in": {
            "$E_{f,int}$": self.E_fan_int,
            "$X_{r,int}$": self.X_r_int,
            },
            "con": {
            "$X_{c,int}$": self.Xc_int,
            },
            "out": {
            "$X_{a,int,out}$": self.X_a_int_out,
            "$X_{a,int,in}$": self.X_a_int_in,
            }
        }
        
        # Refrigerant loop of ASHP
        self.exergy_balance["refrigerant loop"] = {
            "in": {
            "$E_{cmp}$": self.E_cmp,
            },
            "con": {
            "$X_{c,r}$": self.Xc_r,
            },
            "out": {
            "$X_{r,int}$": self.X_r_int,
            "$X_{r,ext}$": self.X_r_ext,
            }
        }

        # External Unit of ASHP
        self.exergy_balance["external unit"] = {
            "in": {
            "$E_{f,ext}$": self.E_fan_ext,
            "$X_{r,ext}$": self.X_r_ext,
            },
            "con": {
            "$X_{c,ext}$": self.Xc_ext,
            },
            "out": {
            "$X_{a,ext,out}$": self.X_a_ext_out,
            "$X_{a,ext,in}$": self.X_a_ext_in,
            }
        }

#%%
# class - GroundSourceHeatPump
@dataclass
class GroundSourceHeatPump:
    def __post_init__(self):

        # subsystem
        self.fan = Fan().fan1
        self.pump = Pump().pump1

        # efficiency
        self.eta_hp = 0.4 # efficiency of heat pump [-]

        # temperature
        self.dT_a        = 10 # internal unit air temperature difference 
        self.dT_r        = 15 # refrigerant temperature difference 
        self.dT_g        = 5  # circulating water temperature difference
        self.T0         = cu.C2K(32) # environmental temperature [K]
        self.T_g         = cu.C2K(22) # ground temperature [K]
        self.T_a_int_in  = cu.C2K(20) # internal unit air inlet temperature [K]

        # Pipe parameters
        self.L_pipe       = 800 # length of pipe [m]
        self.K_pipe       = 0.2 # thermal conductance of pipe [W/m2K]
        self.D_outer_pipe = cu.mm2m(32) # outer diameter of pipe [m]
        self.pipe_thick   = cu.mm2m(2.9) # thickness of pipe [m]
        self.epsilon_pipe = 0.003e-3 # m

        # plate heat exchanger
        self.N_tot = 20
        self.N_pass = 1
        self.L_ex = 0.203 # m
        self.L_w = 0.108 # m
        self.b = 0.002 # m
        self.lamda = 0.007 # m
        self.beta = 60

        # load
        self.Q_r_int = 10000 # [W]

    def system_update(self):

        # temperature
        self.T_r_int = self.T_a_int_in - self.dT_r # internal unit refrigerant temperature [K]
        self.T_r_ext = self.T_g + self.dT_g # external unit refrigerant temperature [K]
        
        # others
        self.COP = self.eta_hp * self.T_r_int / (self.T_r_ext - self.T_r_int) # COP of GSHP [K]
        
        # pipe parameters
        self.D_inner_pipe      = self.D_outer_pipe - 2 * self.pipe_thick # inner diameter of pipe [m]
        self.A_pipe = math.pi * self.D_inner_pipe ** 2 / 4 # area of pipe [m2]
        self.v_pipe = self.dV_pmp * 0.5 / self.A_pipe # velocity in pipe [m/s] 0.5는 왜 곱하는거지?
        self.e_d    = self.epsilon_pipe / self.D_inner_pipe # relative roughness [-]
        self.Re     = rho_w * self.v_pipe * self.D_inner_pipe / mu_w # Reynolds number [-]
        
        self.f = darcy_friction_factor(self.Re, self.e_d) # darcey friction factor [-]
        self.dP_pipe = self.f * (self.L_pipe) / self.D_inner_pipe * (rho_w * self.v_pipe ** 2) / 2 # pipe pressure drop [Pa]
        self.dP_minor = self.K_pipe * (self.v_pipe ** 2) * (rho_w / 2) # minor loss pressure drop [Pa]

        # plate heatexchanger (이거 너무 복잡한데 따로 빼서 쓸수는 없나) -> 변수명도 수정
        self.N_ch   = int((self.N_tot - 1) / (2 * self.N_pass))
        self.psi    = math.pi * self.b / self.lamda
        self.phi    = (1/6) * (1 + np.sqrt(1 + self.psi**2) + 4 * np.sqrt(1 + (self.psi**2) / 2))
        self.D_ex   = 2 * self.b / self.phi # m
        self.G_c    = self.dV_pmp * rho_w / (self.N_ch * self.b * self.L_w) # [kg/m2s]
        self.Re_ex  = self.G_c * self.D_ex / mu_w # 
        self.f_ex   = 0.8 * self.phi ** (1.25) * self.Re_ex ** (-0.25) * (self.beta/30) ** 3.6 # friction factor [-]
        self.dP_ex  = 2 * self.f_ex * (self.L_ex / self.D_ex) * (self.G_c ** 2) / rho_w # Pa
        self.dP_pmp = self.dP_pipe + self.dP_minor + self.dP_ex # pressure difference of pump [Pa]

        # heat rate
        self.Q_r_ext = self.Q_r_int + self.E_cmp # heat transfer from external unit to refrigerant [W]
        self.Q_g = self.Q_r_ext + self.E_pmp # heat transfer from GHE to ground [W]
        
        # pump, compressor
        self.E_cmp = self.Q_r_int / self.COP # compressor power input [W]
        self.dV_pmp  = self.Q_r_ext / (c_w * rho_w * self.dT_g) # volumetric flow rate of pump [m3/s]
        self.E_pmp   = Pump().get_power(self.pump, self.dV_pmp, self.dP_pmp) # pump power input [W]

        # internal, external unit
        self.dV_int = self.Q_r_int / (c_a * rho_a * self.dT_a) # volumetric flow rate of internal unit [m3/s]
        self.dV_ext = self.Q_r_ext / (c_a * rho_a * self.dT_a) # volumetric flow rate of external unit [m3/s]

        # Circulating water parameters
        self.X_a_int_in  = c_a * rho_a * self.dV_int * ((self.T_a_int_in - self.T0) - self.T0 * math.log(self.T_a_int_in / self.T0))
        self.X_a_int_out = c_a * rho_a * self.dV_int * ((self.T_a_int_out - self.T0) - self.T0 * math.log(self.T_a_int_out / self.T0))
        self.X_a_ext_in  = c_a * rho_a * self.dV_ext * ((self.T_a_ext_in - self.T0) - self.T0 * math.log(self.T_a_ext_in / self.T0))
        self.X_a_ext_out = c_a * rho_a * self.dV_ext * ((self.T_a_ext_out - self.T0) - self.T0 * math.log(self.T_a_ext_out / self.T0))

        ## exergy results
        self.X_r_int = - self.Q_r_int * (1 - self.T0 / self.T_r_int)
        self.X_r_ext = - self.Q_r_ext * (1 - self.T0 / self.T_r_ext)
        self.X_g = - self.Q_g * (1 - self.T0 / self.T_g)

        # Internal unit
        self.Xin_int  = self.E_fan_int + self.X_r_int
        self.Xout_int = self.X_a_int_out - self.X_a_int_in
        self.Xc_int   = self.Xin_int - self.Xout_int

        # Closed refrigerant loop system
        self.Xin_r  = self.E_cmp + self.X_r_ext
        self.Xout_r = self.X_r_int
        self.Xc_r   = self.Xin_r - self.Xout_r

        # External unit
        self.Xin_ext  = self.E_pmp + self.X_g
        self.Xout_ext = self.X_r_ext
        self.Xc_ext   = self.Xin_ext - self.Xout_ext

        # Total exergy
        self.Xin  = self.E_fan_int + self.E_cmp + self.E_pmp
        self.Xout = self.X_a_int_out - self.X_a_int_in
        self.Xc   = self.Xin - self.Xout
 
 #%% 
# class - Electric heater
@dataclass
class ElectricHeater:
    def __post_init__(self): 
        # Panel properties (냉간압연 탄소강판 SPCC)
        self.c_p   = 500 # [J/kgK]
        self.rho_p = 7800 # [kg/m3]
        self.k_p   = 50 # [W/mK]
        
        # Panel geometry [m]
        self.D_p = 0.005 
        self.H_p = 0.5 
        self.W_p = 0.5 
        
        # Electricity input to the heater [W]
        self.E_heater = 1000
        
        # Temperature [°C]
        self.T0   = 20
        self.T_mr = 20
        self.T_p_init = 20 # Initial temperature of the panel [°C]
        self.T_ia = 20 # Inlet air temperature [°C]
        
        
        # Emissivity [-]
        self.epsilon_p = 1
        self.epsilon_r = 1
        
        # Constant
        self.A_eff = 2 
        
        # Time step [s]
        self.dt = 10
    
    def system_update(self):
        
        # Temperature [K]
        self.T0   = cu.C2K(self.T0) # 두번 system update를 할 경우 절대온도 변환 중첩됨
        self.T_mr = cu.C2K(self.T_mr)
        self.T_ia = cu.C2K(self.T_ia)
        self.T_p_init = cu.C2K(self.T_p_init)
        self.T_p  = self.T_p_init
        self.T_ps = self.T_p_init
        
        # Panel properties
        self.C_p = self.c_p * self.rho_p
        self.A_p = self.H_p * self.W_p * 2 # double side 
        self.V_p = self.H_p * self.W_p * self.D_p
        
        # Conductance [W/m²K]
        self.K_cond = self.k_p / (self.D_p / 2)
        
        # Iterative calculation
        self.time = []
        self.T_p_list = []
        self.T_ps_list = []
        
        self.E_heater_list = []
        self.Q_stored_list = []
        self.Q_cond_list = []
        self.Q_conv_ps_list = []
        self.Q_rad_ps_list = []
        self.Q_rad_mr_list = []
        
        self.S_stored_list = []
        self.S_heater_list = []
        self.S_cond_list = []
        self.S_conv_ps_list = []
        self.S_rad_mr_list = []
        self.S_rad_ps_list = []
        self.S_g_body_list = []
        self.S_g_surf_list = []
        
        self.X_stored_list = [] 
        self.X_heater_list = []
        self.X_cond_list = []
        self.X_conv_ps_list = []
        self.X_rad_mr_list = []
        self.X_rad_ps_list = []
        self.X_heater_list = []
        self.X_c_body_list = []
        self.X_c_surf_list = []
        
        index = 0
        tolerance = 1e-4
        while True:
            self.time.append(index * self.dt)
            
            # Heat transfer coefficient [W/m²K]
            self.h_cp = calc_h_vertical_plate(self.T_ps, self.T0, self.H_p) 
            
            def residual_Tp(Tp_new):
                # 축열 항
                Q_stored = self.rho_p * self.c_p * self.V_p * (Tp_new - self.T_p) / self.dt

                # Tps 계산 (표면에너지 평형으로부터)
                Tps = (
                    self.K_cond * Tp_new
                    + self.h_cp * self.T_ia
                    + self.epsilon_p * self.epsilon_r * sigma * (self.T_mr**4 - self.T0**4)
                    - self.epsilon_p * self.epsilon_r * sigma * (Tp_new**4 - self.T0**4)
                ) / (self.K_cond + self.h_cp)

                # 전도열
                Q_cond = self.A_p * self.K_cond * (Tp_new - Tps)

                return Q_stored + Q_cond - self.E_heater
            
            self.T_p_guess = self.T_p # 초기 추정값
            
            from scipy.optimize import fsolve
            self.T_p_next = fsolve(residual_Tp, self.T_p_guess)[0]
            self.T_p_old = self.T_p
            
            # Temperature update
            self.T_p = self.T_p_next
            
            # T_ps update (Energy balance surface: Q_cond + Q_rad_mr = Q_conv_ps + Q_rad_ps)
            self.T_ps = (
                self.K_cond * self.T_p
                + self.h_cp * self.T_ia
                + self.epsilon_p * self.epsilon_r * sigma * (self.T_mr ** 4 - self.T0 ** 4)
                - self.epsilon_p * self.epsilon_r * sigma * (self.T_p ** 4 - self.T0 ** 4)
            ) / (self.K_cond + self.h_cp)
            
            # Temperature [K]
            self.T_p_list.append(self.T_p)
            self.T_ps_list.append(self.T_ps)
            
            # Conduction [W]
            self.Q_stored = self.C_p * self.V_p * (self.T_p_next - self.T_p_old) / self.dt
            self.Q_cond = self.A_p * self.K_cond * (self.T_p - self.T_ps)
            self.Q_conv_ps = self.A_p * self.h_cp * (self.T_ps - self.T_ia) # h_cp 추후 변하게
            self.Q_rad_mr = self.A_p * self.epsilon_p * self.epsilon_r * sigma * (self.T_mr ** 4 - self.T0 ** 4)
            self.Q_rad_ps = self.A_p * self.epsilon_p * self.epsilon_r * sigma * (self.T_p ** 4 - self.T0 ** 4)
            
            self.E_heater_list.append(self.E_heater)
            self.Q_stored_list.append(self.Q_stored)
            self.Q_cond_list.append(self.Q_cond)
            self.Q_conv_ps_list.append(self.Q_conv_ps)
            self.Q_rad_ps_list.append(self.Q_rad_ps)
            self.Q_rad_mr_list.append(self.Q_rad_mr)
            
            # Entropy balance
            self.S_stored = (1/self.T_p) * (self.Q_stored)
            self.S_heater = (1/float('inf')) * (self.E_heater)
            self.S_cond = (1/self.T_p) * (self.Q_cond)
            self.S_conv_ps = (1/self.T_ps) * (self.Q_conv_ps)
            self.S_rad_mr  = 4/3 * self.A_p * self.epsilon_p * self.epsilon_r * sigma * (self.T_mr ** 3 - self.T0 ** 3)
            self.S_rad_ps  = 4/3 * self.A_p * self.epsilon_p * self.epsilon_r * sigma * (self.T_p ** 3 - self.T0 ** 3)
            self.S_g_body = self.S_stored + self.S_conv_ps - self.S_heater     
            self.S_g_surf = self.S_rad_ps + self.S_conv_ps - self.S_cond - self.S_rad_mr

            self.S_stored_list.append(self.S_stored)
            self.S_heater_list.append(self.S_heater)
            self.S_cond_list.append(self.S_cond)
            self.S_conv_ps_list.append(self.S_conv_ps)
            self.S_rad_mr_list.append(self.S_rad_mr)
            self.S_rad_ps_list.append(self.S_rad_ps)
            self.S_g_body_list.append(self.S_g_body)
            self.S_g_surf_list.append(self.S_g_surf)
            
            # Exergy balance
            self.X_stored = (1 - self.T0 / self.T_p) * (self.Q_stored)
            self.X_heater = (1 - self.T0 / float('inf')) * (self.E_heater)
            self.X_cond = (1 - self.T0 / self.T_p) * (self.Q_cond)
            self.X_conv_ps = (1 - self.T0 / self.T_ps) * (self.Q_conv_ps)
            self.X_rad_mr = self.Q_rad_mr - self.T0 * self.S_rad_mr
            self.X_rad_ps = self.Q_rad_ps - self.T0 * self.S_rad_ps
            self.X_c_body = -(self.X_stored + self.X_cond - self.X_heater)
            self.X_c_surf = -(self.X_rad_ps + self.X_conv_ps - self.X_cond - self.X_rad_mr)
            
            self.X_stored_list.append(self.X_stored)
            self.X_heater_list.append(self.X_heater)
            self.X_cond_list.append(self.X_cond)
            self.X_conv_ps_list.append(self.X_conv_ps)
            self.X_rad_mr_list.append(self.X_rad_mr)
            self.X_rad_ps_list.append(self.X_rad_ps)
            self.X_c_body_list.append(self.X_c_body)
            self.X_c_surf_list.append(self.X_c_surf)
            
            index += 1
            T_p_rel_change = abs(self.T_p_next - self.T_p_old) / max(abs(self.T_p_next), 1e-6)
            if T_p_rel_change < tolerance:
                break
            
            if index > 10000:
                print("time step is too short")
                break
            
        
        