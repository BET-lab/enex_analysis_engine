import numpy as np
import math
from . import calc_util as cu
# import calc_util as cu
from dataclasses import dataclass
import dartwork_mpl as dm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.special import erf

#%%
# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]
k_a = 0.0257 # Thermal conductivity of air [W/mK]

c_w   = 4186 # Water specific heat [J/kgK]
rho_w = 1000
mu_w = 0.001 # Water dynamic viscosity [Pa.s]
k_w = 0.606 # Water thermal conductivity [W/mK]

sigma = 5.67*10**-8 # Stefan-Boltzmann constant [W/m²K⁴]

# https://www.notion.so/betlab/Scattering-of-photon-particles-coming-from-the-sun-and-their-energy-entropy-exergy-b781821ae9a24227bbf1a943ba9df51a?pvs=4#1ea6947d125d80ddb0a5caec50031ae3
k_D = 0.000462 # direct solar entropy coefficient [-]
k_d = 0.0014 # diffuse solar entropy coefficient [-]

# Shukuya - Exergy theory and applications in the built environment, 2013
# The ratio of chemical exergy to higher heating value of liquefied natural gas (LNG) is 0.93.
ex_eff_NG   = 0.93 # exergy efficiency of natural gas [-]

SP = np.sqrt(np.pi) # Square root of pi

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
    Pr = 0.7 # Prandtl number 
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
    '''
    https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in cooling mode.

    Parameters:
    - T_a_int_out : Indoor air temperature [K]
    - T_a_ext_in  : Outdoor air temperature [K]
    - Q_r_int     : Indoor heat load [W]
    - Q_r_max     : Maximum cooling capacity [W]

    Defines the COP based on the following parameters:
    - PLR : Part Load Ratio
    - EIR : Energy input to cooling output ratio
    - COP_ref : the reference COP at the standard conditions
    '''
    PLR = Q_r_int / Q_r_max
    EIR_by_T = 0.38 + 0.02 * cu.K2C(T_a_int_out) + 0.01 * cu.K2C(T_a_ext_in)
    EIR_by_PLR = 0.22 + 0.50 * PLR + 0.26 * PLR**2
    COP = PLR * COP_ref / (EIR_by_T * EIR_by_PLR)
    return COP

def calculate_ASHP_heating_COP(T0, Q_r_int, Q_r_max):
    '''
    https://www.mdpi.com/2071-1050/15/3/1880
    Calculate the Coefficient of Performance (COP) for an Air Source Heat Pump (ASHP) in heating mode.

    Parameters:
    - T0 : Enviromnetal temperature [K]
    - Q_r_int : Indoor heat load [W]
    - Q_r_max : Maximum heating capacity [W]

    Defines the COP based on the following parameters:
    - PLR : Part Load Ratio
    '''
    PLR = Q_r_int / Q_r_max
    COP = -7.46 * (PLR - 0.0047 * cu.K2C(T0) - 0.477)**2 + 0.0941 * cu.K2C(T0) + 4.34
    return COP

def calculate_GSHP_COP(Tg, T_cond, T_evap, theta_hat):
    """
    https://www.sciencedirect.com/science/article/pii/S0360544219304347?via%3Dihub
    Calculate the Carnot-based COP of a GSHP system using the modified formula:
    COP = 1 / (1 - T0/T_cond + ΔT * θ̂ / T_cond)

    Parameters:
    - Tg: Undisturbed ground temperature [K]
    - T_cond: Condenser refrigerant temperature [K]
    - T_evap: Evaporator refrigerant temperature [K]
    - theta_hat: θ̂(x0, k_sb), dimensionless average fluid temperature -> 논문 Fig 8 참조, Table 1 참조

    Returns:
    - COP_carnot_modified: Modified Carnot-based COP (float)
    """

    # Temperature difference (ΔT = T0 - T1)
    if T_cond <= T_evap:
        raise ValueError("T_cond must be greater than T_evap for a valid COP calculation.")
    
    delta_T = Tg - T_evap

    # Compute COP using the modified Carnot expression
    denominator = 1 - (Tg / T_cond) + (delta_T /(T_cond*theta_hat))

    if denominator <= 0:
        return float('nan')  # Avoid division by zero or negative COP

    COP = 1 / denominator
    return COP

def f(x):
    return x*erf(x) - (1-np.exp(-x**2))/SP

def chi(s, rb, H, z0=0):
    h = H * s
    d = z0 * s
    
    temp = np.exp(-(rb*s)**2) / (h * s)
    Is = 2*f(h) + 2*f(h+2*d) - f(2*h+2*d) - f(2*d)
    
    return temp * Is

_g_func_cache = {}
def G_FLS(t, ks, as_, rb, H):
    key = (round(t, 0), round(ks, 2), round(as_, 6), round(rb, 2), round(H, 0))
    if key in _g_func_cache:
        return _g_func_cache[key]

    factor = 1 / (4 * np.pi * ks)
    
    lbs = 1 / np.sqrt(4*as_*t)
    
    # Scalar 값인 경우 shape == (,).
    single = len(lbs.shape) == 0
    # 0차원에 1차원으로 변경.
    lbs = lbs.reshape(-1)
        
    # 0 부터 inf 까지의 적분값 미리 계산.
    total = integrate.quad(chi, 0, np.inf, args=(rb, H))[0]
    # ODE 초기값.
    first = integrate.quad(chi, 0, lbs[0], args=(rb, H))[0]
   
    # Scipy의 ODE solver의 인자의 함수 형태는 dydx = f(y, x).
    def func(y, s):
        return chi(s, rb, H, z0=0)
    
    values = total - integrate.odeint(func, first, lbs)[:, 0]
    
    # Single time 값은 첫 번째 값만 선택하여 float를 리턴하도록 함.
    if single:
        values = values[0]

    result = factor * values
    _g_func_cache[key] = result
    return result


@dataclass
class DynamicWaterHeater:
    """
    @dataclass를 사용하여 전기 온수기 저탕조의 동적 에너지 모델을 재구성한 클래스.
    이전 ElectricBoiler의 단위 및 변수 체계를 따릅니다.
    """
    # --- 입력 속성 (Input Attributes) ---
    # 탱크 기하학 및 단열 정보
    tank_diameter: float
    tank_height: float
    insulation_conductivity: float
    insulation_thickness: float
    
    # 히터 및 초기 조건
    heater_max_capacity: float
    initial_T_st: float

    # --- 계산된 속성 (Calculated Attributes) ---
    V_st: float = field(init=False)      # 탱크 용적 (Storage tank volume) [m³]
    UA_loss: float = field(init=False)   # 총괄열전달계수 (Overall heat transfer coefficient) [W/K]
    C_st: float = field(init=False)      # 탱크 열용량 (Heat capacity of the tank) [J/K]
    results: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """초기화 후 파생 변수들을 계산합니다."""
        # 탱크 용적 계산
        self.V_st = np.pi * (self.tank_diameter / 2)**2 * self.tank_height
        # 탱크 열용량 계산
        self.C_st = self.V_st * rho_w * c_w
        # 총괄열전달계수 계산
        self.UA_loss = self._calculate_ua_loss()

    def _calculate_ua_loss(self) -> float:
        """탱크의 기하학적 정보를 바탕으로 총괄열전달계수(UA_loss)를 계산합니다."""
        # 표면적 계산
        A_side = np.pi * self.tank_diameter * self.tank_height
        A_tb = np.pi * (self.tank_diameter / 2)**2  # Top/Bottom area

        # 단열재의 열 저항이 열 손실을 지배한다고 가정하여 U-value 계산
        U_value = self.insulation_conductivity / self.insulation_thickness

        # 각 부위의 UA 값 계산
        UA_side = U_value * A_side
        UA_tb = U_value * A_tb

        # 위, 아래, 옆면의 UA 값을 모두 더함
        return UA_side + 2 * UA_tb

    def run_simulation(self,
                       T_set_sch: np.ndarray,
                       T_out_sch: np.ndarray,
                       V_hw_sch: np.ndarray,
                       T_su: float,
                       timestep_duration: int):
        """
        주어진 스케줄에 따라 동적 시뮬레이션을 실행합니다.

        Args:
            T_set_sch (np.ndarray): 타임스텝별 탱크 목표 온도 스케줄 [°C]
            T_out_sch (np.ndarray): 타임스텝별 외기 온도 스케줄 [°C]
            V_hw_sch (np.ndarray): 타임스텝별 온수 사용 체적 유량 스케줄 [L/min]
            T_su (float): 탱크로 공급되는 상수도의 온도 (일정하다고 가정) [°C]
            timestep_duration (int): 각 타임스텝의 시간 간격 [seconds]
        """
        num_timesteps = len(T_set_sch)
        self._initialize_results(num_timesteps)
        self.results["T_st"][0] = self.initial_T_st

        for t in range(1, num_timesteps):
            prev_T_st = self.results["T_st"][t-1]

            T_set = T_set_sch[t]
            Q_heater = 0.0
            if prev_T_st < T_set:
                q_required = self.C_st * (T_set - prev_T_st) / timestep_duration
                Q_heater = min(q_required, self.heater_max_capacity)
            self.results["Q_heater"][t] = Q_heater

            T_out = T_out_sch[t]
            Q_loss = self.UA_loss * (prev_T_st - T_out)
            self.results["Q_loss"][t] = Q_loss
            
            # 체적유량(L/min)을 질량유량(kg/s)으로 변환
            V_hw_lpm = V_hw_sch[t]
            V_hw_mps = V_hw_lpm / (60 * 1000) # L/min -> m³/s
            m_dot = V_hw_mps * rho_w
            
            Q_use_out = m_dot * c_w * prev_T_st
            Q_supply_in = m_dot * c_w * T_su

            delta_energy = (Q_heater - Q_loss - Q_use_out + Q_supply_in) * timestep_duration
            new_tank_energy = self.C_st * prev_T_st + delta_energy
            current_T_st = new_tank_energy / self.C_st
            self.results["T_st"][t] = current_T_st

    def _initialize_results(self, num_timesteps: int):
        self.results["T_st"] = np.zeros(num_timesteps)
        self.results["Q_heater"] = np.zeros(num_timesteps)
        self.results["Q_loss"] = np.zeros(num_timesteps)

    def print_results_summary(self, num_steps_to_print: int = 10):
        print("-" * 80)
        print(f"{'Time (min)':>12} | {'Setpoint (°C)':>15} | {'Tank Temp (°C)':>15} | {'Heater (W)':>12} | {'Heat Loss (W)':>15}")
        print("-" * 80)
        for t in range(num_steps_to_print):
            time_min = t * timestep_duration / 60
            setpoint = T_set_sch[t]
            tank_temp = self.results["T_st"][t]
            heater_w = self.results["Q_heater"][t]
            loss_w = self.results["Q_loss"][t]
            print(f"{time_min:12.1f} | {setpoint:15.2f} | {tank_temp:15.2f} | {heater_w:12.1f} | {loss_w:15.2f}")
        print("-" * 80)

    def plot_results(self):
        num_timesteps = len(self.results["T_st"])
        time_hours = np.arange(num_timesteps) * timestep_duration / 3600

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(time_hours, self.results["T_st"], 'b-', label="Tank Temperature (T_st)")
        ax1.plot(time_hours, T_set_sch, 'r--', label="Setpoint Temperature (T_set)")
        ax1.plot(time_hours, T_out_sch, 'g:', label="Ambient Temperature (T_out)")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Temperature (°C)", color='b')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(time_hours, self.results["Q_heater"], 'm-.', label="Heater Output (Q_heater)")
        ax2.plot(time_hours, V_hw_sch, 'k:', label="Hot Water Usage (V_hw)")
        ax2.set_ylabel("Power (W) / Flow (L/min)", color='m')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.title("Dynamic Water Heater Simulation")
        plt.show()

# ==============================================================================
# --- 예제 코드: 모델 사용법 (EXAMPLE: How to use the model) ---
# ==============================================================================
if __name__ == '__main__':
    total_sim_hours = 24
    timestep_duration = 60
    num_timesteps = int(total_sim_hours * 3600 / timestep_duration)
    
    my_heater = DynamicWaterHeater(
        tank_diameter=0.5,           # 50 cm
        tank_height=1.0,             # 1 m
        insulation_conductivity=0.04,# 0.04 W/mK
        insulation_thickness=0.05,   # 5 cm
        heater_max_capacity=3000,    # 3 kW heater
        initial_T_st=50.0
    )
    # 초기화 후 계산된 값 확인
    print(f"Calculated Tank Volume (V_st): {my_heater.V_st:.3f} m³")
    print(f"Calculated UA_loss: {my_heater.UA_loss:.3f} W/K")

    T_set_sch = np.full(num_timesteps, 60.0)
    T_out_sch = 20 - 5 * np.cos(2 * np.pi * np.arange(num_timesteps) / num_timesteps)
    T_su = 10.0
    
    V_hw_sch = np.zeros(num_timesteps) # Unit: L/min
    liters_per_minute = 8
    
    start_idx_morning = int(7 * 3600 / timestep_duration)
    end_idx_morning = int(8 * 3600 / timestep_duration)
    V_hw_sch[start_idx_morning:end_idx_morning] = liters_per_minute
    
    start_idx_evening = int(20 * 3600 / timestep_duration)
    end_idx_evening = int(21 * 3600 / timestep_duration)
    V_hw_sch[start_idx_evening:end_idx_evening] = liters_per_minute

    my_heater.run_simulation(
        T_set_sch,
        T_out_sch,
        V_hw_sch,
        T_su,
        timestep_duration
    )
    
    my_heater.print_results_summary(num_steps_to_print=15)
    my_heater.plot_results()
