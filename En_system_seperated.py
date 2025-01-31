import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
import constant as c
import math

pi = math.pi

c_air = 1005 # [J/kgK]
rho_air = 1.2 # [kg/m3]

c_w   = 4186 # [J/kgK]
rho_w = 1000 # [kg/m3]

def C2K(temp):
    return temp + 273.15

def K2C(temp):
    return temp - 273.15

@dataclass
class Pump:
    dP   : float # [Pa, J/m3] pressure difference
    Vw   : float # [m3/s]
    eta  : float # [-] 펌프의 투입 전기중 일로 전환되는 비율
    kappa: float # [-] 펌프에서 발생한 열중 유체에 전달되는 비율
    
    def __post_init__(self):
        self.E_pmp  = self.dP * self.Vw / self.eta # [W] 펌프의 전력
        self.W_pmp = self.eta * self.E_pmp # [W] 펌프의 일로 전환되는 전력
        self.Q_pmp_water  = (1-self.eta) * self.kappa * self.E_pmp # [W] 펌프에 의해 유체에 전달되는 열
        self.Q_pmp_loss = self.eta * self.kappa * self.E_pmp # [W] 펌프에 의해 유체에 전달되는 열
        self.dT     = self.Q_pmp_water / (c_w * rho_w * self.Vw) # [K] 펌프에 의해 유체에 전달되는 열로 인한 온도 상승 -> 고려 안하는 듯 


@dataclass
class Compressor:
    pass

@dataclass
class Fan:
    '''
    Fan_coil_unit, 등에 Input으로 들어감
    출력 및 유량을 인풋으로 받게할 것임
    '''
    E_fan: float # [W]
    Va   : float # [m3/s]
    eta  : float # [-] 펌프의 투입 전기중 일로 전환되는 비율
    kappa : float # [-] 펌프에서 발생한 열중 유체에 전달되는 비율
    
    def __post_init__(self):
        self.Q_fan = (1-self.eta) * self.kappa * self.E_fan # [W] 펌프에 의해 유체에 전달되는 열

@dataclass
class Pipe:
    length    : float # [m]
    diameter  : float # [m]
    R_pipe    : float # [m2K/W]
    pump      : Pump  = None
    water_hc  : float = 3000 # [W/m2K] convective heat transfer coefficient of water in pipe (forced convection)
    hco       : float = 10 # [W/m2K] convective heat transfer coefficient of outer surface
    
    def activate_system(self, Vw, T0, Tsur, inlet_water_temp):
        self.inlet_water_temp  = inlet_water_temp # [K]
        
        self.R_tot             = (1/self.water_hc) + self.R_pipe + (1/self.hco)   # [W/m2K] water + pipe + outer surface (thermal resistance)
        self.K_tot             = 1/self.R_tot # [W/mK] total thermal conductivity
        self.cross_area        = pi * self.diameter**2 / 4 # [m2]
        self.pump.v = self.pump.Vw/self.cross_area if self.pump is not None else 0
        self.ksi               = math.exp(- self.K_tot * pi * self.diameter * self.length / (c_w * rho_w * Vw)) # [-]
        self.outlet_water_temp = Tsur + (self.inlet_water_temp - Tsur) * self.ksi + (self.pump.Q_pmp_water/(c_w*rho_w*Vw) if self.pump is not None else 0) # [K] 배관열손실, 펌프의 열전달을 고려한 outlet_water_temp
        self.mass_flow_rate    = Vw * rho_w # [kg/s]
        
        # Energy balance [W]
        self.EN_in  = {
            'pump'       : self.pump.E_pmp if self.pump is not None else 0,
            'inlet_water': c_w * self.mass_flow_rate * (self.inlet_water_temp - T0),
                       }
        self.EN_in['total'] = sum(self.EN_in.values())
        
        self.EN_out = {
            'outlet_water'  : c_w * self.mass_flow_rate * (self.outlet_water_temp - T0),
            'pump_work'     : self.pump.W_pmp if self.pump is not None else 0,
            'pump_heat_gain': self.pump.Q_pmp_water if self.pump is not None else 0,
            'pump_heat_loss': self.pump.Q_pmp_loss if self.pump is not None else 0,
            'pipe_heat_loss': self.EN_in['total'] - (c_w * self.mass_flow_rate * (self.outlet_water_temp - T0) + 
                            (self.pump.W_pmp if self.pump is not None else 0) + 
                            (self.pump.Q_pmp_loss if self.pump is not None else 0)),
            }
        self.EN_out['total'] = sum(self.EN_out.values())
        
        # Entropy balance [W/K]
        self.ENT_in          = {'pump'       : (1/float('inf'))*self.pump.E_pmp if self.pump is not None else 0,
                                'inlet_water': c_w * self.mass_flow_rate * math.log(self.inlet_water_temp/T0)} # Entropy in [W/K]
        self.ENT_in['total'] = sum(self.ENT_in.values())
        self.ENT_out         = {'outlet_water': c_w * self.mass_flow_rate * math.log(self.outlet_water_temp/T0),
                                'pump_work'     : (1/float('inf'))*self.pump.W_pmp if self.pump is not None else 0, # 0
                                'pump_heat_gain': self.pump.Q_pmp_water/Tsur if self.pump is not None else 0,
                                'pump_heat_loss': self.pump.Q_pmp_loss/Tsur if self.pump is not None else 0,
                                'pipe_heat_loss': self.EN_out['pipe_heat_loss']/Tsur} # Entropy out [W/K]
        self.ENT_out['total'] = sum(self.ENT_out.values())
        self.ENT_gen          = self.EN_out['total'] - self.EN_in['total'] # Entropy generation [W/K]
        
        # Exergy balance [W]
        self.EX_in = {'pump'       : self.EN_in['pump'] - T0 * self.ENT_in['pump'],
                      'inlet_water': self.EN_in['inlet_water'] - T0 * self.ENT_in['inlet_water']}
        self.EX_in['total'] = sum(self.EX_in.values())

        self.EX_out = {'outlet_water'  : self.EN_out['outlet_water'] - T0 * self.ENT_out['outlet_water'],
                       'pump_work'     : self.EN_out['pump_work'] - T0 * self.ENT_out['pump_work'],
                       'pump_heat_gain': self.EN_out['pump_heat_gain'] - T0 * self.ENT_out['pump_heat_gain'],
                       'pump_heat_loss': self.EN_out['pump_heat_loss'] - T0 * self.ENT_out['pump_heat_loss'],
                       'pipe_heat_loss': self.EN_out['pipe_heat_loss'] - T0 * self.ENT_out['pipe_heat_loss']}
        self.EX_out['total'] = sum(self.EX_out.values())

        self.EX_con = self.ENT_gen * T0 # Exergy consumption [W]
    
    def info(self, decimal=2):
        print(f"Inlet water temperature: {round(K2C(self.inlet_water_temp), decimal)} °C")
        print(f"Outlet water temperature: {round(K2C(self.outlet_water_temp), decimal)} °C")
        
@dataclass
class Fan_coil_unit:
    capacity        : float # [W] (+: heating, -: cooling)
    Va              : float # [m3/s]
    intake_air_temp : float # [°C]
    fan             : Fan # [-]
    
    def activate_system(self, Vw, T0, inlet_water_temp): 
        self.inlet_water_temp  = inlet_water_temp
        self.outlet_water_temp = self.inlet_water_temp - self.capacity/(c_w * rho_w * Vw)
        self.intake_air_temp   = C2K(self.intake_air_temp) # [K]
        
        self.exhaust_air_temp  = self.intake_air_temp + (self.capacity + self.fan.Q_fan)/(c_air * rho_air * self.Va) # [K]
        self.mean_water_temp   = (self.inlet_water_temp + self.outlet_water_temp) / 2

        # Energy balance [W]
        self.EN_in = {
        'fan'        : self.fan.E_fan,
        'intake_air' : c_air * rho_air * self.Va * (self.intake_air_temp - T0),
        'inlet_water': c_w * rho_w * Vw * (self.inlet_water_temp - T0)
        }
        self.EN_in['total'] = sum(self.EN_in.values())
        self.EN_out = {
        'exhaust_air' : c_air * rho_air * self.Va * (self.exhaust_air_temp - T0),
        'outlet_water': c_w * rho_w * Vw * (self.outlet_water_temp - T0)
        }
        self.EN_out['total'] = sum(self.EN_out.values())

        # Entropy balance [W/K]
        self.ENT_in = {
        'fan'        : (1/float('inf')) * self.fan.E_fan/float('inf'),
        'intake_air' : c_air * rho_air * self.Va * math.log(self.intake_air_temp/T0),
        'inlet_water': c_w * rho_w * Vw * math.log(self.inlet_water_temp/T0)
        }
        self.ENT_in['total'] = sum(self.ENT_in.values())
        self.ENT_out = {
        'exhaust_air' : c_air * rho_air * self.Va * math.log(self.exhaust_air_temp/T0),
        'outlet_water': c_w * rho_w * Vw * math.log(self.outlet_water_temp/T0)
        }
        self.ENT_out['total'] = sum(self.ENT_out.values())
        self.ENT_gen = self.ENT_out['total'] - self.ENT_in['total']

        # Exergy balance [W]
        self.EX_in = {'fan': self.EN_in['fan'] - T0 * self.ENT_in['fan'],
                        'intake_air' : self.EN_in['intake_air'] - T0 * self.ENT_in['intake_air'],
                        'inlet_water': self.EN_in['inlet_water'] - T0 * self.ENT_in['inlet_water']}
        self.EX_in['total'] = sum(self.EX_in.values())
        self.EX_out         = {'exhaust_air': self.EN_out['exhaust_air'] - T0 * self.ENT_out['exhaust_air'],
                        'outlet_water': self.EN_out['outlet_water'] - T0 * self.ENT_out['outlet_water']}
        self.EX_out['total'] = sum(self.EX_out.values())
        self.EX_con          = self.ENT_gen * T0
        
    def info(self, decimal=2):
        print(f"Inlet water temperature: {round(K2C(self.inlet_water_temp), decimal)} °C")
        print(f"Outlet water temperature: {round(K2C(self.outlet_water_temp), decimal)} °C")
        print(f"Exhaust air temperature: {round(K2C(self.exhaust_air_temp), decimal)} °C")
        
@dataclass
class Gas_boiler:
    eta              : float = 0.1 # [-] 폐열 비율
    exhaust_gas_temp : float = 80 # [°C]
    fire_temp        : float = 1000 # [°C]
    outlet_water_temp: float = None # [°C]
    
    def activate_system(self, Vw, T0, inlet_water_temp):
        self.inlet_water_temp = inlet_water_temp # [K]
        self.fire_temp        = C2K(self.fire_temp) # [K]
        self.exhaust_gas_temp = C2K(self.exhaust_gas_temp) # [K]
        self.Q_flame          = c_w * rho_w * Vw * (self.outlet_water_temp - self.inlet_water_temp)/(1-self.eta) # [W]
        
        # Energy balance [W]
        self.EN_in    = {
            'exhaust_air': (1-self.eta)*self.Q_flame,
            'inlet_water': c_w * rho_w * Vw * (self.inlet_water_temp - T0)
            } # [W]
        self.EN_in['total'] = sum(self.EN_in.values())
        self.EN_out   = {
            'flame'       : self.eta*self.Q_flame,
            'outlet_water': c_w * rho_w * Vw * (self.outlet_water_temp - T0)
            } # [W]
        self.EN_out['total'] = sum(self.EN_out.values())
        # Entropy balance [W/K]
        self.ENT_in  = {
            'exhaust_air': (1-self.eta)*self.Q_flame/self.fire_temp,
            'inlet_water': c_w * rho_w * Vw * math.log(self.inlet_water_temp/T0)
            }
        self.ENT_in['total'] = sum(self.ENT_in.values())
        self.ENT_out = {
            'flame'       : self.eta*self.Q_flame/self.fire_temp,
            'outlet_water': c_w * rho_w * Vw * math.log(self.outlet_water_temp/T0)
            }
        self.ENT_out['total'] = sum(self.ENT_out.values()) 
        self.ENT_gen = self.ENT_out['total'] - self.ENT_in['total']

        # Exergy balance [W]
        self.EX_in = {
            'exhaust_air': self.EN_in['exhaust_air'] - T0 * self.ENT_in['exhaust_air'],
            'inlet_water': self.EN_in['inlet_water'] - T0 * self.ENT_in['inlet_water']
        }
        self.EX_in['total'] = sum(self.EX_in.values())
        self.EX_out         = {
            'flame'       : self.EN_out['flame'] - T0 * self.ENT_out['flame'],
            'outlet_water': self.EN_out['outlet_water'] - T0 * self.ENT_out['outlet_water']
        }
        self.EX_out['total'] = sum(self.EX_out.values())
        self.EX_con          = self.EX_in['total'] - self.EX_out['total']
    
    def info(self, decimal=2):
        print(f"Inlet water temperature: {round(K2C(self.inlet_water_temp), decimal)} °C")
        print(f"Outlet water temperature: {round(K2C(self.outlet_water_temp), decimal)} °C")
        print(f"Exhaust gas temperature: {round(K2C(self.exhaust_gas_temp), decimal)} °C")
    

@dataclass
class Single_loop_system:
    '''
    loop는 반시계방향으로 돎
    '''
    T0                       : float
    Tsur                     : float # [°C] system surrounding temperature
    pipe_e2i_inlet_water_temp: float
    pipe_e2i                 : Pipe # Vw,        self.inlet_water_temp, T0 이 정의되어야 activate_psystem이 가능
    internal_unit            : Fan_coil_unit # Vw, self.inlet_water_temp, T0 이 정의되어야 activate_psystem이 가능
    pipe_i2e                 : Pipe # Vw,        self.inlet_water_temp, T0 이 정의되어야 activate_psystem이 가능
    external_unit            : Fan_coil_unit; Gas_boiler # capacity가 정의되어야 activate_psystem이 가능
    
    def __post_init__(self):
        '''
        우선 추후 계산에 필요한 온도들만 모두 정의하고, 각 서브시스템 별 result를 확인할 수 있는 함수를 만들어야 함
        각 시스템들의 activate_system 함수를 실행하는데 필요한 변수는 모두 똑같아야함.
        '''
        # Temperature unit conversion
        self.T0 = C2K(self.T0)
        self.Tsur = C2K(self.Tsur)
        
        # Volumetric flow rate from pump
        if self.pipe_i2e.pump is not None:
            self.volume_flow_rate = self.pipe_i2e.pump.Vw
        if self.pipe_e2i.pump is not None:
            self.volume_flow_rate = self.pipe_e2i.pump.Vw
                 
        # Pipe from external unit to internal unit
        self.pipe_e2i.activate_system(self.volume_flow_rate, self.T0, self.Tsur, C2K(self.pipe_e2i_inlet_water_temp))

        # Internal unit
        self.internal_unit.inlet_water_temp = self.pipe_e2i.outlet_water_temp
        self.internal_unit.activate_system(self.volume_flow_rate, self.T0, self.pipe_e2i.outlet_water_temp)
        
        # Pipe from internal unit to external unit
        self.pipe_i2e.activate_system(self.volume_flow_rate, self.T0, self.Tsur, self.internal_unit.outlet_water_temp)
        
        # External unit
        self.external_unit.inlet_water_temp = self.pipe_i2e.outlet_water_temp
        self.external_unit.outlet_water_temp = self.pipe_e2i.inlet_water_temp
        self.external_unit.activate_system(self.volume_flow_rate, self.T0, self.pipe_i2e.outlet_water_temp)
        
    def show_internal_unit_result(self, decimal=2):
        # Energy balance
        print("Internal Unit Energy Balance ======================")
        print(f"Energy in: {round(self.internal_unit.EN_in['total'], decimal)} W")
        print(f"Energy out: {round(self.internal_unit.EN_out['total'], decimal)} W")
        print('')
        # Entropy balance
        print("Internal Unit Entropy Balance ======================")
        print(f"Entropy in: {round(self.internal_unit.ENT_in['total'], decimal)} W/K")
        print(f"Entropy out: {round(self.internal_unit.ENT_out['total'], decimal)} W/K")
        print(f"Entropy generation: {round(self.internal_unit.ENT_gen, decimal)} W/K")
        print('')
        # Exergy balance
        print("Internal Unit Exergy Balance ======================")
        print(f"Exergy in: {round(self.internal_unit.EX_in['total'], decimal)} W")
        print(f"Exergy out: {round(self.internal_unit.EX_out['total'], decimal)} W")
        print(f"Exergy consumption: {round(self.internal_unit.EX_con, decimal)} W")
        print('')
        
    def show_external_unit_result(self, decimal=2):
        # Energy balance
        print("External Unit Energy Balance ======================")
        print(f"Energy in: {round(self.external_unit.EN_in['total'], decimal)} W")
        print(f"Energy out: {round(self.external_unit.EN_out['total'], decimal)} W")
        print('')
        # Entropy balance
        print("External Unit Entropy Balance ======================")
        print(f"Entropy in: {round(self.external_unit.ENT_in['total'], decimal)} W/K")
        print(f"Entropy out: {round(self.external_unit.ENT_out['total'], decimal)} W/K")
        print(f"Entropy generation: {round(self.external_unit.ENT_gen, decimal)} W/K")
        print('')
        # Exergy balance
        print("External Unit Exergy Balance ======================")
        print(f"Exergy in: {round(self.external_unit.EX_in['total'], decimal)} W")
        print(f"Exergy out: {round(self.external_unit.EX_out['total'], decimal)} W")
        print(f"Exergy consumption: {round(self.external_unit.EX_con, decimal)} W")
        print('')
        
    def show_pipe_e2i_result(self, decimal=2):
        # Energy balance
        print("Pipe e2i Energy Balance ======================")
        print(f"Energy in: {round(self.pipe_e2i.EN_in['total'], decimal)} W")
        print(f"Energy out: {round(self.pipe_e2i.EN_out['total'], decimal)} W")
        print('')
        # Entropy balance
        print("Pipe e2i Entropy Balance ======================")
        print(f"Entropy in: {round(self.pipe_e2i.ENT_in['total'], decimal)} W/K")
        print(f"Entropy out: {round(self.pipe_e2i.ENT_out['total'], decimal)} W/K")
        print(f"Entropy generation: {round(self.pipe_e2i.ENT_gen, decimal)} W/K")
        print('')
        # Exergy balance
        print("Pipe e2i Exergy Balance ======================")
        print(f"Exergy in: {round(self.pipe_e2i.EX_in['total'], decimal)} W")
        print(f"Exergy out: {round(self.pipe_e2i.EX_out['total'], decimal)} W")
        print(f"Exergy consumption: {round(self.pipe_e2i.EX_con, decimal)} W")
        print('')
        
    def show_pipe_i2e_result(self, decimal=2):
        # Energy balance
        print("Pipe i2e Energy Balance ======================")
        print(f"Energy in: {round(self.pipe_i2e.EN_in['total'], decimal)} W")
        print(f"Energy out: {round(self.pipe_i2e.EN_out['total'], decimal)} W")
        print('')
        # Entropy balance
        print("Pipe i2e Entropy Balance ======================")
        print(f"Entropy in: {round(self.pipe_i2e.ENT_in['total'], decimal)} W/K")
        print(f"Entropy out: {round(self.pipe_i2e.ENT_out['total'], decimal)} W/K")
        print(f"Entropy generation: {round(self.pipe_i2e.ENT_gen, decimal)} W/K")
        print('')
        # Exergy balance
        print("Pipe i2e Exergy Balance ======================")
        print(f"Exergy in: {round(self.pipe_i2e.EX_in['total'], decimal)} W")
        print(f"Exergy out: {round(self.pipe_i2e.EX_out['total'], decimal)} W")
        print(f"Exergy consumption: {round(self.pipe_i2e.EX_con, decimal)} W")
        print('')
        
    def show_whole_system_result(self, decimal=2):
        '미완성, 전체 시스템 스케일에서 in, out, consumption을 올바르게 사전정의해야함'
        # Energy balance
        print("Whole System Energy Balance ======================")
        print(f"Exergy in: {round(sum([self.internal_unit.EX_in['total'], self.external_unit.EX_in['total'], self.pipe_e2i.EX_in['total'], self.pipe_i2e.EX_in['total']]), decimal)} W")
        print(f"Exergy out: {round(self.internal_unit.EX_out['total'], decimal)} W")
        print(f"Exergy consumption: {round(sum([self.internal_unit.EX_con, self.external_unit.EX_con, self.pipe_e2i.EX_con, self.pipe_i2e.EX_con]), decimal)} W")


fan1 = Fan(E_fan=100,
           Va=0.1,
           eta=0.8,
           kappa=0.8)

# Pump
pump1 = Pump(dP=1000,
             Vw=0.00003,
             eta=0.8,
             kappa=0.8)
pump2 = Pump(dP=1000,
             Vw=0.1,
             eta=0.8,
             kappa=0.8)

# Pipe
k_metal = 200 # [W/mK]
dx_pipe = 0.01 # [m]
R_pipe = dx_pipe/(k_metal*pi*0.1**2) # [m2K/W]

pipe1 = Pipe(length=10,
             diameter=0.1,
             R_pipe=R_pipe,
             pump=pump1)

pipe2 = Pipe(length=10,
             diameter=0.1,
             R_pipe=R_pipe)

# Fan_coil_unit
fancoil1 = Fan_coil_unit(capacity=1000,
                       Va=0.1,
                       intake_air_temp=20,
                       fan=fan1)

# Gas_boiler
gas_boiler1 = Gas_boiler(eta=0.1,
                         exhaust_gas_temp=80,
                         fire_temp=1000,
                         outlet_water_temp=60)

# Single_loop_system
system1 = Single_loop_system(T0=20,
                             Tsur=20,
                             pipe_e2i_inlet_water_temp=70,
                             pipe_e2i=pipe1,
                             internal_unit=fancoil1,
                             pipe_i2e=pipe2,
                             external_unit=gas_boiler1)

system1.external_unit.info() 
system1.internal_unit.info() 
system1.pipe_e2i.info() 
system1.pipe_i2e.info()
system1.show_internal_unit_result()