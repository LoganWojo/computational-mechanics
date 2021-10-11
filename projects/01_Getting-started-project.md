---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
T1=85
t_1=11
T2=74
t_2=11+2
Tamb=65
dt=t_2-t_1

K=((T2-T1)/dt)/(-(T2-Tamb))
print(round(K,4))
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def K_finder(T1,T2,T_ambiant,Elap_t):
    '''This function takes in an initial and final temp, as well as the ambiant temperature, and the 
    elapsed time to form an equation constant K'''
    T_amb=T_ambiant #assign the ambiant temp
    dt=Elap_t #define dt as the difference between current and starting time
    T_int=T1 #Define the initial temperature at t_int
    T_cur=T2 #Define the current temperature at t_cur
    
    K=((T_cur-T_int)/dt)/(-(T_cur-T_amb))
    return K

ans=K_finder(85,74,65,2)
print(round(ans,4))
```

```{code-cell} ipython3
ans1=K_finder(85,74,65,4)
ans2=K_finder(85,74,65,6)
print(round(ans1,4),round(ans2,4))
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
N=100 #Number of points
t=np.linspace(-0.35,10,N) #Numerical Solution timeframe
t_an=np.linspace(-1,10,N) #Analytical solution timeframe
dt=t[1]-t[0] #numerical timestep

T_zero=98.6
T_am=65
T_hour2=74
t_elap_K=2

K=K_finder(T_zero,T_hour2,T_am,t_elap_K)

T_analytical=T_am+(85-T_am)*np.exp(-K*t_an)

T_num=np.zeros(len(t))
T_num[0]=T_zero
for i in range(len(t)-1):
    T_num[i+1]=((-K*(T_num[i]-T_am))*dt)+T_num[i]

plt.plot(t_an,T_analytical,color='orange',label='analytical');
plt.plot(t,T_num,'o',color='blue',label='numerical');
plt.title('Numerical vs Analytical Temperature of Body solution');
plt.xlabel('Time in Hours');
plt.ylabel('Degrees Fareinheit');
plt.xlim((-1,5));
plt.ylim((60,100));
plt.plot(0,85,'s',markersize=10,color='red');

```

```{code-cell} ipython3
#3.b
cutoff=round(0.8*len(t))
T_inf_range=np.array([T_num[cutoff:]])
T_inf=np.mean(T_inf_range)
print('The temperature as time goes to infinity is {:.4f} degrees fareinheit'.format(T_inf))

#3.c
t_ind_death=t[T_num==98.6]
death_min=60+t_ind_death*60
print("The time of death was at 10:{}AM".format(death_min))
```

4. Now that we have a working numerical model, we can look at the results if the
ambient temperature is not constant i.e. T_a=f(t). We can use the weather to improve our estimate for time of death. Consider the following Temperature for the day in question. 

    |time| Temp ($^o$F)|
    |---|---|
    |6am|50|
    |7am|51|
    |8am|55|
    |9am|60|
    |10am|65|
    |11am|70|
    |noon|75|
    |1pm|80|

    a. Create a function that returns the current temperature based upon the time (0 hours=11am, 65$^{o}$F) 
    *Plot the function $T_a$ vs time. Does it look correct? Is there a better way to get $T_a(t)$?

    b. Modify the Euler approximation solution to account for changes in temperature at each hour. 
    Compare the new nonlinear Euler approximation to the linear analytical model. 
    At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
time_tab=np.array([6,7,8,9,10,11,12,13]) #24 hour time
Temp_tab=np.array([50,51,55,60,65,70,75,80])

def Ambient_Temp(hour_array, Temp_array):
    Ambient_temp=np.zeros(len(hour_array))
    Hour_scaled=np.zeros(len(hour_array))
    for i in range(len(hour_array)):
        if hour_array[i]>=6 and hour_array[i]<7:
            Ambient_temp[i]=Temp_array[i]
            Hour_scaled[i]=-5
        elif hour_array[i]>=7 and hour_array[i]<8:
            Ambient_temp[i]=Temp_array[i]
            Hour_scaled[i]=-4
        elif hour_array[i]>=8 and hour_array[i]<9:
            Ambient_temp[i]=Temp_array[i]
            Hour_scaled[i]=-3
        elif hour_array[i]>=9 and hour_array[i]<10:
            Ambient_temp[i]=Temp_array[i]
            Hour_scaled[i]=-2
        elif hour_array[i]>=10 and hour_array[i]<11:
            Ambient_temp[i]=Temp_array[i]
            Hour_scaled[i]=-1
        elif hour_array[i]>=11 and hour_array[i]<12:
            Ambient_temp[i]=Temp_array[i]
            Hour_scaled[i]=0
        elif hour_array[i]>=12 and hour_array[i]<13:
            Ambient_temp[i]=Temp_array[i]
            Hour_scaled[i]=1
        elif hour_array[i]>=13:
            Ambient_temp[i]=Temp_array[-1]
            Hour_scaled[i]=hour_array[i]-11
    return Ambient_temp , Hour_scaled

Ambient_T,Scaled_time=Ambient_Temp(time_tab,Temp_tab)
plt.plot(Scaled_time,Ambient_T);
plt.title('Ambient Temperature vs 24 hour time');
plt.xlabel('24 Hour Time');
plt.ylabel('Ambient Temp in fahrenheit');
```

## 4a
Based on the table of temperatures over time, this plot does look correct. However, it is not the best model for the ambient temperature. Ambient temperature generally follows an almost bell curve path throughout the day. It is less of a linear line, and closer to a gradual rise and fall associated with a bell curve, or a polynomial representation.

```{code-cell} ipython3
#4b
time_tab=np.array([6,7,8,9,10,11,12,13]) #24 hour time
Temp_tab=np.array([50,51,55,60,65,70,75,80])
Ambient_T,Scaled_time=Ambient_Temp(time_tab,Temp_tab)

N=100 #Number of points
t_an=np.linspace(Scaled_time[1],Scaled_time[-1],N)
t=np.linspace(-0.15,Scaled_time[-1],N)
dt=t[1]-t[0] 

T_zero=98.6
T_am=np.mean(Temp_tab[-3:-1])
T_hour2=74
t_elap_K=2

K=K_finder(T_zero,T_hour2,T_am,t_elap_K)

for i in range(len(Ambient_T)):
    T_analytical=Ambient_T[i]+(85-Ambient_T[i])*np.exp(-K*t_an)

    T_num=np.zeros(len(t))
    T_num[0]=T_zero
    for j in range(len(t)-1):
        T_num[j+1]=((-K*(T_num[j]-Ambient_T[i]))*dt)+T_num[j]

plt.plot(t_an,T_analytical,color='orange',label='analytical');
plt.plot(t,T_num,'o',color='blue',label='numerical');
plt.title('Numerical vs Analytical Temperature of Body solution');
plt.xlabel('Time in Hours');
plt.ylabel('Degrees Fareinheit');
plt.xlim((-0.5,2));
plt.ylim((75,100));
plt.plot(0,85,'s',markersize=10,color='red');
```

```{code-cell} ipython3
cutoff=round(0.8*len(t))
T_inf_range=np.array([T_num[cutoff:]])
T_inf=np.mean(T_inf_range)
print('The temperature as time goes to infinity is {:.4f} degrees fareinheit'.format(T_inf))

t_ind_death=t[T_num==98.6]
death_min=60+t_ind_death*60
print("The time of death was at 10:{}AM".format(death_min))
```

## Project Answer Rationale
   For the constant ambient temperature, there is a much longer time between a body temp of 85 and the ambient temp then there is with the more accurate changing ambient temp. In theory, this does seem to make sense. If the ambient temp was cold compared to the temp of the body, it would take time for the body to get to ambient, since the body is a natural heat source. However, with the changing ambient temp, the temperature around the body is constantly geting warmer. This means there is less heat transfer that needs to happen for the body to get to the surrounding temp after it loses its energy source. 

With this in mind, I must also say that the time of death estimates also match the same logic, but they also seem rather short. For instance the time of death for the changing ambient temperature was about 9 minutes before our initial measurement. meaning the body dropped roughly 14 degrees in 9 minutes, which seemes unlikley. This is most likley because we are using estimation methods.
