# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 17:50:40 2016

@author: neal
"""



from scipy import signal
from pylab import plot, xlabel, ylabel, title, rcParams, figure
import numpy as np
pltwidth = 16
pltheight = 8
rcParams['figure.figsize'] = (pltwidth, pltheight)

csv = np.genfromtxt('./stress_strain1.csv', delimiter=",")
disp = csv[:,0]
force = csv[:,1]
print('number of data points = #i' # len(disp))

def moving_average(x, window):
    """Moving average of 'x' with window size 'window'."""
    y = np.empty(len(x)-window+1)
    for i in range(len(y)):
        y[i] = np.sum(x[i:i+window])/window
    return y

plt1 = plot(disp, force)
xlabel('displacement')
ylabel('force')




#case 'Stress-Strain with data table'


A = 1

stress = force/A
strain = disp/25.4

stress_range = np.array([5, 15])
PL = 0.005
E_sec = stress/strain
# Initialize table

for y = 1:length(strain)
    if stress(y) >= stress_range(2)
        break


PLindex = find( (stress(1:y) < stress_range(2) & stress(1:y) > stress_range(1)) )

stressPL = stress(PLindex)
strainPL = strain(PLindex)

# perform the fit and plot
linfit = fit(strainPL+PL,stressPL,'poly1')
# stressPLfit = linfit.p1.*(strainPL-PL) + linfit.p2
allstressPLfit = linfit.p1.*(strain) + linfit.p2
E_0 = linfit.p1/10

PLPointindex = find( allstressPLfit >= stress ,1,'first')

if isempty(PLPointindex)
    stressPLplot = allstressPLfit(allstressPLfit>=0)
    strainPLplot = strain(allstressPLfit>=0)
else
    stressPLplot = allstressPLfit(allstressPLfit(1:PLPointindex)>=0)
    strainPLplot = strain(allstressPLfit(1:PLPointindex)>=0)
end

# if the PL limit was not reached
if isempty(PLPointindex)
    PLPointindex = length(stress)
end  

plot(curaxes,strain,stress,linecol{k})
hold on
if k == 1
    maxstrain = 0
    maxstress = 0
end
if max(strain) > maxstrain
    maxstrain = max(strain)
end
if max(stress) > maxstress
    maxstress = max(stress)
end
set(curaxes,'XGrid','on','YGrid','on')
#     set(handles.axes1,'XTick',[0:0.1:max(strain)],'YTick',[0:100:max(stress)])
set(curaxes,'YLim',[0 maxstress*1.1],'XLim',[0 maxstrain*1.1])
title('Stress-Strain Curve')
xlabel('\epsilon (#)')
ylabel('\sigma (MPa)')
plot(strainPLplot,stressPLplot,'Linestyle', ':')
scatter(curaxes, [min(strainPL) max(strainPL)] ,[min(stressPL) max(stressPL)])
scatter(curaxes, strain(PLPointindex), stress(PLPointindex))
hold off
UT_stress = max(stress)
UT_strain = max(strain) 
PL_stress = stress(PLPointindex)
PL_strain = strain(PLPointindex)
enloudind = log10(en)>=1
enloud = en(enloudind)
stressloud = stresser(enloudind)
tabledat(k,:) = { sprintf('#3.0f',E_0)    sprintf('#0.3f',PL_strain)     sprintf('#3.0f',PL_stress)       sprintf('#0.3f',UT_strain)     sprintf('#3.0f',UT_stress)         sprintf('#3.0f',stresser(1))      sprintf('#3.0f',stressloud(1))        sprintf('#3.0f',stressloud(5))  numel(en)  }
# complete table with data for multi-selection
set(uitable_results , 'Data', tabledat)
# plot(handles.axes1,strainPL,stressPL,'r')          
newfontsize = 12       
set(gcf,'Position',[150   150   1000   475],'Color',[1 1 1])



"""

#case 'Stress-Strain tangent modulus and energy calcs'
## 
# calculations may vary depending on the smoothness of
#
# parameters to adjust to make calculations better
# span = smoothing parameter
# span2 = smoothing parameter
# deltajump = jump when finding tangent modulus

set(gcf,'position',[35 1133 1254 702])
if length(specname) > 1
    errordlg('Cannot perform multispecimen analysis')
    close(curfig)
    return
end
# Hz=sample rate
Hz = 1/(mydata.machine_data.selection(2,1) - mydata.machine_data.selection(1,1))
span = Hz / 0.5  # smoothing parameter # good when Hz = 100
stress = smooth(mydata.machine_data.selection(:,5),span,'moving')
strain = smooth(mydata.machine_data.selection(:,4),span,'moving')
deltajump = 3 # 3 jump when finding tangent modulus
delta = round(Hz/deltajump) # 35 is good for 100 Hz, 200 is good for 100 Hz DAQ
PL = 0.005   # strain proportiona limit

# stress at which strain == PL
stressPL_point = stress(find((strain > PL),1,'first'))
# linear-elastic modulus 
EPL = stressPL_point / PL
# strain for plotting linear curve
strainPL = strain - PL
# stress for plotting linear curve
stressPL = EPL*strainPL
yieldindex = find(stressPL >= stress,1,'first')
strainPL = strainPL(1:yieldindex)+PL
stressPL = stressPL(1:yieldindex)
yieldstrain = strainPL(yieldindex)
yieldstress = stressPL(yieldindex)
E_s = stress./strain

#                     # # two point linear fit method
#                     k1=1
#                     E_t = []
#                     strain_mod = []
#                     stress_mod = []
#                     for x = 1:delta:length(stress)-delta
#                         E_t(k1) = ( stress(x+delta) - stress(x) ) / ( strain(x+delta) - strain(x) )
#                         strain_mod(k1) = mean([strain(x+delta) strain(x)])
#                         stress_mod(k1) = mean([stress(x+delta) stress(x)])
#                         k1=k1+1
#                     end

# least squares linear fit method
k1=1
for k2 = 1:delta:length(stress)-delta
    # perform the fit and plot
    fitobject = fit(strain(k2:k2+delta),stress(k2:k2+delta),'poly1')
    stress_fit = fitobject.p1.*strain(k2:k2+delta) + fitobject.p2
    E_t(k1) = fitobject.p1    
    strain_mod(k1) = mean(strain(k2:k2+delta))  # avearage over the length of fitting
    stress_mod(k1) = mean(stress(k2:k2+delta))   
    k1=k1+1
end

# smoothing method for E_t
span2 = span/8   # smoothing parameter
E_t = smooth(E_t,span2,'lowess')

# # find loopss and calculaute strain energy dissappted
subplot(2,3,1)
plot(strain,stress)hold on
strainincr = max(strain)
[~, loopindex] = findpeaks(-1*strain)
loopindex = [1 loopindex]
for k1 = 1:length(loopindex)-1
    loopstrain = strain(loopindex(k1):loopindex(k1+1))
    loopstress = stress(loopindex(k1):loopindex(k1+1))
    plot(loopstrain + strainincr, loopstress)
    strainincr = strainincr + max(loopstrain)
    maxindex = find(max(loopstress)== loopstress)
    disapenergy(k1) = trapz(loopstrain,loopstress)
    totalloopenergy(k1) = trapz(loopstrain(1:maxindex),loopstress(1:maxindex))
    recovloopenergy(k1) = totalloopenergy(k1) - disapenergy(k1)
    maxstrain(k1) = max(loopstrain)
    maxstress(k1) = max(loopstress)
end
xlabel('\epsilon(#)')ylabel('\sigma(MPa)')
title('Individual loops with energy')
axis tight

subplot(2,3,2)
# xbarval = ['loop 1','loop 2','loop 3']
ybarval = [totalloopenergy recovloopenergy disapenergy]'
# can also use area(strain,stress) to fill in plot areas with color
bar(ybarval)
xlabel('loop #')
ylabel('Strain Energy')
legend('Total','Recovered','Dissappated')

subplot(2,3,3)
# find the modulus peaks and plot
[Epeaks, Elocs] = findpeaks(E_t)
strain_mod_peaks = strain_mod(Elocs)
stress_mod_peaks = stress_mod(Elocs)
strain_beginloops = 0.1
Eind = strain_mod_peaks > strain_beginloops
E_unload = Epeaks(Eind)
stress_unload = stress_mod_peaks(Eind)                    
strain_mod(Elocs)
strain_unload = strain_mod_peaks(Eind)
try
    assert(~isempty(strain_unload))
catch
    disp('no loops, quitting')
    continue
end
[AX,H1,H2] = plotyy(strain,stress,strain_mod,E_t)
hold on
# plot(AX(1),strainPL,stressPL,'r')   
# set(get(AX(1),'Ylabel'),'String','\sigma (MPa)') 
set(get(AX(1),'Xlabel'),'String','\epsilon(#)')
set(get(AX(2),'Ylabel'),'String','E_{tan^{-1}}(MPa)') 
set(AX(2),'YLim',[0  E_t(1)] ,'YTick',0:500:max(E_t))
set(AX(1),'YLim',[0  max(stress)],'YTick',0:50:max(stress) )
title('stress-strain plot with E_{tan}')
hold off   

subplot(2,3,6)
# smoothing method for inverse E_t
Einv_t = 1./E_t
Einv_t = smooth(Einv_t,span2,'lowess')                     
[AX,H1,H2] = plotyy(strain,stress,strain_mod,Einv_t)
hold on
set(get(AX(1),'Xlabel'),'String','\epsilon(#)')
set(get(AX(2),'Ylabel'),'String','E_{tan^{-1}} , MPa') 
set(AX(2),'YLim',[0  0.002]) # ,'YTick',0:500:max(1./E_t))
set(AX(1),'YLim',[0  max(stress)])# ,'YTick',0:50:max(stress) )
title('stress-strain plot with E_{tan^{-1}} , MPa')
hold off 

subplot(2,3,4)
x = 1:length(E_t)
[AX,~,~] = plotyy(x, E_t, x, Einv_t)
set(AX(2),'YLim',[0  0.002]) 
title('Strain vs stress E_{tan} ')
xlabel('time')
set(get(AX(2),'Ylabel'),'String','E_{tan^{-1}} , MPa')                     

subplot(2,3,5)
for k1 = 1:length(E_unload)
    strainshift = (strain_unload(k1)*E_unload(k1)- stress_unload(k1))/ E_unload(k1)
    strainpeak(:,k1) = linspace(-0.1,strain_unload(k1),100)
    stresspeak(:,k1) = strainpeak(:,k1)*E_unload(k1)-strainshift*E_unload(k1)
end                    
plot(strainpeak,stresspeak)
plot(strain,stress,strainpeak,stresspeak)
title('tangent modulus on unload')
ylim([-100 max(stress)])
 
"""