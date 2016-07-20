
    plotnames = {'Stress-Strain',...
                 'Stress-Strain with data table',...
                 'Stress-Strain tangent modulus and energy calcs',...
                 'Histogram of Energy',...
                 'Stress-Frequency Energy bubble',...
                 'Scatterhist Stress-Frequency-Energy',...
                 'Scatterhist Stress-Frequency Centroid',...
                 'Line Stress-Frequency','Scatterhist Stress-Energy',...
                 'Scatterplot Stress-Frequency-Energy(bubble)',...
                 'Scatterplot Stress-ShortFrequency-Energy(bubble)',...
                 'Scatterplot Strain-Frequency-Energy(bubble)',...
                 'ScatHistBub Stress-Strain-Energy(bubble) with Location',...
                 'Stress-strain with location',...
                 'Stress-Cumulative Frequency',...
                 'Stress-Cumulative Energy',...
                 'Stress-Cumulative Events',...
                 'Strain-Cumulative Energy',...
                 'Strain-Cumulative Events',...
                 'Stress-Stratified Cumulative Energy',...
                 'Strain-Stratified Cumulative Energy',...
                 'Stress-Stratified Cumulative Events',...
                 'Strain-Stratified Cumulative Events',...
                 'Stress vs Electrical Resistivity',...
                 'Stress vs Electrical Resistance',...
                 'Strain vs Electrical Resistance',...
                 'Stress vs Electrical Conductance',...
                 'ER / Strain',...
                 'Time,Electrical Conductance, Tangent Modulus',...
                 'Spectral analysis'};
    plotnamedata  = [];
    for k = 1:length(plotnames)
        if k < length(plotnames)
            plotnamedata = strcat(plotnamedata,plotnames{k},'|');
        else
            plotnamedata = strcat(plotnamedata,plotnames{k});
        end
    end    
    h_popupmenu_plottype = uicontrol(h_fig0, ...
        'Style','popupmenu','Enable','off',...
        'Units','normalized','Position',[0.01 0.06 0.98 0.09],...
        'String',plotnamedata,...
        'Callback','mainUI(''popupmenu_plottype_callback'');','Tag','h_popupmenu_plottype');    
    % Select test folder
    case 'pushbutton_maindirectory_callback'  
        folder = uigetdir('C:\','Choose Directory');
        if folder == 0
            set(h_listbox_specimens,'String','')
            set(h_pushbutton_maindirectory,'String','choose directory');
            set(h_pushbutton_plot,'Enable','off')
            set(h_popupmenu_plottype,'Enable','off')
            set(h_listbox_specimens,'Enable','off')
            set(h_checkbox_multiwindow,'Enable','off')            
            return
        end
        set(h_pushbutton_maindirectory,'String',folder);
        d = dir(folder);
        isub = [d(:).isdir]; %# returns logical vector
        dirNames = {d(isub).name}';
        dirNames(ismember(dirNames,{'.','..','old'})) = []; % removes folder 'old'
        listboxdata  = [];
        for k = 1:length(dirNames)
            if k < length(dirNames)
                listboxdata = strcat(listboxdata,dirNames{k},'|');
            else
                listboxdata = strcat(listboxdata,dirNames{k});
            end
        end
        specimens = dirNames;
        set(h_listbox_specimens,'String',listboxdata)
        set(h_pushbutton_plot,'Enable','on')
        set(h_popupmenu_plottype,'Enable','on')
        set(h_listbox_specimens,'Enable','on')
        set(h_checkbox_multiwindow,'Enable','on')
        
    case 'h_checkbox_multiwindow'
        % no action, PLOT button is action
    case 'listbox_specimens_callback'
        % no action, PLOT button is action        
    case 'popupmenu_plottype_callback'
        % no action, plot button is action
    case 'pushbutton_close_callback'     
        delete(h_fig0);close all       
    case 'pushbutton_plot_callback'
%         set(gcf,'Menubar','none')
        newfontsize = 12;
        linecol = {'k','r','b','g','c','y'};
        linetype = {'-','-.',':','--'};
        plotnames = cellstr(get(h_popupmenu_plottype,'String'));
        plottype = plotnames{get(h_popupmenu_plottype,'Value')};
        specimens = cellstr(get(h_listbox_specimens,'String'));
        specname = cellstr( specimens(get(h_listbox_specimens,'Value')) );
        folder = get(h_pushbutton_maindirectory,'String');
        multiwindow = get(h_checkbox_multiwindow,'Value');
        for k = 1:length(specname)
            if k > 1 && multiwindow  % initialize figure and axes
                curfig = figure;
                curaxes = gca;
                linecol = {'k','k','k','k','k','k','k','k'};
            elseif k == 1
                curfig = figure; %(2);
                curaxes = gca;
                hold on
            else
                hold on
            end
            
            try
                waitbar1 = waitbar(0,'please wait while file is loaded');
                load(strcat(folder,'\',specname{k},'\mydata.mat'));
                close(waitbar1)            
            catch
                sprintf('\n %s skipped \n',specname{k})
                close(waitbar1) 
                close(curfig)
                continue
            end

            try 
                arch = mydata.test_properties.arch;
                ann = mydata.test_properties.annealed;
            catch
                disp('no arch or ann data available')
                arch = '-';
                ann = '-';
            end
            % Plotting Cases
            
% % % -------------------- add more plots here -------------------- % % %              
            switch plottype
   

                case 'Stress-Strain'
                    %%

                    stress = mydata.machine_data.selection(:,5);
                    strain = mydata.machine_data.selection(:,4);
                    plot(strain,stress,linecol{k});
                    if multiwindow
                        plotlabel{1} = strcat(specname{k},':',arch,':',ann);
                    else
                        plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    end
                    title('Stress-strain plot')
                    xlabel('\epsilon , %')
                    ylabel('\sigma , Mpa')
                    legend(plotlabel,'Location','SouthEast','Fontsize',12)    

                case 'Stress-Strain with data table'
                    %%
                    stress = mydata.machine_data.selection(:,5);
                    strain = mydata.machine_data.selection(:,4);
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    strainer = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));  
                    stresser = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/mydata.test_properties.area;
                    stress_range = [5 15];
                    PL = 0.005;
                    E_sec = stress./strain;
                    % Initialize table
                    if k == 1
                        cnames = {'E,GPa','PL_Strain,%','PL Stress,MPa','UT_Strain,%','UT_Stress,MPa','1st AE,MPa','1st Loud AE,MPa','5th Loud AE,MPa','# Gage Events'};
                        tabledat = { 0         0,            0,                0,              0  ,       0                0                  0              0};
                        uitable_results = uitable(curfig,'ColumnName', cnames , 'RowName', specname,...
                         'Units', 'normalized','Position',[0.02 0.02 .95 .15],...
                         'Data', tabledat);
                        set(curaxes,'Position',[0.11 0.28 .85 .65])
                    end
                    for y = 1:length(strain)
                        if stress(y) >= stress_range(2)
                            break
                        end
                    end
                    PLindex = find( (stress(1:y) < stress_range(2) & stress(1:y) > stress_range(1)) );

                    stressPL = stress(PLindex);
                    strainPL = strain(PLindex);

                    % perform the fit and plot
                    linfit = fit(strainPL+PL,stressPL,'poly1');
                    % stressPLfit = linfit.p1.*(strainPL-PL) + linfit.p2;
                    allstressPLfit = linfit.p1.*(strain) + linfit.p2;
                    E_0 = linfit.p1/10;

                    PLPointindex = find( allstressPLfit >= stress ,1,'first');

                    if isempty(PLPointindex)
                        stressPLplot = allstressPLfit(allstressPLfit>=0);
                        strainPLplot = strain(allstressPLfit>=0);
                    else
                        stressPLplot = allstressPLfit(allstressPLfit(1:PLPointindex)>=0);
                        strainPLplot = strain(allstressPLfit(1:PLPointindex)>=0);
                    end

                    % if the PL limit was not reached
                    if isempty(PLPointindex)
                        PLPointindex = length(stress);
                    end  

                    plot(curaxes,strain,stress,linecol{k})
                    hold on
                    if k == 1
                        maxstrain = 0;
                        maxstress = 0;
                    end
                    if max(strain) > maxstrain
                        maxstrain = max(strain);
                    end
                    if max(stress) > maxstress
                        maxstress = max(stress);
                    end
                    set(curaxes,'XGrid','on','YGrid','on')
                    %     set(handles.axes1,'XTick',[0:0.1:max(strain)],'YTick',[0:100:max(stress)])
                    set(curaxes,'YLim',[0 maxstress*1.1],'XLim',[0 maxstrain*1.1])
                    title('Stress-Strain Curve')
                    xlabel('\epsilon (%)')
                    ylabel('\sigma (MPa)')
                    plot(strainPLplot,stressPLplot,'Linestyle', ':')
                    scatter(curaxes, [min(strainPL) max(strainPL)] ,[min(stressPL) max(stressPL)]);
                    scatter(curaxes, strain(PLPointindex), stress(PLPointindex))
                    hold off
                    UT_stress = max(stress);
                    UT_strain = max(strain); 
                    PL_stress = stress(PLPointindex);
                    PL_strain = strain(PLPointindex);
                    enloudind = log10(en)>=1;
                    enloud = en(enloudind);
                    stressloud = stresser(enloudind);
                    tabledat(k,:) = { sprintf('%3.0f',E_0)    sprintf('%0.3f',PL_strain)     sprintf('%3.0f',PL_stress)       sprintf('%0.3f',UT_strain)     sprintf('%3.0f',UT_stress)         sprintf('%3.0f',stresser(1))      sprintf('%3.0f',stressloud(1))        sprintf('%3.0f',stressloud(5))  numel(en)  };
                    % complete table with data for multi-selection
                    set(uitable_results , 'Data', tabledat)
                    % plot(handles.axes1,strainPL,stressPL,'r')          
                    newfontsize = 12;       
                    set(gcf,'Position',[150   150   1000   475],'Color',[1 1 1])
                    
                case 'Stress-Strain tangent modulus and energy calcs'
                    %% 
                    % calculations may vary depending on the smoothness of
                    %
                    % parameters to adjust to make calculations better
                    % span = smoothing parameter
                    % span2 = smoothing parameter
                    % deltajump = jump when finding tangent modulus
                    
                    set(gcf,'position',[35 1133 1254 702])
                    if length(specname) > 1
                        errordlg('Cannot perform multispecimen analysis')
                        close(curfig)
                        return
                    end
                    % Hz=sample rate
                    Hz = 1/(mydata.machine_data.selection(2,1) - mydata.machine_data.selection(1,1))
                    span = Hz / 0.5  % smoothing parameter % good when Hz = 100
                    stress = smooth(mydata.machine_data.selection(:,5),span,'moving');
                    strain = smooth(mydata.machine_data.selection(:,4),span,'moving');
                    deltajump = 3 % 3 jump when finding tangent modulus
                    delta = round(Hz/deltajump) % 35 is good for 100 Hz, 200 is good for 100 Hz DAQ
                    PL = 0.005;   % strain proportiona limit

                    % stress at which strain == PL
                    stressPL_point = stress(find((strain > PL),1,'first'));
                    % linear-elastic modulus 
                    EPL = stressPL_point / PL;
                    % strain for plotting linear curve
                    strainPL = strain - PL;
                    % stress for plotting linear curve
                    stressPL = EPL*strainPL;
                    yieldindex = find(stressPL >= stress,1,'first');
                    strainPL = strainPL(1:yieldindex)+PL;
                    stressPL = stressPL(1:yieldindex);
                    yieldstrain = strainPL(yieldindex);
                    yieldstress = stressPL(yieldindex);
                    E_s = stress./strain;

%                     % % two point linear fit method
%                     k1=1;
%                     E_t = [];
%                     strain_mod = [];
%                     stress_mod = [];
%                     for x = 1:delta:length(stress)-delta
%                         E_t(k1) = ( stress(x+delta) - stress(x) ) / ( strain(x+delta) - strain(x) );
%                         strain_mod(k1) = mean([strain(x+delta) strain(x)]);
%                         stress_mod(k1) = mean([stress(x+delta) stress(x)]);
%                         k1=k1+1;
%                     end

                    % least squares linear fit method
                    k1=1;
                    for k2 = 1:delta:length(stress)-delta
                        % perform the fit and plot
                        fitobject = fit(strain(k2:k2+delta),stress(k2:k2+delta),'poly1');
                        stress_fit = fitobject.p1.*strain(k2:k2+delta) + fitobject.p2;
                        E_t(k1) = fitobject.p1;    
                        strain_mod(k1) = mean(strain(k2:k2+delta));  % avearage over the length of fitting
                        stress_mod(k1) = mean(stress(k2:k2+delta));   
                        k1=k1+1;
                    end

                    % smoothing method for E_t
                    span2 = span/8;   % smoothing parameter
                    E_t = smooth(E_t,span2,'lowess');

                    % % find loopss and calculaute strain energy dissappted
                    subplot(2,3,1)
                    plot(strain,stress);hold on
                    strainincr = max(strain);
                    [~, loopindex] = findpeaks(-1*strain);
                    loopindex = [1; loopindex];
                    for k1 = 1:length(loopindex)-1
                        loopstrain = strain(loopindex(k1):loopindex(k1+1));
                        loopstress = stress(loopindex(k1):loopindex(k1+1));
                        plot(loopstrain + strainincr, loopstress)
                        strainincr = strainincr + max(loopstrain);
                        maxindex = find(max(loopstress)== loopstress);
                        disapenergy(k1) = trapz(loopstrain,loopstress);
                        totalloopenergy(k1) = trapz(loopstrain(1:maxindex),loopstress(1:maxindex));
                        recovloopenergy(k1) = totalloopenergy(k1) - disapenergy(k1);
                        maxstrain(k1) = max(loopstrain);
                        maxstress(k1) = max(loopstress);
                    end
                    xlabel('\epsilon(%)');ylabel('\sigma(MPa)')
                    title('Individual loops with energy')
                    axis tight
                    
                    subplot(2,3,2)
                    % xbarval = ['loop 1','loop 2','loop 3'];
                    ybarval = [totalloopenergy; recovloopenergy; disapenergy]';
                    % can also use area(strain,stress) to fill in plot areas with color
                    bar(ybarval)
                    xlabel('loop #')
                    ylabel('Strain Energy')
                    legend('Total','Recovered','Dissappated')

                    subplot(2,3,3)
                    % find the modulus peaks and plot
                    [Epeaks, Elocs] = findpeaks(E_t);
                    strain_mod_peaks = strain_mod(Elocs);
                    stress_mod_peaks = stress_mod(Elocs);
                    strain_beginloops = 0.1;
                    Eind = strain_mod_peaks > strain_beginloops;
                    E_unload = Epeaks(Eind);
                    stress_unload = stress_mod_peaks(Eind);                    
                    strain_mod(Elocs);
                    strain_unload = strain_mod_peaks(Eind);
                    try
                        assert(~isempty(strain_unload))
                    catch
                        disp('no loops, quitting')
                        continue
                    end
                    [AX,H1,H2] = plotyy(strain,stress,strain_mod,E_t);
                    hold on
                    % plot(AX(1),strainPL,stressPL,'r')   
                    % set(get(AX(1),'Ylabel'),'String','\sigma (MPa)') 
                    set(get(AX(1),'Xlabel'),'String','\epsilon(%)')
                    set(get(AX(2),'Ylabel'),'String','E_{tan^{-1}}(MPa)') 
                    set(AX(2),'YLim',[0  E_t(1)] ,'YTick',0:500:max(E_t))
                    set(AX(1),'YLim',[0  max(stress)],'YTick',0:50:max(stress) )
                    title('stress-strain plot with E_{tan}')
                    hold off   
                    
                    subplot(2,3,6)
                    % smoothing method for inverse E_t
                    Einv_t = 1./E_t;
                    Einv_t = smooth(Einv_t,span2,'lowess');                     
                    [AX,H1,H2] = plotyy(strain,stress,strain_mod,Einv_t);
                    hold on
                    set(get(AX(1),'Xlabel'),'String','\epsilon(%)')
                    set(get(AX(2),'Ylabel'),'String','E_{tan^{-1}} , MPa') 
                    set(AX(2),'YLim',[0  0.002]) % ,'YTick',0:500:max(1./E_t))
                    set(AX(1),'YLim',[0  max(stress)])% ,'YTick',0:50:max(stress) )
                    title('stress-strain plot with E_{tan^{-1}} , MPa')
                    hold off 
                    
                    subplot(2,3,4)
                    x = 1:length(E_t);
                    [AX,~,~] = plotyy(x, E_t, x, Einv_t);
                    set(AX(2),'YLim',[0  0.002]) 
                    title('Strain vs stress E_{tan} ')
                    xlabel('time')
                    set(get(AX(2),'Ylabel'),'String','E_{tan^{-1}} , MPa')                     
                    
                    subplot(2,3,5)
                    for k1 = 1:length(E_unload)
                        strainshift = (strain_unload(k1)*E_unload(k1)- stress_unload(k1))/ E_unload(k1);
                        strainpeak(:,k1) = linspace(-0.1,strain_unload(k1),100);
                        stresspeak(:,k1) = strainpeak(:,k1)*E_unload(k1)-strainshift*E_unload(k1);
                    end                    
                    plot(strainpeak,stresspeak)
                    plot(strain,stress,strainpeak,stresspeak)
                    title('tangent modulus on unload')
                    ylim([-100 max(stress)])
                     
                case 'Histogram of Energy'
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    stresser = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/mydata.test_properties.area;
                    
                    enloudind = log10(en)>=1;
                    enloud = en(enloudind);
                    stressloud = stresser(enloudind);
                    meanEn = mean(en);
                    medEn = median(en);
                    max(en);
                    
                    % plotting normal energy
                    subplot(2,2,1)
                    hist(en)
                    axis tight
                    xlimvhist = get(gca,'Xlim');
                    ylabel('counts')
                    xlabel('Energy , V^2\mus')
                    title('Histogram of Energy')
                    
                    subplot(2,2,3)
                    boxplot(en)
                    set(gca,'Ylim',xlimvhist)
                    ylabel('')
                    hold on
                    plot(1,meanEn,'+')
                    plot(1,enloud(1),'*')
                    hold off                   
                    camroll(-90)
                    set(gca,'YtickLabel','')
                    set(gca,'XtickLabel','')                     
                    legend('Mean','1st Loud','Location','SouthWest')
                    title(strcat(specname{k},':',arch,':',ann))
                    
                    % plotting log energy
                    subplot(2,2,2)
                    hist(log10(en))
                    axis tight
                    xlimvhist = get(gca,'Xlim');
                    ylabel('counts')
                    xlabel('log Energy , V^2\mus')
                    title('Histogram of log_{10} Energy')

                    subplot(2,2,4)
                    boxplot(log10(en))
                    set(gca,'Ylim',xlimvhist)                   
                    hold on
                    plot(1,log10(meanEn),'+')   
                    plot(1,log10(enloud(1)),'*')
                    hold off                    
                    camroll(-90)  
                    set(gca,'YtickLabel','')
                    set(gca,'XtickLabel','')                     
                    legend('Mean','1st Loud','Location','SouthWest')
                    xlabel(sprintf('ratio max En to median En = %0.1f %%',100*medEn/max(en)))
                    title(strcat(specname{k},':',arch,':',ann))
                    
                case 'Stress-Frequency Energy bubble'
                    %%

                    area = mydata.test_properties.area;
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area;
                    if strcat(ann) == strcat('an')
                        scatter(en,fc,'r');
                    else
                        scatter(en,fc,'b');
                    end                     
                    ax = findobj(curfig,'type','axes');
                    %             ax = get(gcf,'Children');
                    set(ax,'FontSize',13)
                    title(strcat(specname{k},':',arch,':',ann,':Energy-Frequency plot'))
                    xlabel('Energy , V^2\mus')
                    ylabel('Frequency Centroid , kHz')
                    %     xlim([0 300])
                    ylim([0 1500])
                    axis tight

                case 'Scatterhist Stress-Frequency-Energy'
                    %%
                    area = mydata.test_properties.area;
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area;
                    scatterhist(stress,fc);
                    title(strcat(specname{k},':',arch,':',ann,':Energy-Frequency plot'))
                    xlabel('Energy , V^2\mus')
                    ylabel('Frequency Centroid , kHz')
                    %     xlim([0 300])
                    ylim([0 1500])                    
                    
                case 'Line Stress-Frequency'
                    %%
                    area = mydata.test_properties.area;
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area;
                    plot(stress,fc,linecol{k})
                    ax = findobj(curfig,'type','axes');
                    %             ax = get(gcf,'Children');
                    set(ax,'FontSize',13)
                    title(strcat(specname{k},':',arch,':',ann,':Stress-Frequency Centroid plot'))
                    xlabel('Stress')
                    ylabel('Frequency Centroid , kHz')
                    if multiwindow
                        plotlabel{1} = strcat(specname{k},':',arch,':',ann);
                    else
                        plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    end                     
                    legend(plotlabel,'Location','SouthEast','Fontsize',12)  
                    %     xlim([0 300])
                    ylim([0 1500])              
                    
                case 'Scatterhist Stress-Energy'
                    %%
                    set(h_checkbox_multiwindow,'Value',1)
                    area = mydata.test_properties.area;
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area;
                    if strcat(ann) == strcat('an')
                        hscat = scatterhist(stress,en,'nbins',20,'Color','r');
                    else
                        hscat = scatterhist(stress,en,'nbins',20,'Color','b');
                    end
                    ax = findobj(hscat,'type','axes');
                    set(ax,'FontSize',13)
                    title(strcat(specname{k},':',arch,':',ann))
                    xlabel('\sigma , Mpa')
                    ylabel('Energy , V^2\mus')
%                     xlim([100 300])
                %     ylim([0 50])s
                    set(gcf,'Position',[400 400 350 300])
                    
                case 'Scatterhist Stress-Frequency Centroid'
                    %%
                    set(h_checkbox_multiwindow,'Value',1)
                    area = mydata.test_properties.area;
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area;
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    if strcat(ann) == strcat('an')
                        hscat = scatterhist(stress,fc,'nbins',20,'Color','r');
                    else
                        hscat = scatterhist(stress,fc,'nbins',20,'Color','b');
                    end
                    ax = findobj(hscat,'type','axes');
                    set(ax,'FontSize',13)
                    title(strcat(specname{k},':',arch,':',ann))
                    xlabel('\sigma , Mpa')
                    ylabel('Frequency Centroid , kHz')
%                     xlim([100 300])
                %     ylim([0 50])s
                    set(gcf,'Position',[400 400 350 300])  
                    ylim([220 1200])  

                case 'Scatterplot Stress-Frequency-Energy(bubble)'
                    %%
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    area = mydata.test_properties.area;
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area;
                    if strcat(ann) == strcat('an')
                        scatter(stress,fc,en*2,'r');
                    else
                        scatter(stress,fc,en*2,'b');
                    end                    
                    ax = findobj(gcf,'type','axes');
                    ylim([100 1500])
%                     xlim([100 300])
                    xlim([min(stress)*0.95 max(stress)*1.05])
%                     ylim([min(fc)*0.95 max(fc)*1.05])
                    set(ax,'FontSize',13)
                    title(strcat(specname{k},':',arch,':',ann,':Stress-Frequency plot'))
                    xlabel('\sigma , Mpa')
                    ylabel('Frequency Centroid , kHz')
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    if ~multiwindow
                        legend(plotlabel,'Location','NorthEast')
                    end
                    
                case 'Scatterplot Stress-ShortFrequency-Energy(bubble)'   
                    %%
                    area = mydata.test_properties.area;
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area;        
                    events = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Index'));   
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    strain = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));                    
                    Fs = 10e3;       % Sample Frequency (kHz)
                    Fsmin = 50;    %  kHz
                    Fsmax = 2e3;     %  kHz
                    
                    s3 = mydata.AE_data.sensor3; % sensor 3 events
                    
                    FCshort = zeros(length(events),1);
                    for ev = 1:length(FCshort)
                        x = s3(:,events(ev)); % time domain sample and event num
                        overlap = 30;
                        hamwindow = 200;
                        S = abs(spectrogram(x,hamwindow,overlap,[],[]));  % window  = 275, is a Hamming window of length nfft.
                        Slen = length(S);
                        Sfreq = (0:Slen-1)*(Fs/Slen)';% xaxis of fft plot
                        Sfreq = Sfreq(1:Slen)/2;
                        Sfreq = Sfreq(Sfreq<=Fsmax); % filter out frequency above Fsmax 
                        Sfreq = Sfreq(Sfreq>=Fsmin); % filter out frequency below Fsmin 
                        S =         S(Sfreq<=Fsmax,:); % filter out frequency above Fsmax 
                        S =         S(Sfreq>=Fsmin,:); % filter out frequency below Fsmin 
                        shortsegmentnum = 2;
                        Smean = mean(S,shortsegmentnum);
                        Sshort = S(:,shortsegmentnum)';
                        FCshort(ev) = sum(Sshort.*Sfreq)/sum(Sshort);
                    end
                    if strcat(ann) == strcat('an')
                        scatter(stress,FCshort,en*2,'r');
                    else
                        scatter(stress,FCshort,en*2,'b');
                    end 
                    ax = findobj(gcf,'type','axes');
                    ylim([Fsmin Fsmax])
%                     xlim([100 300])
%                     xlim([min(stress)*0.95 max(stress)*1.05])
%                     ylim([min(fc)*0.95 max(fc)*1.05])
                    set(ax,'FontSize',13)
                    title(strcat(specname{k},':',arch,':',ann,':Stress-ShortFrequency plot'))
                    xlabel('\sigma , Mpa')
                    ylabel('Short Frequency Centroid , kHz')
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    if ~multiwindow
                        legend(plotlabel,'Location','SouthEast')
                    end
                    
                case 'Scatterplot Strain-Frequency-Energy(bubble)'
                    %%
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    strain = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));               
                    if strcat(ann) == strcat('an')
                        scatter(strain,fc,en*2,'r');
                    else
                        scatter(strain,fc,en*2,'b');
                    end 
                    ax = findobj(gcf,'type','axes');
                    ylim([100 1500])
%                     xlim([100 300])
                    xlim([min(strain)*0.75 max(strain)])
%                     ylim([min(fc)*0.95 max(fc)*1.05])
                    set(ax,'FontSize',13)
                    title(strcat(specname{k},':',arch,':',ann,':Strain-Frequency plot'))
                    xlabel('\epsilon , %')
                    ylabel('Frequency Centroid , kHz')
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    if ~multiwindow
                        legend(plotlabel,'Location','NorthEast')                    
                    end
                    
                case 'ScatHistBub Stress-Strain-Energy(bubble) with Location'
                    %%
                    %     stress = mydata.machine_data.selection(:,5);
                    en = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    area = mydata.test_properties.area;
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area; 
                    strain = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    En3 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));  
                    loc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Loc'));
                    fail = mydata.test_properties.failure_loc;
                    s1loc = mydata.AE_parameters.sensors_location(1,1);
                    s2loc = mydata.AE_parameters.sensors_location(2,1);
                    ScatHistBub(stress,loc,En3,fail,[s1loc s2loc])
                    ax = findobj(gcf,'type','axes');
                    title(ax(4),strcat(specname{k},':',arch,':',ann))
                    
                case 'Stress-strain with location'
                    %%
                    area = mydata.test_properties.area;
                    stress = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'))/area; 
                    strain = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    loc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Loc'));
                    
                    hl1 = line(strain,stress,'Color','k');
                    ax1 = gca;
                    set(ax1,'XColor','k','YColor','k')
                    ax2 = axes('Position',get(ax1,'Position'),...
                               'XAxisLocation','top',...
                               'YAxisLocation','right',...
                               'Color','none',...
                               'XColor','b','YColor','b');
                    hl2 = line(strain,loc,'Color','b','Parent',ax2,'Marker','o','Linestyle','none');       
                    xlabel(ax1,'\epsilon , %')
                    ylabel(ax1,'\sigma, MPa')
                    ylabel(ax2,'location , mm')
                    set(ax2,'YTicklabel',[],'XTicklabel',[])
                
                case 'Stress-Cumulative Frequency'             
                    %%
                    param1 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'));
                    stress = param1 / mydata.test_properties.area;  
                    fc = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'FC 3'));
                    CumFreq = zeros(length(fc),1);
                    for k1=1:length(fc)-1; CumFreq(k1+1) = CumFreq(k1) + fc(k1);end
                    CumFreq = CumFreq/CumFreq(end);
                    plot(stress,CumFreq,linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Stress-Cumulative Frequency plot')
                    ylabel('Normalized Cumulative Frequency')
                    xlabel('\sigma , Mpa')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)                     
                    
                case 'Stress-Cumulative Energy'
                    %%
                    param1 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'));
                    stress = param1 / mydata.test_properties.area;  
                    En = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    CumEn = zeros(length(En),1);
                    for k1=1:length(En)-1; CumEn(k1+1) = CumEn(k1) + En(k1);end
                    plot(stress,CumEn,linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Stress-Cumulative Energy plot')
                    ylabel('Cumulative Energy , V^2\mus')
                    xlabel('\sigma , Mpa')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)       
                
                
                case 'Stress-Cumulative Events'
                    %%
                    param1 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'));
                    stress = param1 / mydata.test_properties.area;  
                    En3 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    CumEvents = 1:length(En3);
                    plot(stress,CumEvents,linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Stress-Cumulative Events plot')
                    ylabel('Cumulative Events')
                    xlabel('\sigma , Mpa')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)       
                                    
                    
                case 'Strain-Cumulative Energy'
                    %%
                    param2 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    strain = param2; 
                    En = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    CumEn = zeros(length(En),1);
                    for k1=1:length(En)-1; CumEn(k1+1) = CumEn(k1) + En(k1);end
                    plot(strain,CumEn,linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Strain-Cumulative Energy plot')
                    ylabel('Cumulative Energy , V^2\mus')
                    xlabel('\epsilon , %')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)  
                    
                
                case 'Strain-Cumulative Events'
                    %%
                    param2 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    strain = param2; 
                    En = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    CumEvents = 1:length(En);
                    plot(strain,CumEvents,linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Strain-Cumulative Events plot')
                    ylabel('Cumulative Events')
                    xlabel('\epsilon , %')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)  
                    
                                        
                case 'Stress-Stratified Cumulative Energy'
                    %%
                    param1 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'));
                    param2 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    strain = param2;
                    stress = param1 / mydata.test_properties.area;  
                    En = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    Ensize = median(En);
                    stress_CumEnLarge = stress( find(En>Ensize,1) );
                    stress_CumEnSmall = stress( find(En<=Ensize,1) );
                    CumEn = En(1);
                    CumEnLarge = En( find(En>Ensize,1) );
                    CumEnSmall = En( find(En<=Ensize,1) );
                    for k1=2:length(En); 
                        CumEn = [ CumEn (CumEn(end)+En(k1)) ];
                        if En(k1) > Ensize
                            CumEnLarge = [CumEnLarge (CumEnLarge(end)+En(k1))];
                            stress_CumEnLarge = [stress_CumEnLarge stress(k1)];
                        else
                            CumEnSmall = [CumEnSmall (CumEnSmall(end)+En(k1))];
                            stress_CumEnSmall = [stress_CumEnSmall stress(k1)];
                        end
                    end
                    plot(stress,CumEn,linecol{k})
                    hold on
                    plot(stress_CumEnSmall, CumEnSmall,'b')
                    plot(stress_CumEnLarge, CumEnLarge,'r')
                    hold off
                    title(strcat(specname{k},':',arch,':',ann,':Stratified by median Cumulative Energy'))
                    ylabel('Cumulative Energy , V^2\mus')
                    xlabel('\sigma , Mpa')
                    legend({'Total Cumulative Energy',sprintf('Cumulative Energy<=%0.1f',Ensize),sprintf('Cumulative Energy>%0.1f',Ensize)},'Location','NorthWest','Fontsize',12)
                    
                    
                case 'Strain-Stratified Cumulative Energy'
                    %%
                    param1 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'));
                    param2 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    strain = param2;
                    stress = param1 / mydata.test_properties.area;  
                    En = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    Ensize = median(En);
                    strain_CumEnLarge = strain( find(En>Ensize,1) );
                    strain_CumEnSmall = strain( find(En<=Ensize,1) );
                    CumEn = En(1);
                    CumEnLarge = En( find(En>Ensize,1) );
                    CumEnSmall = En( find(En<=Ensize,1) );
                    for k1=2:length(En); 
                        CumEn = [ CumEn (CumEn(end)+En(k1)) ];
                        if En(k1) > Ensize
                            CumEnLarge = [CumEnLarge (CumEnLarge(end)+En(k1))];
                            strain_CumEnLarge = [strain_CumEnLarge strain(k1)];
                        else
                            CumEnSmall = [CumEnSmall (CumEnSmall(end)+En(k1))];
                            strain_CumEnSmall = [strain_CumEnSmall strain(k1)];
                        end
                    end
                    plot(strain,CumEn,linecol{k})
                    hold on
                    plot(strain_CumEnSmall, CumEnSmall,'b')
                    plot(strain_CumEnLarge, CumEnLarge,'r')
                    hold off
                    title(strcat(specname{k},':',arch,':',ann,':Stratified by median Cumulative Energy'))
                    ylabel('Cumulative Energy , V^2\mus')
                    xlabel('\epsilon , %')
                    legend({'Total Cumulative Energy',sprintf('Cumulative Energy<=%0.1f',Ensize),sprintf('Cumulative Energy>%0.1f',Ensize)},'Location','NorthWest','Fontsize',12)
                    
                case 'Stress-Stratified Cumulative Events'
                    %%
                    param1 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'));
                    param2 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    strain = param2;
                    stress = param1 / mydata.test_properties.area;  
                    En = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    Ensize = median(En);
                    stress_CumEventLarge = stress( En>Ensize );
                    stress_CumEventSmall = stress( En<=Ensize );
                    CumEvent = 1:length(En);
                    CumEventLarge = 1:length(find(En>Ensize)) ;
                    CumEventSmall = 1:length(find(En<=Ensize));
                    plot(stress,CumEvent,linecol{k})
                    hold on
                    plot(stress_CumEventSmall, CumEventSmall,'b')
                    plot(stress_CumEventLarge, CumEventLarge,'r')
                    hold off
                    title(strcat(specname{k},':',arch,':',ann,':Stratified by median Cumulative Events'))
                    ylabel('Cumulative Events')
                    xlabel('\sigma , Mpa')
                    legend({'Total Cumulative Events',sprintf('Cumulative events with Energy<=%0.1f',Ensize),sprintf('Cumulative events with Energy>%0.1f',Ensize)},'Location','NorthWest','Fontsize',12)

                case 'Strain-Stratified Cumulative Events'
                    %%
                    param1 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 1'));
                    param2 = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Param 2'));
                    strain = param2;
                    stress = param1 / mydata.test_properties.area;  
                    En = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Energy 3'));
                    Ensize = median(En);
                    strain_CumEventLarge = strain( En>Ensize );
                    strain_CumEventSmall = strain( En<=Ensize );
                    CumEvent = 1:length(En);
                    CumEventLarge = 1:length(find(En>Ensize)) ;
                    CumEventSmall = 1:length(find(En<=Ensize));
                    plot(strain,CumEvent,linecol{k})
                    hold on
                    plot(strain_CumEventSmall, CumEventSmall,'b')
                    plot(strain_CumEventLarge, CumEventLarge,'r')
                    hold off
                    title(strcat(specname{k},':',arch,':',ann,':Stratified by median Cumulative Events'))
                    ylabel('Cumulative Events')
                    xlabel('\epsilon , %')
                    legend({'Total Cumulative Events',sprintf('Cumulative events with Energy<=%0.1f',Ensize),sprintf('Cumulative events with Energy>%0.1f',Ensize)},'Location','NorthWest','Fontsize',12)
                                        
                case 'Stress vs Electrical Resistivity'
                    %%
                    wid = mydata.test_properties.width;  % in mm
                    thk = mydata.test_properties.thickness; % in mm
                    leng = mydata.ER_parameters.probes_location(2) - mydata.ER_parameters.probes_location(3);
                    time_MTS = mydata.machine_data.selection(:,1);
                    stress = mydata.machine_data.selection(:,5);
                    % strain = mydata.machine_data.selection(:,4);
                    time_stress = [time_MTS stress];
                    % time_strain = [time strain];
                    time_res = mydata.ER_data.selection(:,[1 2]);
                    time_res(:,2) = time_res(:,2)*wid*thk/leng;
                    ER_sync = mydata.plot_parameters.ER_sync;
                    mecha_sync = mydata.plot_parameters.mecha_sync;
                    time_res(:,1) = (time_res(:,1) - (mecha_sync - ER_sync));
                    time_res_stress = vlookup(time_res,time_stress,2);
                    % convert to change in resistance
                    hold on ; 
                    plot(time_res_stress(:,3),time_res_stress(:,2),linecol{k})
                    % plot(time_res_strain(:,3),time_res_strain(:,2),linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Stress-Resistivity plot')
                    ylabel('\rho , \Omega * mm')
                    xlabel('\sigma , Mpa')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)                     
                    
                    
                case 'Strain vs Electrical Resistance'
                    %%
                    time_MTS = mydata.machine_data.selection(:,1);
                    strain = mydata.machine_data.selection(:,4);
                    time_strain = [time_MTS strain];
                    time_res = mydata.ER_data.selection(:,[1 2]);
                    ER_sync = mydata.plot_parameters.ER_sync;
                    mecha_sync = mydata.plot_parameters.mecha_sync;
                    time_res(:,1) = (time_res(:,1) - (mecha_sync - ER_sync));
                    time_res_strain = vlookup(time_res,time_strain,2);
                    % convert to change in resistance
                    time_res_strain(:,2) = ((time_res_strain(:,2) - time_res_strain(1,2)) ./ time_res_strain(1,2))*100;
                    % time_res_strain = vlookup(time_res,time_strain,2);
                    hold on ; 
                    plot(time_res_strain(:,3),time_res_strain(:,2),linecol{k})
                    % plot(time_res_strain(:,3),time_res_strain(:,2),linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Strain-Resistance plot')
                    ylabel('Resistance change, %')
                    xlabel('\epsilon , %')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)  
                    
                case 'Stress vs Electrical Resistance'
                    %%
                    time_MTS = mydata.machine_data.selection(:,1);
                    stress = mydata.machine_data.selection(:,5);
                    % strain = mydata.machine_data.selection(:,4);
                    time_stress = [time_MTS stress];
                    % time_strain = [time strain];
                    time_res = mydata.ER_data.selection(:,[1 2]);
                    ER_sync = mydata.plot_parameters.ER_sync;
                    mecha_sync = mydata.plot_parameters.mecha_sync;
                    time_res(:,1) = (time_res(:,1) - (mecha_sync - ER_sync));
                    time_res_stress = vlookup(time_res,time_stress,2);
                    % convert to change in resistance
                    time_res_stress(:,2) = ((time_res_stress(:,2) - time_res_stress(1,2)) ./ time_res_stress(1,2))*100;
                    % time_res_strain = vlookup(time_res,time_strain,2);
                    hold on ; 
                    plot(time_res_stress(:,3),time_res_stress(:,2),linecol{k})
                    % plot(time_res_strain(:,3),time_res_strain(:,2),linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Stress-Resistance plot')
                    ylabel('Resistance change, %')
                    xlabel('\sigma , Mpa')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12) 
                    
                case 'Stress vs Electrical Conductance'
                    %%
                    time_MTS = mydata.machine_data.selection(:,1);
                    stress = mydata.machine_data.selection(:,5);
                    % strain = mydata.machine_data.selection(:,4);
                    time_stress = [time_MTS stress];
                    % time_strain = [time strain];
                    time_res = mydata.ER_data.selection(:,[1 2]);
                    ER_sync = mydata.plot_parameters.ER_sync;
                    mecha_sync = mydata.plot_parameters.mecha_sync;
                    time_res(:,1) = (time_res(:,1) - (mecha_sync - ER_sync));
                    time_res_stress = vlookup(time_res,time_stress,2);
                    % time_res_strain = vlookup(time_res,time_strain,2);
                    time_con_stress = time_res_stress;
                    time_con_stress(:,2) = 1./time_res_stress(:,2);
                    time_con_stress(:,2) = ((time_con_stress(:,2) - time_con_stress(1,2)) ./ time_con_stress(1,2))*100;
                    hold on ; 
                    time_con_stress(:,2) = smooth(time_con_stress(:,2));
                    plot(time_con_stress(:,3),time_con_stress(:,2),linecol{k})
                    % plot(time_res_strain(:,3),time_res_strain(:,2),linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('Stress-Conductance plot')
                    ylabel('Conductance change, %')
                    xlabel('\sigma , Mpa')
                    legend(plotlabel,'Location','NorthWest','Fontsize',12)     
                    
                case 'ER / Strain'
                    %%
                    time_MTS = mydata.machine_data.selection(:,1);
                    strain = mydata.machine_data.selection(:,4);
                    time_strain = [time_MTS strain];
                    time_res = mydata.ER_data.selection(:,[1 2]);
                    ER_sync = mydata.plot_parameters.ER_sync;
                    mecha_sync = mydata.plot_parameters.mecha_sync;
                    time_res(:,1) = (time_res(:,1) - (mecha_sync - ER_sync));
                    time_res_strain = vlookup(time_res,time_strain,2);
                    % convert to change in resistance
                    time_res_strain(:,2) = ((time_res_strain(:,2) - time_res_strain(1,2)) ./ time_res_strain(1,2))*100;
                    hold on ; 
                    plot(time_res_strain(:,1), time_res_strain(:,2)./time_res_strain(:,3),linecol{k})
                    plotlabel{k} = strcat(specname{k},':',arch,':',ann);
                    title('ER/Strain plot')
                    ylabel('ER/Strain plot')
                    xlabel('time , sec')
                    legend(plotlabel,'Location','NorthEast','Fontsize',12)    
                    ylim([0 10e3])
                
                case 'Time,Electrical Conductance, Tangent Modulus'                    
                    %%
                    if length(specname) > 1
                        errordlg('Cannot perform multispecimen analysis')
                        close(curfig)
                        return
                    end
                    Hz = 1/(mydata.machine_data.selection(2,1) - mydata.machine_data.selection(1,1))
                    span = Hz / 0.5  % smoothing parameter % good when Hz = 100
                    stress = smooth(mydata.machine_data.selection(:,5),span,'moving');
                    strain = smooth(mydata.machine_data.selection(:,4),span,'moving');   
                    time_MTS = mydata.machine_data.selection(:,1)
                    delta = round(Hz/3) % 3 jump when finding tangent modulus , 35 is good for 100 Hz, 200 is good for 100 Hz DAQ
                    PL = 0.005;   % strain proportiona limit
                    % stress at which strain == PL
                    stressPL_point = stress(find((strain > PL),1,'first'));
                    % linear-elastic modulus 
                    EPL = stressPL_point / PL;
                    % strain for plotting linear curve
                    strainPL = strain - PL;
                    % stress for plotting linear curve
                    stressPL = EPL*strainPL;
                    yieldindex = find(stressPL >= stress,1,'first');
                    strainPL = strainPL(1:yieldindex)+PL;
                    stressPL = stressPL(1:yieldindex);
                    yieldstrain = strainPL(yieldindex);
                    yieldstress = stressPL(yieldindex);
                    E_s = stress./strain;
                    % linear fit method
                    k1=1;
                    for k2 = 1:delta:length(stress)-delta
                        % perform the fit and plot
                        fitobject = fit(strain(k2:k2+delta),stress(k2:k2+delta),'poly1');
                        stress_fit = fitobject.p1.*strain(k2:k2+delta) + fitobject.p2;
                        E_t(k1) = fitobject.p1;    
                        strain_mod(k1) = mean(strain(k2:k2+delta));  % avearage over the length of fitting
                        stress_mod(k1) = mean(stress(k2:k2+delta));
                        time_mod(k1) = mean(time_MTS(k2:k2+delta));   
                        k1=k1+1;
                    end
                    % smoothing method for E_t
                    span2 = span/8;   % smoothing parameter
                    E_t = smooth(E_t,span2,'lowess');
                    % % find loopss and calculaute strain energy dissappted
                    strainincr = max(strain);
                    [~, loopindex] = findpeaks(-1*strain);
                    loopindex = [1; loopindex];

                    % find the modulus peaks and plot
                    [Epeaks, Elocs] = findpeaks(E_t);
                    strain_mod_peaks = strain_mod(Elocs);
                    stress_mod_peaks = stress_mod(Elocs);
                    strain_beginloops = 0.1;
                    Eind = strain_mod_peaks > strain_beginloops;
                    E_unload = Epeaks(Eind);
                    stress_unload = stress_mod_peaks(Eind);                    
                    strain_mod(Elocs);
                    strain_unload = strain_mod_peaks(Eind);
                    try
                        assert(~isempty(strain_unload))
                    catch
                        disp('no loops, quitting')
                        continue
                    end
                    Einv_t = 1./E_t;
                    Einv_t = smooth(Einv_t,span2,'lowess');                     
                    time_stress = [time_MTS stress];
                    % time_strain = [time strain];
                    time_res = mydata.ER_data.selection(:,[1 2]);
                    ER_sync = mydata.plot_parameters.ER_sync;
                    mecha_sync = mydata.plot_parameters.mecha_sync;
                    time_res(:,1) = (time_res(:,1) - (mecha_sync - ER_sync));
                    time_res_stress = vlookup(time_res,time_stress,2);
                    % time_res_strain = vlookup(time_res,time_strain,2);
                    time_con_stress = time_res_stress;
                    time_con_stress(:,2) = 1./time_res_stress(:,2);
                    time_con_stress(:,2) = ((time_con_stress(:,2) - time_con_stress(1,2)) ./ time_con_stress(1,2))*100;
                    time_con_stress(:,2) = smooth(time_con_stress(:,2),10);
                    
                    subplot(1,2,1)
                    plot(time_con_stress(:,3),time_con_stress(:,2))
                    ylabel(' \Delta \kappa  , \Omega^{-1}')
                    xlabel('\sigma , Mpa')                    
                    ylim([min(time_con_stress(:,2)) 0])
                    title('Stress-Conductivity')
                    
                    subplot(1,2,2)
                    plot(stress_mod, E_t)
                    ylabel('E_{tan} , MPa')
                    xlabel('\sigma , Mpa')    
                    ylim([0 E_t(1)])
                    title('Stress-Tangent Modulus')
                    set(gcf,'position',[200 1180 1100 400])
                    newfontsize = 14
                    
                    figure
                    subplot(2,1,1)
                    plot(time_mod, Einv_t)
                    ylim([0 0.006 ])
                    xlabel('time')
                    ylabel('E_{tan}^{-1}')
                    subplot(2,1,2)
                    plot(time_res_stress(:,1),time_res_stress(:,2))
                    xlabel('time')
                    ylabel('Resistance, Ohm')
                    
                    figure
                    subplot(2,1,1)
                    plot(stress_mod, Einv_t)
                    ylim([0 0.006 ])
                    xlabel('stress')
                    ylabel('E_{tan}^{-1}')
                    subplot(2,1,2)
                    plot(time_res_stress(:,3),time_res_stress(:,2))
                    xlabel('stress')
                    ylabel('Resistance, Ohm')
                    
                case 'Spectral analysis'  
                    %%
                    events = mydata.AE_data.selection3(:,strcmp(mydata.AE_data.selection3_columnname,'Index'));
                    eventnum = inputdlg(sprintf('Please enter a single gage event from 1 to %i',length(events)));
                    eventnum = str2num(eventnum{:});
                    Spectral_analysis(mydata,events(eventnum))
                 
                    
            end % switch for plotting
            % change fontsize of everything in the figure
            ax = findobj(gcf,'type','axes');
            set(ax,'FontSize',newfontsize)
            if length(get(ax,'xlabel')) == 1
                h_xlabel = get(ax,'xlabel');
                h_ylabel = get(ax,'ylabel');
                h_title = get(ax,'title');
            else
                h_xlabel = cell2mat(get(ax,'xlabel'));
                h_ylabel = cell2mat(get(ax,'ylabel'));
                h_title = cell2mat(get(ax,'title'));
            end
            set(h_xlabel, 'FontSize', newfontsize)  
            set(h_ylabel, 'FontSize', newfontsize) 
            set(h_title , 'FontSize', newfontsize) 
            set(curfig,'Color',[1 1 1])
            set(curfig,'Name',strcat(plottype))
        end % for
end % command_str

function h_close_fig(h_fig0,eventdata)
% if the x is closed 
delete(gcf)