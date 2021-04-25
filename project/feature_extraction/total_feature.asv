clear;
clc;
path_directory = '/Users/kyeomeunjang/Desktop/DataPaper'; 
original_files = dir([path_directory '/user_*']);
total = [];
label = [];
pos = [1, 2, 4:13, 15:22];
for i = pos
    F = [];
    filename = [path_directory+ "/" + original_files(i).name];
    %% user_info ㅇ
    %feature: bmi, age
    userInfo = readmatrix(filename + "/user_info.csv");
    weight = userInfo(:,3);
    height =  userInfo(:,4);
    bmi = weight/((height/100)^2);
    age = userInfo(:,5);
    userInfoFeatures = [bmi, age];
    %% sleep ㅇ
    %feature: Efficiency,Total Minutes in Bed, TST(totalsleeptime), Sleep
    %Fragmentation Index, The number of Awake
    sleep = readmatrix(filename + "/sleep.csv");
    sleep(isnan(sleep))=-1;
    efficiency = mean(sleep(:,9));
    totalInBed = sum(sleep(:,10));
    TST = sum(sleep(:,11));
    sleepFragment = mean(sleep(:,17));
    numAwake = sum(sleep(:,13));
    sleepFeatures = [efficiency, totalInBed, TST, sleepFragment, numAwake];
    %% questionnaire ㅇㅇ 
    %feature: Daily_stress, STAI1(anxiety)
    questionnaire = readmatrix(filename + "/questionnaire.csv");
    questionnaire(isnan(questionnaire))=-1;
    psqi = questionnaire(:,5);
    stress = questionnaire(:,6);
    anxiety = questionnaire(:,3);
    questionnaireFeatures = [stress, anxiety];
    %% Actigraph, RR ㅇㅇ 
    %feature: steps, mesor, acrophase, amplitude, HRRmean, SDNN
    Actigraph = readmatrix(filename + "/Actigraph.csv");
    heartRate = Actigraph(:, 6)';
    RR = readmatrix(filename + "/RR.csv");
    RR = RR(:,2)';
    hRHrvFeatures= extract(heartRate, RR);
    steps = sum(Actigraph(:, 5));
    actiRRFeatures = [steps, hRHrvFeatures];
    %% hormone ㅇㅇ 
     %feature:cortisol_wakeup,cortisol_wakeup 
    hormone = readmatrix(filename + "/saliva.csv");
    cortisol_wakeup = hormone(2,3);
    melatonin_beforesleep = hormone(1,4);
    hormoneFeatures = [cortisol_wakeup, melatonin_beforesleep];
   
    %% combine
    F = [F, userInfoFeatures, sleepFeatures, questionnaireFeatures, actiRRFeatures, hormoneFeatures];
    total = [total;F];
    label = [label; uint8(psqi<6)+1];
end 

%% activity 
%feature: caffein(lasttime), smoking(times a day), light movement(duration), medium(duration), heavy(duration)
activityFeatures = readmatrix(path_directory + "/activity.xlsx");
total = [total, activityFeatures];

save("totalFeature.mat", "total");
save("totalLabel.mat", "label");