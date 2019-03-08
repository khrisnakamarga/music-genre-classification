% % Khrisna Kamarga
% % AMATH 482 - Homework 4
% clear all; close all; clc;
% 
% cd 'C:\Users\Khrisna Adi Kamarga\Desktop\AMATH 482\HW4'
% cd 'yalefaces';
% 
% files = dir('*.*');
% files(1:2) = [];
% for i=1:length(files)
%     eval(['yalefaces' num2str(i) ' = imread(files(i).name);']);
% end
% 
% cd ..
% cd 'CroppedYale'
% 
% files = dir('*.*');
% files(1:2) = [];
% for i=1:9
%     eval(['cd yaleB0' num2str(i)]);
%     subFiles = dir('*.*');
%     subFiles(1:2) = [];
%     for j=1:length(subFiles)
%         eval(['yaleB' num2str(i) '_' num2str(j) ' = imread(subFiles(j).name);']);
%     end
%     cd ..
% end
% for i=[10:13, 15:length(files)+1]
%     eval(['cd yaleB' num2str(i)]);
%     subFiles = dir('*.*');
%     subFiles(1:2) = [];
%     for j=1:length(subFiles)
%         eval(['yaleB' num2str(i) '_' num2str(j) ' = imread(subFiles(j).name);']);
%     end
%     cd ..
% end
% cd ..
% save loaded
% 
% %% Cropped
% clear all; close all; clc;
% load loaded
% 
% X = [];
% for i=[1:13, 15:length(files)+1]
%     for j=1:length(subFiles)
%         eval(['currImage = reshape(yaleB' num2str(i) '_' num2str(j) ', [1 192*168]);']);
%         X = [X; currImage];
%     end
% end
% clearvars -except X
% %% 1)
% clc;
% X=double(X');
% [m,n]=size(X); % compute data siz
% [U S V] = svd(X/sqrt(n-1), 'econ');
% lambda=diag(S).^2; % produce diagonal variances
% Y=U*X; % produce the principal components projection
% %% 2)
% close all; clc;
% figure(1)
% plot(1:length(lambda), lambda,'rx');
% title("Energy Plot")
% xlabel("Principal Component")
% ylabel("Energy")
% 
% figure(2)
% for i=1:12
%     subplot(3,4,i)
%     pcolor(flipud(reshape(X(i,:), 192, 168))), shading interp, colormap hot
% end
% 
% figure(3)
% for i=1:12
%     subplot(3,4,i)
%     pcolor(flipud(reshape(Y(i,:), 192, 168))), shading interp, colormap hot
% end
% 
% %% 3)
% clc;
% 
% % Reducing the rank
% r = 4;
% U = U(:,1:r); S = S(1:r,1:r); V = V(:,1:r);
% Xr = U*S*V';
% 
% figure(4)
% for i=1:12
%     subplot(3,4,i)
%     pcolor(flipud(reshape(Xr(i,:), 192, 168))), shading interp, colormap hot
% end
% 
% %% 4) - Repeat 1) 2) 3) for uncropped
% 
% clear all; close all; clc;
% load loaded
% 
% X = [];
% for i=1:length(files)
%     eval(['currImage = reshape(yalefaces' num2str(i) ', [1 243*320]);']);
%     X = [X; currImage];
% end
% clearvars -except X
% %% 1)
% clc;
% X=double(X);
% [m,n]=size(X); % compute data siz
% [U S V] = svd(X/sqrt(n-1), 'econ');
% lambda=diag(S).^2; % produce diagonal variances
% Y=U'*X; % produce the principal components projection
% %% 2)
% close all; clc;
% figure(1)
% plot(1:length(lambda), lambda,'rx');
% title("Energy Plot")
% xlabel("Principal Component")
% ylabel("Energy")
% 
% figure(2)
% for i=1:12
%     subplot(3,4,i)
%     pcolor(flipud(reshape(X(i,:), 243, 320))), shading interp, colormap hot
% end
% 
% figure(3)
% for i=1:12
%     subplot(3,4,i)
%     pcolor(flipud(reshape(Y(i,:), 243, 320))), shading interp, colormap hot
% end
% 
% %% 3)
% clc;
% 
% % Reducing the rank
% r = 5;
% U = U(:,1:r); S = S(1:r,1:r); V = V(:,1:r);
% Xr = U*S*V';
% 
% figure(4)
% for i=1:12
%     subplot(3,4,i)
%     pcolor(flipud(reshape(Xr(i,:), 243, 320))), shading interp, colormap hot
% end

%% Music Classification
clear all; close all; clc;
load automate

eval(['cd test' num2str(folder)])
files = dir('*.*');
files(1:2) = [];
for i=1:5
    eval(['test1_1_' num2str(i) ' = audioread(files(i).name);']);
end
for i=6:10
    eval(['test1_2_' num2str(i-5) ' = audioread(files(i).name);']);
end
for i=11:15
    eval(['test1_3_' num2str(i-10) ' = audioread(files(i).name);']);
end
cd ..
%% X Preparation by Reading Clips
display("preparing big X");

Fs = 44000; % sampling frequency
downSample = 8; % down sampling ratio
Fs = Fs/downSample; % new sampling frequency
T = 1/Fs; % sampling rate
fiveSeconds = 5/T; % data points for 5 seconds
X = [];
labelLength = [];
len = 0;
for i=1:3
    for j=1:5
        eval(['to = 1:length(test1_' num2str(i) '_' num2str(j) ');']);
        t = linspace(min(to),max(to), length(to)/downSample);
        samples = 20;
        eval(['temp = interp1(to, test1_' num2str(i) '_' num2str(j) ', t, ''linear'');']);
        for k=1:samples
            start = 5*fiveSeconds + round(k*fiveSeconds); % the start of sampling 5 seconds every 40 seconds
            eval(['clip' num2str(i) '_' num2str(j) '_' num2str(k) ' = temp(start:start+fiveSeconds-1);']);
            eval(['clip' num2str(i) '_' num2str(j) '_' num2str(k) ' = mean(clip' num2str(i) '_' num2str(j) '_' num2str(k) ',1);']);
            eval(['X = [X; clip' num2str(i) '_' num2str(j) '_' num2str(k) '];']);
        end
        len = len + samples; %amount of data points for each group
    end
    labelLength = [labelLength len];
end

%% Spectrogram Generation
clearvars -except X Fs labelLength;
display("Preparing Spectrograms");

allGabor = [];
for i=1:size(X,1)
    v = X(i,:); % current clip
    t = (1:length(v))/Fs; % making time vector (length Fs)
    L = max(t); n = length(t); % prepare the variables for fft
    k=(2*pi/L)*[0:n/2-1 -n/2:-1];

    t_sample = 0.1; %sampling rate

    tslide = 0:t_sample:L; % sampling time
    twindow = 20; % the width of the super gaussian
    spc=[]; %matrix of all the wavelets
    for j=1:length(tslide)
        g = exp(-(twindow*(t-tslide(j))).^10); % super gaussian
        vf=g.*v;
        yft=fft(vf);
        spc=[spc;abs(fftshift(yft))];
    end
    spc = spc.';
    spc(1:round(size(spc, 1)/2),:) = [];
%     %visualizing the spectrogram
%     pcolor(spc), shading interp, colormap(hot)
%     drawnow
    [m n] = size(spc);
    allGabor = [allGabor; reshape(spc, 1, m*n)];
    if (mod(i,10) == 0 | i == size(X,1))
        fprintf('%f percent completed \n', i/size(X,1)*100);
    end
end

%% SVD
clearvars -except allGabor labelLength;
display("Performing SVD")

n = size(allGabor, 1);
[U S V] = svd(allGabor/sqrt(n-1), 'econ');
lambdaBig=diag(S).^2; % produce diagonal variances
Y=U.'*allGabor; % produce the principal components projection
% figure(1)
% plot(1:length(lambdaBig), lambdaBig,'rx');
% title("Energy Plot")
% xlabel("Principal Component")
% ylabel("Energy")

%% Training Data Preparation
display("Preparing Training Data");

Xtrain = [];
for i=1:4
    projection = [];
    for j=1:size(allGabor, 1)
        projection = [projection; dot(allGabor(j,:),Y(i,:))];
    end
    Xtrain = [Xtrain projection];
end

% figure(2)
% hold on
% plot(Xtrain(1:labelLength(1),1), Xtrain(1:labelLength(1), 2), 'rx');
% plot(Xtrain(labelLength(1)+1:labelLength(2),1), Xtrain(labelLength(1)+1:labelLength(2), 2), 'bo');
% plot(Xtrain(labelLength(2)+1:end,1), Xtrain(labelLength(2)+1:end, 2), 'k.');
% legend group1 group2 group3

%% Naive Bayes Training Model
display("Creating Naive Bayes Training Model");

label = [];
for i=1:labelLength(1)
    label = [label; "group1"];
end
for i=labelLength(1)+1:labelLength(2)
    label = [label; "group2"];
end
for i=labelLength(2)+1:length(Xtrain)
    label = [label; "group3"];
end

mdl = fitcnb(Xtrain, label);

%% Resample
display("Resampling new clips")

load automate

eval(['cd test' num2str(folder)])

files = dir('*.*');
files(1:2) = [];
for i=1:5
    eval(['test1_1_' num2str(i) ' = audioread(files(i).name);']);
end
for i=6:10
    eval(['test1_2_' num2str(i-5) ' = audioread(files(i).name);']);
end
for i=11:15
    eval(['test1_3_' num2str(i-10) ' = audioread(files(i).name);']);
end
cd ..

Fs = 44000; % sampling frequency
downSample = 8; % down sampling ratio
Fs = Fs/downSample; % new sampling frequency
T = 1/Fs; % sampling rate
fiveSeconds = 5/T; % data points for 5 seconds
X = [];
labelLength = [];
len = 0;
for i=1:3
    for j=1:5
        eval(['to = 1:length(test1_' num2str(i) '_' num2str(j) ');']);
        t = linspace(min(to),max(to), length(to)/downSample);
        samples = 3;
        eval(['temp = interp1(to, test1_' num2str(i) '_' num2str(j) ', t, ''linear'');']);
        for k=1:samples
            start = 20*fiveSeconds + round(k*fiveSeconds*3); % the start of sampling 5 seconds every 40 seconds
            eval(['clip' num2str(i) '_' num2str(j) '_' num2str(k) ' = temp(start:start+fiveSeconds-1);']);
            eval(['clip' num2str(i) '_' num2str(j) '_' num2str(k) ' = mean(clip' num2str(i) '_' num2str(j) '_' num2str(k) ',1);']);
            eval(['X = [X; clip' num2str(i) '_' num2str(j) '_' num2str(k) '];']);
        end
        len = len + samples; %amount of data points for each group
    end
    labelLength = [labelLength len];
end

%% Regenerate Spectrogram
clearvars -except X Fs labelLength mdl Xtrain lambdaBig;
display("Regenerating Spectrograms")

allGabor = [];
for i=1:size(X,1)
    v = X(i,:); % current clip
    t = (1:length(v))/Fs; % making time vector (length Fs)
    L = max(t); n = length(t); % prepare the variables for fft
    k=(2*pi/L)*[0:n/2-1 -n/2:-1];

    t_sample = 0.1; %sampling rate

    tslide = 0:t_sample:L; % sampling time
    twindow = 20; % the width of the super gaussian
    spc=[]; %matrix of all the wavelets
    for j=1:length(tslide)
        g = exp(-(twindow*(t-tslide(j))).^10); % super gaussian
        vf=g.*v;
        yft=fft(vf);
        spc=[spc;abs(fftshift(yft))];
    end
%     visualize the spectrogram
%     pcolor(tslide,fftshift(k)/(2*pi),spc.'), shading interp, colormap(hot)
    spc = spc.';
    spc(1:round(size(spc, 1)/2),:) = [];
    [m n] = size(spc);
    allGabor = [allGabor; reshape(spc, 1, m*n)];
    if (mod(i,10) == 0 | i == size(X,1))
        fprintf('%f percent completed \n', i/size(X,1)*100);
    end
end

%% SVD of the Resample
clearvars -except allGabor labelLength label mdl Xtrain lambdaBig;
display("SVDing the resampled data")

n = size(allGabor, 1);
[U S V] = svd(allGabor/sqrt(n-1), 'econ');
lambda=diag(S).^2; % produce diagonal variances
Y=U.'*allGabor; % produce the principal components projection
% figure(3)
% plot(1:length(lambda), lambda,'rx');
% title("Energy Plot")
% xlabel("Principal Component")
% ylabel("Energy")

clc;
Xpredict = [];
for i=1:4
    projection = [];
    for j=1:size(allGabor, 1)
        projection = [projection; dot(allGabor(j,:),Y(i,:)')];
    end
    Xpredict = [Xpredict projection];
end
%%
% figure(2)
% hold on
% plot(Xpredict(1:labelLength(1),1), Xpredict(1:labelLength(1), 2), 'rx', 'LineWidth', 5);
% plot(Xpredict(labelLength(1)+1:labelLength(2),1), Xpredict(labelLength(1)+1:labelLength(2), 2), 'bo', 'LineWidth', 5);
% plot(Xpredict(labelLength(2)+1:end,1), Xpredict(labelLength(2)+1:end, 2), 'k^', 'LineWidth', 5);
% legend group1 group2 group3

%%
display("Predicting")

answer = [];
correctAnswer = 0;
for i=1:length(Xpredict)
    currAnswer = predict(mdl, Xpredict(i,:));
    currAnswer = string(currAnswer);
    answer = [answer currAnswer];
    if (i <= 15 & currAnswer == 'group1' | ...
        i> 15 & i <= 30 & currAnswer == 'group2' | ...
        i> 30 & currAnswer == 'group3')
        correctAnswer = correctAnswer +1;
    end
end

accuracy = correctAnswer/length(Xpredict) * 100;
display(answer)