% Khrisna Kamarga
% AMATH482 - HW4 Automate
close all; clear all; clc;
display("Test1")
folder = 1;
save automate
Kamarga_AMATH482_HW4
save log_test1

close all; clear all; clc;
display("Test2")
folder = 2;
save automate
Kamarga_AMATH482_HW4
save log_test2

close all; clear all; clc;
display("Test3")
folder = 3;
save automate
Kamarga_AMATH482_HW4
save log_test3
close all; clear all; clc;

% %%
% clear all; close all; clc
% load log_test3
% 
% figure(1)
% plot(1:length(lambdaBig), lambdaBig,'rx');
% title("Energy Plot")
% xlabel("Principal Component")
% ylabel("Energy")
% 
% figure(2)
% plot3(Xtrain(1:100,1), Xtrain(1:100, 2), Xtrain(1:100, 3), 'rx');
% hold on
% plot3(Xtrain(101:200,1), Xtrain(101:200, 2), Xtrain(101:200, 3), 'bo');
% hold on
% plot3(Xtrain(201:end,1), Xtrain(201:end, 2), Xtrain(201:end, 3), 'k.');
% legend group1 group2 group3
% % hold on
% % plot3(Xpredict(1:labelLength(1),1), Xpredict(1:labelLength(1), 2), Xpredict(1:labelLength(1), 3), 'rx', 'LineWidth', 5);
% % hold on
% % plot3(Xpredict(labelLength(1)+1:labelLength(2),1), Xpredict(labelLength(1)+1:labelLength(2), 2), Xpredict(labelLength(1)+1:labelLength(2), 3), 'bo', 'LineWidth', 5);
% % hold on
% % plot3(Xpredict(labelLength(2)+1:end,1), Xpredict(labelLength(2)+1:end, 2), Xpredict(labelLength(2)+1:end, 3), 'k^', 'LineWidth', 5);
% % legend group1 group2 group3