clc; clear all; close all; %Clear terminal, workspace and close any open files before running
disp('SBJ00 S00:  RUNNING  | Starting CNN Test (SBJ1-15, S1-7)'); %Print in the terminal that indicates the start of the test
SBJS=fopen('/Users/mysftz/BCI/SBJ-S/AllSBJ-S.txt','r'); %Open a file outlining the all the subjects, sessions 1-7 and cell ranges for the matrix to print too
rt = tic; %Starting a timer, to measure the run time of each loop and overall test

for i=1:105 %15 subjects * sessions 1 to 7 = 105 sessions to run in total, also the file opening in line 3 contains 105 lines
    clear S; %Ensuring data isn't being taken from the woorkspace if line 1 fails to execute
    S=fgetl(SBJS); %Based on line i in the file in line 3, gets line/ row data
    sub=S(1:5); sess=S(7:9); cellrange=S(11:13); matrixrange=S(15:end); %Characters 1 to 5 on row i will be labelled sub (subject), 7 to 9 labelled as sess (session), 11 to 13 is the cellrange to print to an excel sheet and the same for the matrixrange aswell. cellrange is 1 row by 45 columns and matrixrange is 45 rows by 50 columns

    %Load relevant data from subject and the session in the dataset folder
    load (strcat('/Users/mysftz/BCI/Dataset/', sub, '/', sess, '/trainData.mat'));
    load (strcat('/Users/mysftz/BCI/Dataset/', sub, '/', sess, '/testData.mat'));
    target=load(strcat('/Users/mysftz/BCI/Dataset/', sub, '/', sess, '/trainTargets.txt')); 
    label=load(strcat('/Users/mysftz/BCI/Dataset/', sub, '/', sess, '/testLabels.txt')); 
    event=load(strcat('/Users/mysftz/BCI/Dataset/', sub, '/', sess, '/testEvents.txt')); 
    runs=load(strcat('/Users/mysftz/BCI/Dataset/', sub, '/', sess, '/runs_per_block.txt'));

    [sess_accuracy, predicted_object] = BCI_CNN_Function(event, label, runs, sess, sub, target, testData, trainData); %Send all labels and data to the function
    runtime=toc(rt); %saves loop run time and at the end overall run time

    writematrix(predicted_object, '/Users/mysftz/BCI/Results/CNN_Results_S1_7.xlsx', 'Range', matrixrange); %Save results data
    writematrix(sess_accuracy, '/Users/mysftz/BCI/Results/CNN_Accuracy_S1_7.xlsx', 'Range', cellrange); %Save accuracy data
    writematrix(runtime, '/Users/mysftz/BCI/Results/CNN_RunTime_S1_7.xlsx', 'Range', cellrange); %Save run time data
    disp([sub, ' ', num2str(sess), ': COMPLETED | ', num2str(sess_accuracy), '% Accuracy in ', num2str(runtime), 's']); %Displays accuracy and run time in the terminal
end
disp('SBJ00 S00:  ENDING   | Ending CNN Test (SBJ1-15, S1-7)'); %Print in the terminal that indicates the end of the test
fclose(SBJS); %Closes the file