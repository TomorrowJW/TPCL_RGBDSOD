clear all; close all; clc;

%SaliencyMap Path
SalMapPath = '/home/jasonwu/桌面/Github/Save_Results_2985/prediction';           
%Evaluated Models
Models = {'Net_2985'};
%Datasets
DataPath = '/home/jasonwu/0-Datasets/2-RGB-D/New_Dataset/test_dataset/';
Datasets = {'STERE', 'NLPR', 'SIP', 'LFSD', 'DUT-RGBD', 'NJU2K','STEREO','DES','ReDWeb','SSD'};

%Evaluated Score Results
ResDir = '/home/jasonwu/桌面/Github/Save_Results_2985/Results_2985/';

%Initial paramters setting
Thresholds = 1:-1/255:0;
datasetNum = length(Datasets);
modelNum = length(Models);

for d = 1:datasetNum
    
    tic;
    dataset = Datasets{d};
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    ResPath = [ResDir dataset '-mat/'];
    if ~exist(ResPath,'dir')
        mkdir(ResPath);
    end
    resTxt = [ResDir dataset '_result-overall.txt'];
    fileID = fopen(resTxt,'w');
    
    for m = 1:modelNum
        model = Models{m};
        
        gtPath = [DataPath dataset '/GT/'];
                
        salPath = [SalMapPath '/' dataset '/'];
        
        imgFiles = dir([gtPath '*.png']); %imgFiles = dir([salPath '*.png']);
        imgNUM = length(imgFiles);
        
        [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(imgNUM,length(Thresholds)));
        
        [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));
        
        [Smeasure, adpFmeasure, adpEmeasure, MAE] =deal(zeros(1,imgNUM));
        
        parfor i = 1:imgNUM  %parfor i = 1:imgNUM  You may also need the parallel strategy. 
            
            fprintf('Evaluating(%s Dataset,%s Model): %d/%d\n',dataset, model, i,imgNUM);
            name =  imgFiles(i).name;
            
            %load gt
            gt = imread([gtPath name]);
            
            if (ndims(gt)>2)
                gt = rgb2gray(gt);
            end
            
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            %load salency
            sal  = imread([salPath name]);
            
            %check size
            if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
                sal = imresize(sal,size(gt));
                imwrite(sal,[salPath name]);
                fprintf('Error occurs in the path: %s!!!\n', [salPath name]); %check whether the size of the salmap is equal the gt map.
            end
            
            sal = im2double(sal(:,:,1));
            
            %normalize sal to [0, 1]
            sal = reshape(mapminmax(sal(:)',0,1),size(sal));
            Sscore = StructureMeasure(sal,logical(gt));
            Smeasure(i) = Sscore;
            
            % Using the 2 times of average of sal map as the adaptive threshold.
            threshold =  2* mean(sal(:)) ;
            [~,~,adpFmeasure(i)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
            
            
            Bi_sal = zeros(size(sal));
            Bi_sal(sal>threshold)=1;
            adpEmeasure(i) = Enhancedmeasure(Bi_sal,gt);
            
            [threshold_F, threshold_E]  = deal(zeros(1,length(Thresholds)));
            [threshold_Pr, threshold_Rec]  = deal(zeros(1,length(Thresholds)));
            
            for t = 1:length(Thresholds)
                threshold = Thresholds(t);
                [threshold_Pr(t), threshold_Rec(t), threshold_F(t)] = Fmeasure_calu(sal,double(gt),size(gt),threshold);
                
                Bi_sal = zeros(size(sal));
                Bi_sal(sal>threshold)=1;
                threshold_E(t) = Enhancedmeasure(Bi_sal,gt);
            end
            
            threshold_Fmeasure(i,:) = threshold_F;
            threshold_Emeasure(i,:) = threshold_E;
            threshold_Precion(i,:) = threshold_Pr;
            threshold_Recall(i,:) = threshold_Rec;
            
            MAE(i) = mean2(abs(double(logical(gt)) - sal));
            
        end
        
        %Precision and Recall 
        column_Pr = mean(threshold_Precion,1);
        column_Rec = mean(threshold_Recall,1);
        
        %Mean, Max F-measure score
        column_F = mean(threshold_Fmeasure,1);
        meanFm = mean(column_F);
        maxFm = max(column_F);
        
        %Mean, Max E-measure score
        column_E = mean(threshold_Emeasure,1);
        meanEm = mean(column_E);
        maxEm = max(column_E);
        
        %Adaptive threshold for F-measure and E-measure score
        adpFm = mean2(adpFmeasure);
        adpEm = mean2(adpEmeasure);
        
        %Smeasure score
        Smeasure = mean2(Smeasure);
        
        %MAE score
        mae = mean2(MAE);
        
        %Save the mat file so that you can reload the mat file and plot the PR Curve
        save([ResPath model],'Smeasure', 'mae', 'column_Pr', 'column_Rec', 'column_F', 'adpFm', 'meanFm', 'maxFm', 'column_E', 'adpEm', 'meanEm', 'maxEm');
       
        fprintf(fileID, '(Dataset:%s; Model:%s) Smeasure:%.3f; MAE:%.3f; adpEm:%.3f; meanEm:%.3f; maxEm:%.3f; adpFm:%.3f; meanFm:%.3f; maxFm:%.3f.\n',dataset,model,Smeasure, mae, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm);   
    end
    toc;
    
end


