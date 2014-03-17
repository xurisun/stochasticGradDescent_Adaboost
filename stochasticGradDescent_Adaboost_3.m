function stochasticGradDescent_Adaboost
%{
Y = load('20data.dat');% 225084
%Y = load('test1.dat');% 225084
p = randperm(length(Y));
Y(:,1) = Y(p,1);
Y(:,2) = Y(p,2);
Y(:,3) = Y(p,3);

num = length(Y);
first=1;
testSize=floor(num/5);
last = first+testSize-1;

testY = Y(first:last,:);%45016
trainY = Y((last+1):end,:); % 180068

numTrain = length(trainY);
trainY(:,4) = ones(numTrain,1)*(1/numTrain);
originWeight = 1/numTrain;
originWeight
trainSet = sparse(trainY(:,1),trainY(:,2),trainY(:,3));
weightSet = sparse(trainY(:,1),trainY(:,2),trainY(:,4));
testSet = sparse(testY(:,1),testY(:,2),testY(:,3));



%% Training Phase --

L = full(trainSet);
k=20;
%save configs2;
trainSet = sparse(L);
nusers = size(L,1);%1429
nitems = size(L,2);%3191

u = sum(L(:))/nnz(L); %3.62
P = rand(nusers,k)/10;
Q = rand(nitems,k)/10;
Bu = zeros(nusers,1);
Bi = zeros(nitems,1);

finalQ = zeros(nitems,k);
finalP = zeros(nusers,k);
finalBu = zeros(nusers,1);
finalBi = zeros(nitems,1);

[row,col,rating] = find(trainSet);
[row,col,weight] = find(weightSet);
[rowT,colT,ratingT] = find(testSet);
%}
load test10W
% load test

tempQ = zeros(nitems,k);
tempP = zeros(nusers,k);
tempBu = zeros(nusers,1);
tempBi =zeros(nitems,1);

threshold = 0.25; % if err large than this value,regard it as -1,else 1
nIterations = 10;
testRMSE = 2;
% record the weight after each iteration
weightArray = zeros(nIterations,1);

lambda = 0.02; % penalty term
alpha = 0.005;  % learning rate
numPasses = 100;
count = 0;
for iteration = 1:nIterations,
    iteration
    %     tempLength = length(row);
    %     errRate = 0;
    %     A = 0;
    %     for i=1:tempLength
    %         user = row(i);
    %         item = col(i);
    %         predictRating = u + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
    %         err = rating(i) - predictRating;
    %         A = A + abs(err);
    %     end
    %     threshold = A/tempLength
    
    for pass=1:numPasses,
        tempLength = length(row);
        prm = randperm(tempLength);
        errRate = 0;
        for i=1:tempLength
            user = row(prm(i));
            item = col(prm(i));
            err = rating(prm(i)) - u - P(user,:)*Q(item,:)' -Bu(user) -Bi(item);
            
            trainWeight = weight(prm(i))/originWeight;
            
            P(user,:) = P(user,:) + trainWeight*alpha*(err*Q(item,:) - lambda*P(user,:));
            Q(item,:) = Q(item,:) + trainWeight*alpha*(err*P(user,:) - lambda*Q(item,:));
            Bu(user) = Bu(user) + trainWeight*alpha*(err-lambda*Bu(user));
            Bi(item) = Bi(item) + trainWeight*alpha*(err-lambda*Bi(item));
            
            %             P(user,:) = P(user,:) + alpha*(trainWeight*err*Q(item,:) - lambda*P(user,:));
            %             Q(item,:) = Q(item,:) + alpha*(trainWeight*err*P(user,:) - lambda*Q(item,:));
            %             Bu(user) = Bu(user) + alpha*(trainWeight*err-lambda*Bu(user));
            %             Bi(item) = Bi(item) + alpha*(trainWeight*err-lambda*Bi(item));
        end
        
        errors=0;
        for i=1:tempLength
            user = row(i);
            item = col(i);
            predictRating = u + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
            err = rating(i) - predictRating;
            errors = errors + err^2;
        end
        fprintf('Training rmse pass %d: \t %e\n', pass, sqrt(errors/tempLength));
        
        
        tempLength = length(colT);
        errors=0;
        for i=1:tempLength
            user = rowT(i);
            item = colT(i);
            predictRating = u + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
            err = ratingT(i) - predictRating;
            errors = errors + err^2;
        end
        
        if(testRMSE > sqrt(errors/tempLength) )
            testRMSE = sqrt(errors/tempLength);
            %fprintf('!!!!Test rmse:%e\n',testRMSE);
                        tempQ = Q;
                        tempP = P;
                        tempBu = Bu;
                        tempBi = Bi;
                        count1 = 0;
        else
                        count1 = count1 + 1;
                        if(count1 > 10)
                            break;
                        end
%             break;
        end
        fprintf('Test rmse pass %d: \t %e\n', pass,sqrt(errors/tempLength));
    end
    %
        Q = tempQ;
        P = tempP;
        Bu = tempBu;
        Bi = tempBi;
    
    %% calculate the error and update
    tempLength = length(row);
    for i=1:tempLength
        user = row(i);
        item = col(i);
        predictRating = u + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
        err = abs(rating(i) - predictRating);
        % fprintf('err:%d \t %e\n',i,err);
        if(err/rating(i) > threshold)
             count = count + 1;
            errRate = errRate + weight(i);
            %  fprintf('err:%d \t %e\n',i,err);
        end
    end
    count
    %         errRate = errRate*errRate;
%       errRate = sqrt(errRate);
    errRate
%     if(errRate > 0.5)
%         weight =  ones(tempLength,1)*(1/numTrain);
%         continue;
%         break;
%     end
    
%     weightRate = 0.5*log((1-errRate)/errRate) ; % the weight of this trainning iteration
        weightRate = log(1/errRate); % another way to calculate weight of
    %     temp traning iteration
    weightRate
    weightArray(iteration) = weightRate;
    %weightArray
    
%     weightRight = exp(-weightRate);
%     weightWrong = exp(weightRate);
%     weightRight
%     weightWrong
    %%  Normalization
    tempLength = length(row);
    for i=1:tempLength
        user = row(i);
        item = col(i);
        predictRating = u + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
        err = abs(rating(i) - predictRating);
        % fprintf('err:%d \t %e\n',i,err);
        if(err/rating(i) > threshold)
%             weight(i) = weight(i) * weightWrong;
                        weight(i) = weight(i) * 1;
            %             fprintf('Wrong: \t %e\n',weight(i));
        else
%             weight(i) = weight(i) * weightRight;
                        weight(i) = weight(i) * sqrt(errRate);
            %            fprintf('Right: \t %e\n',weight(i));
        end
    end
    weight = weight/sum(weight);
    %     weight
    finalQ = finalQ + weightArray(iteration)* Q;
    finalP = finalP + weightArray(iteration)* P;
    finalBu = finalBu + weightArray(iteration)* Bu;
    finalBi = finalBi + weightArray(iteration)* Bi;
    
    P = rand(nusers,k)/10;
    Q = rand(nitems,k)/10;
    Bu = zeros(nusers,1);
    Bi = zeros(nitems,1);
    testRMSE = 2;
    
end
%% Get the final result
sumWeight = sum(weightArray);
finalQ = finalQ/sumWeight;
finalP = finalP/sumWeight;
finalBu = finalBu/sumWeight;
finalBi = finalBi/sumWeight;

errors=0;
tempLength = length(col);
for i=1:tempLength
    user = row(i);
    item = col(i);
    predictRating = u + finalP(user,:)*finalQ(item,:)' + finalBu(user) + finalBi(item);
    err = rating(i) - predictRating;
    errors = errors + err^2;
end
fprintf('Adaboost Training rmse: \t %e\n', sqrt(errors/tempLength));

tempLength = length(colT);
errors=0;
for i=1:tempLength
    user = rowT(i);
    item = colT(i);
    predictRating = u + finalP(user,:)*finalQ(item,:)' + finalBu(user) + finalBi(item);
    err = ratingT(i) - predictRating;
    errors = errors + err^2;
end
fprintf('Adaboost Test rmse: \t %e\n',sqrt(errors/tempLength));


