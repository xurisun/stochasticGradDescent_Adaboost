function stochasticGradDescent_Adaboost
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
%{
temp=full(sparse(Y(:,1),Y(:,2),Y(:,3)));
nusers=size(temp,1);
nitems=size(temp,2);
for i=1:nusers
    [tempRow,tempCol,tempRating]=find(temp(i,:));
    userBought=length(tempRating);
    p=randperm(userBought);
    tempTestNumber=floor(userBought/5);
    tempTrainNumber=userBought-tempTestNumber;
    tempP1=p(1:tempTestNumber);
    tempP2=p(tempTestNumber+1:end);
    if (i==1)
        testY = [ones(tempTestNumber,1)*i,tempCol(tempP1)',tempRating(tempP1)'];
        trainY =[ones(tempTrainNumber,1)*i,tempCol(tempP2)',tempRating(tempP2)'];
    else
        testY = [testY ;[ones(tempTestNumber,1)*i,tempCol(tempP1)',tempRating(tempP1)']];
        trainY =[trainY ;[ones(tempTrainNumber,1)*i,tempCol(tempP2)',tempRating(tempP2)']];
    end
end
%}

numTrain = length(trainY);
trainY(:,4) = ones(numTrain,1)*(1/numTrain);
originWeight = 1/numTrain;

trainSet = sparse(trainY(:,1),trainY(:,2),trainY(:,3));
weightSet = sparse(trainY(:,1),trainY(:,2),trainY(:,4));
testSet = sparse(testY(:,1),testY(:,2),testY(:,3));

testRMSE = 2;

%% Training Phase --

L = full(trainSet);
k=20;
%save configs2;
trainSet = sparse(L);
nusers = size(L,1);%1429
nitems = size(L,2);%3191

mu = sum(L(:))/nnz(L); %3.62
P = rand(nusers,k)/10;
Q = rand(nitems,k)/10;

[row,col,rating] = find(trainSet);
[row,col,weight] = find(weightSet);
[rowT,colT,ratingT] = find(testSet);

lambda = 0.02; % penalty term
alpha = 0.005;  % learning rate
numPasses = 100;


Bu = zeros(nusers,1);
Bi = zeros(nitems,1);

threshold = 0.7; % if err large than this value,regard it as -1,else 1
nIterations = 10;
% record the weight after each iteration
weightArray = zeros(nIterations,1);
finalQ = zeros(nitems,k);
finalP = zeros(nusers,k);
finalBu = zeros(nusers,1);
finalBi = zeros(nitems,1);

weightRight = 1;
weightWrong = 1;

for iteration = 1:nIterations,
    
    iteration
    for pass=1:numPasses,
        tempLength = length(row);
        prm = randperm(tempLength);
        errRate = 0;
        
        for i=1:tempLength
            user = row(prm(i));
            item = col(prm(i));
            err = rating(prm(i)) - mu - P(user,:)*Q(item,:)' -Bu(user) -Bi(item);
            if(err > threshold || err < -threshold)
                %trainWeight = weightWrong/weightRight;
                trainWeight = 1;
            else
                %trainWeight = weightRight/weightWrong;
                trainWeight = 1;
            end
            P(user,:) = P(user,:) + alpha*(trainWeight*err*Q(item,:) - lambda*P(user,:));
            Q(item,:) = Q(item,:) + alpha*(trainWeight*err*P(user,:) - lambda*Q(item,:));
            Bu(user) = Bu(user) + alpha*(trainWeight*err-lambda*Bu(user));
            Bi(item) = Bi(item) + alpha*(trainWeight*err-lambda*Bi(item));
        end
        
        errors=0;
        for i=1:tempLength
            user = row(i);
            item = col(i);
            predictRating = mu + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
            err = rating(i) - predictRating;
            errors = errors + err^2;
        end
        fprintf('Training rmse pass %d: \t %e\n', pass, sqrt(errors/tempLength));
        
        tempLength = length(colT);
        errors=0;
        for i=1:tempLength
            user = rowT(i);
            item = colT(i);
            predictRating = mu + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
            err = ratingT(i) - predictRating;
            errors = errors + err^2;
        end
        
        if(testRMSE > sqrt(errors/tempLength) )
            testRMSE = sqrt(errors/tempLength);
            %fprintf('!!!!Test rmse:%e\n',testRMSE);
        else 
            break;
        end
        fprintf('Test rmse pass %d: \t %e\n', pass,sqrt(errors/tempLength));
        
    end
    
    %% calculate the error and update
    tempLength = length(row);
    for i=1:tempLength
        user = row(i);
        item = col(i);
        predictRating = mu + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
        err = rating(i) - predictRating;
        % fprintf('err:%d \t %e\n',i,err);
        if(err > threshold || err < -threshold)
            errRate = errRate + weight(i);
            %  fprintf('err:%d \t %e\n',i,err);
        end
    end
    
    errRate
    if(errRate > 0.5)
        break;
    end
    
    weightRate = (1/2)*log((1-errRate)/errRate) ; % the weight of this trainning iteration
    weightRate;
    weightArray(iteration) = weightRate;
    %weightArray
    
    weightRight = exp(-weightRate);
    weightWrong = exp(weightRate);
    
    temp = (weightRight*(1 - errRate) + weightWrong*errRate)* numTrain;
    weightRight = weightRight/temp;
    weightWrong = weightWrong/temp;
    
    tempLength = length(col);
    for i=1:tempLength
        user = row(i);
        item = col(i);
        predictRating = mu + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
        err = rating(i) - predictRating;
        if(err > threshold || err < -threshold)
            weight(i) = weightWrong;
            %  fprintf('err:%d \t %e\n',i,err);
        else
            weight(i) = weightRight;
        end
    end
    
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
    predictRating = mu + finalP(user,:)*finalQ(item,:)' + finalBu(user) + finalBi(item);
    err = rating(i) - predictRating;
    errors = errors + err^2;
end
fprintf('Adaboost Training rmse: \t %e\n', sqrt(errors/tempLength));

tempLength = length(colT);
errors=0;
for i=1:tempLength
    user = rowT(i);
    item = colT(i);
    predictRating = mu + finalP(user,:)*finalQ(item,:)' + finalBu(user) + finalBi(item);
    err = ratingT(i) - predictRating;
    errors = errors + err^2;
end
fprintf('Adaboost Test rmse: \t %e\n',sqrt(errors/tempLength));






