function stochasticGradDescent_Adaboost
Y = load('20data.dat');% 225084

p = randperm(length(Y));
Y(:,1) = Y(p,1);
Y(:,2) = Y(p,2);
Y(:,3) = Y(p,3);
num = length(Y);
first=1;
testSize=floor(num/5);
last = first+testSize-1;

testY = Y(first:last,:);%45016
trainY = [Y(1:(first-1),:);Y((last+1):end,:)]; % 180068
numTrain = length(trainY);
trainY(:,4) = ones(numTrain,1)*(1/numTrain);
originWeight = 1/numTrain
%weight
%errRate

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

mu = sum(L(:))/nnz(L);
P = rand(nusers,k)/10;
Q = rand(nitems,k)/10;

[row,col,rating] = find(trainSet);
[row,col,weight] = find(weightSet);
[rowT,colT,ratingT] = find(testSet);

lambda = 0.02; % penalty term
alpha = 0.005;  % learning rate
numPasses = 30;

Bu = zeros(nusers,1);
Bi = zeros(nitems,1);

% record the weight after each iteration
weightArray = zeros(numPasses,1);


for pass=1:numPasses,
    weightRate = 0;
    tempLength = length(row);
    prm = randperm(tempLength);
    errRate = 0;
    
    for i=1:tempLength
        
        user = row(prm(i));
        item = col(prm(i));
        err = rating(prm(i)) - mu - P(user,:)*Q(item,:)' -Bu(user) -Bi(item);
        
        P(user,:) = P(user,:) + alpha*(err*Q(item,:) - lambda*P(user,:));
        Q(item,:) = Q(item,:) + alpha*(err*P(user,:) - lambda*Q(item,:));
        Bu(user) = Bu(user) + alpha*(err-lambda*Bu(user));
        Bi(item) = Bi(item) + alpha*(err-lambda*Bi(item));
    end
    
    for i=1:tempLength
        user = row(i);
        item = col(i);
        predictRating = mu + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
        err = rating(i) - predictRating;
        % fprintf('err:%d \t %e\n',i,err);
        if(err > 0.7 || err < -0.7)
            errRate = errRate + weight((i));
            %  fprintf('err:%d \t %e\n',i,err);
        end
    end
    
    errRate
    if(errRate > 0.5)
        break;
    end
    
    weightRate = (1/2)*log((1-errRate)/errRate) ;
    %weightRate
    weightArray(pass) = weightRate;
    %weightArray
    
    weightRight = exp(-weightRate);
    weightWrong = exp(weightRate);
    
    temp = (weightRight*(1 - errRate) + weightWrong*errRate)* numTrain
    weightRight = weightRight/temp;
    weightWrong = weightWrong/temp;
    
    for i=1:tempLength
        user = row(i);
        item = col(i);
        predictRating = mu + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
        err = rating(i) - predictRating;
        if(err > 0.7 || err < -0.7)
            weight(i) = weightWrong;
            %  fprintf('err:%d \t %e\n',i,err);
        else
            weight(i) = weightRight;
        end
    end
    
    weight
    
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
    fprintf('Test rmse pass %d: \t %e\n', pass,sqrt(errors/tempLength));
end



