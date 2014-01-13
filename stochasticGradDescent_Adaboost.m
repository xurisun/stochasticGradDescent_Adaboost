function stochasticGradDescent_Adaboost
Y = load('20data.dat');% 225084
p = randperm(length(Y));
Y(:,1) = Y(p,1);
Y(:,2) = Y(p,2);
Y(:,3) = Y(p,3);
numTrans = length(Y);

first=1;
testSize=floor(numTrans/5);
last = first+testSize-1;

testY = Y(first:last,:);%45016
trainY = [Y(1:(first-1),:);Y((last+1):end,:)]; % 180068
weight = 

trainSet = sparse(trainY(:,1),trainY(:,2),trainY(:,3));
testSet = sparse(testY(:,1),testY(:,2),testY(:,3));

k=20;
%% Training Phase --

[ P,Q ] =model1(full(trainSet),k,testSet);

function [ P,Q ] = model1( L,k,testSet)
%save configs2;
trainSet = sparse(L);
nusers = size(L,1);
nitems = size(L,2);

mu = sum(L(:)) / nnz(L);
P = rand(nusers,k)/100-0.005;
Q = rand(nitems,k)/100-0.005;

[row,col,rating] = find(trainSet);
[rowT,colT,ratingT] = find(testSet);

lambda = 0.02; % penalty term
alpha = 0.005;  % learning rate
numPasses = 1000;

Bu = zeros(nusers,1);
Bi = zeros(nitems,1);

%preTestRmse=100;
for pass=1:numPasses,
    tempLength = length(row);
    prm = randperm(tempLength);
    for i=1:tempLength
        user = row(prm(i));
        item = col(prm(i));
        err = rating(prm(i)) - mu - P(user,:)*Q(item,:)' -Bu(user) -Bi(item);
        P(user,:) = P(user,:) + alpha*(err*Q(item,:) - lambda*P(user,:));
        Q(item,:) = Q(item,:) + alpha*(err*P(user,:) - lambda*Q(item,:));
        Bu(user) = Bu(user) + alpha*(err-lambda*Bu(user));
        Bi(item) = Bi(item) + alpha*(err-lambda*Bi(item));
        
    end
    
    errors=0;
    for i=1:tempLength
        user=row(i);
        item=col(i);
        predictRating = mu + P(user,:)*Q(item,:)' + Bu(user) + Bi(item);
        err=rating(i) - predictRating;
        errors=errors + err^2;
    end
    
    fprintf('Training rmse pass %d: \t %e\n', pass,sqrt(errors/tempLength));
    
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
    if (pass == 30)
        break;
    end
end

