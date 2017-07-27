function [ emd ] = EMD(amp1, f1, amp2, f2, noiseCancellationParameter)
    %Important - The input have to be column vectors
    %EMD for two signals based on their frequency spectrum 
    %Inputs are amplitudes and frequencies
    %Two signals need not have same size
    %Remove the additive noise in the signal and add it to the array W 
    %preallocate the array X
    W1 = zeros(size(amp1)); %store the amplitudes as weights
    X = zeros(size(f1)); %store the freq. as the data points which need to be moved
    count = 1;
    for j=1:length(amp1)
       if(amp1(j) <= noiseCancellationParameter)
           continue;
       else
           W1(count) = amp1(j) - noiseCancellationParameter;
           X(count) = f1(j);
           count = count + 1;
       end
    end
    %Now truncate the array to required size
    index = find(W1 == 0,1,'first');
    
    %this step is important because, sometimes, element zero may not be
    %present in the weigths
    if(~isempty(index))
        W1 = W1(1:index-1);
        X = X(1:index-1);
    end
    
   
    W2 = zeros(size(amp2));
    Y = zeros(size(f2));
    
    count = 1;
    for j=1:length(amp2)
       if(amp2(j) <= noiseCancellationParameter)
           continue;
       else
           W2(count) = amp2(j) - noiseCancellationParameter;
           Y(count) = f2(j);
           count = count + 1;
       end
    end
    %Now truncate the array to required size
    index = find(W2 == 0,1,'first');
    
    if(~isempty(index))
        W2 = W2(1:index-1);
    Y = Y(1:index-1);
    end
    
    %Calculate the distance matrix
    %We will be using the euclidean distance as the ground distance
    %Since, what we have is 1-dimensional, we get simply |X(i) - Y(j)|
    %[m,~] = size(X);
    m = numel(X);
    %[n,~] = size(Y);
    n = numel(Y);
    D = zeros(m,n);
    for i=1:m
        %disp('size');
        %disp(n);
        for j=1:n
            D(i,j) = abs(X(i) - Y(j));
        end
    end
    D = D';
    D = D(:);
    %str = strcat('m = ',num2str(m),',n = ',num2str(n));
    %disp(str);
    % inequality constraints
    A1 = zeros(m, m * n);
    A2 = zeros(n, m * n);
    for i = 1:m
        for j = 1:n
            k = j + (i - 1) * n;
            A1(i, k) = 1;
            A2(j, k) = 1;
        end
    end
    A = [A1; A2];
    b = [W1; W2];
    
    %equality constraints
    Aeq = ones(m + n, m * n);
    beq = ones(m + n, 1) * min(sum(W1), sum(W2));
    
    %lower bound
    %lb = zeros(1, m * n);
    lb = zeros(m*n,1);
    t = m*n;
    %linear programming
    %case where m = n = 0. Note that m and n are positive integers
    if(m+n == 0)
        emd = 0;
    %this means either m or n is 0 but not both in which case emd is Inf
    elseif(and(m + n > 0, m*n==0) == 1)
        emd = Inf;
    else
        %[x, fval, exitflag, output] = linprog(D, A, b, Aeq, beq, lb);
        %x is the flow value;
        %emd1 = fval / sum(x);
        %disp(sum(x));
        %disp(fval);
        if(D == 0)
            emd = 0;
        else    
            cvx_begin quiet
                variable x(t);
                minimize (D'*x);
                subject to
                    A*x <= b;
                    Aeq*x == beq;
                    x >= lb;
            cvx_end
            emd = D'*x/sum(x);
            if(isnan(emd))
                emd = 0;
            end
        end
        %disp(sum(x));
        %disp(D'*x);
    end
end

