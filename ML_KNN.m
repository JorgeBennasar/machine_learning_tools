function [target, X_train_update, Y_train_update] = ...
    ML_KNN(x, X_train, Y_train, k, norm_order, update)

% x: [n_modes, t_time]
% X_train = [n_modes, m_trials, t_time]
% Y_train = [target, m_trials]
% k: nearest neighbours
% norm_order: 1, 2, 3, 4...
% update: 'yes' or 'no'

persistent add;
persistent initial_m;

[n_modes, m_train, ~] = size(X_train);

if isempty(add)
    add = zeros(1,8);
    initial_m = m_train;
end

k_init = k;

D = zeros(1,m_train);
for i = 1:m_train
    for j = 1:n_modes
        A = transpose(x(j,:));
        B = squeeze(X_train(j,i,:)); 
        V = A - B;
        ND = norm(V,norm_order);
        D(i) = D(i) + ND;
    end
end

while k < initial_m/8
    knn = mink(D,k);
    targets = zeros(1,8);
    for i = 1:k
        j = find(D == knn(i));
        targets(Y_train(1,j)) = targets(Y_train(1,j)) + 1;
    end
    for i = 1:8
        targets(i) = fix(targets(i)*(initial_m/8)/ ...
            ((initial_m/8)+add(i)));
    end
    target = find(targets == max(targets));   
    if length(target) == 1
        break;
    end   
    k = k + 1;   
end

if length(target) > 1
    target = target(1);
end

target_cache = target;

for i = 1:8
    if i == 1
        lim_1 = 8;
    else
        lim_1 = i-1;
    end
    if i == 8
        lim_2 = 1;
    else
        lim_2 = i+1;
    end
    if target_cache == lim_1
        %if i == 3
        %    M = [4 18 22 31 33 34 36 44 54 55 69 75 77 85 90 97];
        %elseif i == 4
        %    M = [4 22 31 33 34 36 65 69 71 93 96];
        %elseif i == 5
        %    M = [2 5 31 33 44 59 66 68 69 71];
        %else
        %    M = linspace(1,n_modes,n_modes);
        %end
        M = linspace(1,n_modes,n_modes);
        k = k_init;
        if i == 1
            X_train_aux = X_train(:,[1:((initial_m/8)+add(1)) ...
                ((initial_m*7/8+1)+sum(add(1:7))):m_train],:);
            Y_train_aux = Y_train(:,[1:((initial_m/8)+add(1)) ...
                ((initial_m*7/8+1)+sum(add(1:7))):m_train]);
        else
            X_train_aux = X_train(:, ...
                ((initial_m*(i-2)/8+1)+sum(add(1:(i-2)))): ...
                ((initial_m*i/8)+sum(add(1:i))),:);
            Y_train_aux = Y_train(:,...
                ((initial_m*(i-2)/8+1)+sum(add(1:(i-2)))): ...
                ((initial_m*i/8)+sum(add(1:i))));
        end
        D = zeros(1,2*initial_m/8+add(i)+add(lim_1));
        for p = 1:(2*initial_m/8+add(i)+add(lim_1))
            for j = M
                A = transpose(x(j,:));
                B = squeeze(X_train_aux(j,p,:)); 
                V = A - B;
                ND = norm(V,norm_order);
                D(p) = D(p) + ND;
            end
        end

        while k < initial_m/8
            knn = mink(D,k);
            targets = zeros(1,8);
            for p = 1:k
                j = find(D == knn(p));
                targets(Y_train_aux(1,j)) = targets(Y_train_aux(1,j)) + 1;
            end
            for p = 1:8
                targets(p) = fix(targets(p)*(initial_m/8)/ ...
                    ((initial_m/8)+add(p)));
            end
            target = find(targets == max(targets));   
            if length(target) == 1
                break;
            end   
            k = k + 1;   
        end    

    elseif target_cache == lim_2
        %if i == 2
        %    M = [4 18 22 31 33 34 36 44 54 55 69 75 77 85 90 97];
        %elseif i == 3
        %    M = [4 22 31 33 34 36 65 69 71 93 96];
        %elseif i == 4
        %    M = [2 5 31 33 44 59 66 68 69 71];
        %else
        %    M = linspace(1,n_modes,n_modes);
        %end
        M = linspace(1,n_modes,n_modes);
        k = k_init;
        if i == 8
            X_train_aux = X_train(:,[1:((initial_m/8)+add(1)) ...
                ((initial_m*7/8+1)+sum(add(1:7))):m_train],:);
            Y_train_aux = Y_train(:,[1:((initial_m/8)+add(1)) ...
                ((initial_m*7/8+1)+sum(add(1:7))):m_train]);
        else
            X_train_aux = X_train(:, ...
                ((initial_m*(i-1)/8+1)+sum(add(1:(i-1)))): ...
                ((initial_m*(i+1)/8)+sum(add(1:(i+1)))),:);
            Y_train_aux = Y_train(:,...
                ((initial_m*(i-1)/8+1)+sum(add(1:(i-1)))): ...
                ((initial_m*(i+1)/8)+sum(add(1:(i+1)))));
        end
        D = zeros(1,2*initial_m/8+add(i)+add(lim_2));
        for p = 1:(2*initial_m/8+add(i)+add(lim_2))
            for j = M
                A = transpose(x(j,:));
                B = squeeze(X_train_aux(j,p,:)); 
                V = A - B;
                ND = norm(V,norm_order);
                D(p) = D(p) + ND;
            end
        end

        while k < initial_m/8
            knn = mink(D,k);
            targets = zeros(1,8);
            for p = 1:k
                j = find(D == knn(p));
                targets(Y_train_aux(1,j)) = targets(Y_train_aux(1,j)) + 1;
            end
            for p = 1:8
                targets(p) = fix(targets(p)*(initial_m/8)/ ...
                    ((initial_m/8)+add(p)));
            end
            target = find(targets == max(targets));   
            if length(target) == 1
                break;
            end   
            k = k + 1;   
        end    
    end   
end

if length(target) > 1
    target = target(1);
end

if strcmp(update,'yes') 
    X_train_update = zeros(size(X_train,1),size(X_train,2)+1, ...
        size(X_train,3));
    Y_train_update = zeros(size(Y_train,1),size(Y_train,2)+1);
    X_train_update(:,1:(initial_m*target/8+ ...
        sum(add(1:target))),:) = X_train(:,1: ...
        (initial_m*target/8+sum(add(1:target))),:);
    Y_train_update(:,1:(initial_m*target/8+ ...
        sum(add(1:target)))) = Y_train(:,1:(initial_m*target/8+ ...
        sum(add(1:target))));
    X_train_update(:,(initial_m*target/8+ ...
        sum(add(1:target)))+1,:) = x;
    Y_train_update(:,(initial_m*target/8+ ...
        sum(add(1:target)))+1) = target;
    X_train_update(:,(initial_m*target/8+ ...
        sum(add(1:target))+2):end,:) = X_train(:,(initial_m*target/8+ ...
        sum(add(1:target))+1):end,:);
    Y_train_update(:,(initial_m*target/8+ ...
        sum(add(1:target))+2):end) = Y_train(:,(initial_m*target/8+ ...
        sum(add(1:target))+1):end);
    add(target) = add(target) + 1;
else
    X_train_update = X_train;
    Y_train_update = Y_train;
end

end