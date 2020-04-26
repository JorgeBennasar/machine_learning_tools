function [x_isomap,exp_var_isomap] = get_isomap(x,num_dims)

iso_n = 12;   
iso_function = 'k';    
iso_opts = struct('dims',1:num_dims,'comp',1,'display',false, ...
    'overlay',true,'verbose',true);

x_concat = zeros(size(x,2)*size(x,3),size(x,1));
for i = 1:size(x,1)
    for j = 1:size(x,2)
        for k = 1:size(x,3)
            x_concat((j-1)*size(x,3)+k+1,i) = x(i,j,k);
        end
    end
end

mu = mean(x_concat,1);
x_concat = x_concat - repmat(mu,size(x_concat,1),1);

D = L2_distance(x_concat',x_concat');

[Y,R,~,~] = isomap(D,iso_function,iso_n,iso_opts);

scores = Y.coords{end}';

x_isomap = zeros(num_dims,size(x,2),size(x,3));
for i = 1:size(x,2)
    for j = 1:size(x,3)
        for k = 1:num_dims
            x_isomap(k,i,j) = scores((i-1)*size(x,3)+j+1,k);
        end
    end
end

exp_var_isomap = zeros(1,num_dims);
for i = 1:num_dims
    exp_var_isomap(i) = 1 - R(i);
end
    
end