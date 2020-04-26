function [w,error] = train_perceptron(x,y_target,bias,alpha,iterations)

[n,p] = size(x);
epsilon = 1e-08;
w = zeros(1,n);
error = zeros(1,iterations);

for i = 1:iterations
    y_pred = heaviside(w*x - bias + epsilon);
    dw = alpha*x*transpose(y_target - y_pred);
    w = w + transpose(dw);
    error(i) = sum(abs(y_target - y_pred))/p;
end

end