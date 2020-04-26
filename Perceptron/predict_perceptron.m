function y_pred = predict_perceptron(x,w,bias)

epsilon = 1e-08;
y_pred = heaviside(w*x - bias + epsilon);

end